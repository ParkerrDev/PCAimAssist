import simpleobsws
import numpy as np
import cv2
import torch
import cupy as cp
import pandas as pd
import time
from PIL import Image
import io
import asyncio
import base64  # Import base64
from models.common import DetectMultiBackend

# Import necessary variables from config
from config import (
    useMask,
    maskHeight,
    maskWidth,
    cpsDisplay,
    visuals,
    screenShotWidth,
    screenShotHeight,
    ThirdPerson,
    confidence,
    obs_host,  # Add this
    obs_port,  # Add this
)
import targeting

async def connect_obs():
    parameters = simpleobsws.IdentificationParameters(
        ignoreNonFatalRequestChecks = False
    )
    ws = simpleobsws.WebSocketClient(
        url=f"ws://{obs_host}:{obs_port}",
        password=None,  # Add password if needed
        identification_parameters=parameters
    )
    
    await ws.connect()
    await ws.wait_until_identified()
    return ws

async def capture_frame_from_obs(ws, width, height, source_name=None):
    try:
        if source_name is None:
            # Get current scene
            scene_request = simpleobsws.Request('GetCurrentProgramScene')
            scene_result = await ws.call(scene_request)
            source_name = scene_result.responseData['currentProgramSceneName']

        # Get screenshot using the more efficient simpleobsws method
        request = simpleobsws.Request('GetSourceScreenshot', {
            'sourceName': source_name,
            'imageFormat': 'jpg',
            'imageWidth': width,
            'imageHeight': height,
            'imageCompressionQuality': 85
        })

        result = await ws.call(request)
        if not result.ok():
            print(f"OBS request failed: {result.error}")
            return None

        # Process image data
        img_data = result.responseData['imageData']
        if ',' in img_data:
            img_data = img_data.split(',', 1)[1]

        # Decode base64 directly to numpy array
        img_bytes = base64.b64decode(img_data)
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Convert to RGB float32
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        return img

    except Exception as e:
        print(f"Error capturing frame from OBS: {e}")
        return None

async def process_frames(ws, model, cWidth, cHeight, last_mid_coord):
    last_mid_coord = last_mid_coord
    COLORS = np.random.uniform(0, 255, size=(1500, 3))
    count = 0
    sTime = time.time()
    last_trigger_time = 0

    # Target Locking Variables
    locked_target = None
    lock_start_time = 0
    INITIAL_TARGET_SIDE = None  # Track which side target was first acquired on
    last_auto_fire_time = 0  # track last auto-fire time

    # Variables for aiming prediction based on target movement
    prev_locked = None
    prev_lock_time = time.time()
    tracking_velocity = [0, 0]  # Store target velocity

    while True:
        frame = await capture_frame_from_obs(ws, screenShotWidth, screenShotHeight)
        if frame is None:
            await asyncio.sleep(0.001)  # Minimal sleep to prevent CPU thrashing
            continue

        # Create single batch numpy array
        npImg = cp.array([frame])  # Frame is already RGB, no conversion needed

        if npImg.shape[3] == 4:
            npImg = npImg[:, :, :, :3]

        from config import maskSide

        if useMask:
            maskSide = maskSide.lower()
            if maskSide == "right":
                npImg[:, -maskHeight:, -maskWidth:, :] = 0
            elif maskSide == "left":
                npImg[:, -maskHeight:, :maskWidth, :] = 0
            else:
                raise Exception('ERROR: Invalid maskSide! Please use "left" or "right"')

        # Removed the redundant division by 255
        im = npImg.astype(cp.half)
        im = cp.moveaxis(im, 3, 1)
        im = torch.from_numpy(cp.asnumpy(im)).to("cuda")

        from utils.general import cv2, non_max_suppression, xyxy2xywh

        results = model(im)
        # Only apply different thresholds if in third person mode
        base_confidence = confidence
        right_side_confidence = confidence * 1.5 if ThirdPerson else confidence

        # Apply different confidence thresholds based on position and game mode
        pred = []
        for det in non_max_suppression(
            results, base_confidence, base_confidence, 0, False, max_det=10
        ):
            filtered_det = []
            for *xyxy, conf, cls in det:
                # Only apply position-based threshold in third person mode
                if ThirdPerson:
                    x_center = (xyxy[0] + xyxy[2]) / 2
                    screen_ratio = x_center / im.shape[3]
                    threshold = (
                        right_side_confidence if screen_ratio > 0.5 else base_confidence
                    )
                else:
                    threshold = base_confidence

                if conf >= threshold:
                    filtered_det.append((*xyxy, conf, cls))
            pred.append(
                torch.tensor(filtered_det) if filtered_det else torch.empty((0, 6))
            )

        targets = []
        for i, det in enumerate(pred):
            if len(det):
                for c in det[:, -1].unique():
                    pass
                for *xyxy, conf, cls in reversed(det):
                    targets.append(
                        (
                            xyxy2xywh(torch.tensor(xyxy).view(1, 4))
                            / torch.tensor(im.shape)[[0, 0, 0, 0]]
                        )
                        .view(-1)
                        .tolist()
                        + [float(conf)]
                    )
        targets = pd.DataFrame(
            targets,
            columns=["current_mid_x", "current_mid_y", "width", "height", "confidence"],
        )

        center_screen = [cWidth, cHeight]

        # Call the targeting logic from targeting.py
        (
            locked_target,
            lock_start_time,
            prev_locked,
            prev_lock_time,
            tracking_velocity,
            INITIAL_TARGET_SIDE,
            last_mid_coord,
            last_trigger_time,
            last_auto_fire_time,
        ) = await targeting.process_targets(
            targets,
            cWidth,
            cHeight,
            locked_target,
            lock_start_time,
            prev_locked,
            prev_lock_time,
            tracking_velocity,
            INITIAL_TARGET_SIDE,
            last_mid_coord,
            last_trigger_time,
            last_auto_fire_time,
        )

        if visuals:
            # Convert RGB back to BGR for OpenCV display
            dispImg = cv2.cvtColor(cp.asnumpy(npImg[0]), cv2.COLOR_RGB2BGR)

            # Draw detections
            for i in range(len(targets)):
                halfW = round(targets["width"][i] / 2)
                halfH = round(targets["height"][i] / 2)
                midX = targets["current_mid_x"][i]
                midY = targets["current_mid_y"][i]
                (startX, startY, endX, endY) = (
                    int(midX + halfW),
                    int(midY + halfH),
                    int(midX - halfW),
                    int(midY - halfH),
                )
                idx = 0
                label = "{}: {:.2f}%".format("Human", targets["confidence"][i] * 100)
                cv2.rectangle(dispImg, (startX, startY), (endX, endY), COLORS[idx], 2)
                yLoc = startY - 15 if startY - 15 > 15 else startY + 15
                cv2.putText(
                    dispImg,
                    label,
                    (startX, yLoc),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    COLORS[idx],
                    2,
                )

            # Display the BGR frame
            cv2.imshow("Live Feed", dispImg)

            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                exit()

        count += 1
        if (time.time() - sTime) > 1:
            if cpsDisplay:
                print("CPS: {}".format(count))
            count = 0
            sTime = time.time()
    return last_mid_coord

async def gameSelection():
    try:
        ws = await connect_obs()
        cWidth = 640 // 2
        cHeight = 640 // 2
        return ws, cWidth, cHeight
    except Exception as e:
        print(f"Failed to connect to OBS: {e}")
        return None, None, None
