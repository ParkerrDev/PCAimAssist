import torch
import numpy as np
import cv2
import time
import win32api
import win32con
import pandas as pd
from utils.general import (cv2, non_max_suppression, xyxy2xywh)
from models.common import DetectMultiBackend
import cupy as cp
import threading
import queue

# Could be do with
# from config import *
# But we are writing it out for clarity for new devs
from config import aaMovementAmp, useMask, maskHeight, maskWidth, aaQuitKey, confidence, headshot_mode, cpsDisplay, visuals, centerOfScreen, screenShotWidth, ThirdPerson
import gameSelection

def capture_frames(camera, frame_queue):
    while True:
        frame = camera.get_latest_frame()
        if frame is not None:
            try:
                # Always replace the oldest frame with the newest
                if frame_queue.full():
                    frame_queue.get_nowait()
                frame_queue.put(frame, timeout=0.1)  # Add frame to queue
            except queue.Full:
                pass  # Queue is full, drop the frame

def process_frames(frame_queue, model, names, confidence, cWidth, cHeight, last_mid_coord):
    last_mid_coord = last_mid_coord
    COLORS = np.random.uniform(0, 255, size=(1500, 3))
    count = 0
    sTime = time.time()
    target_fps = 60
    process_interval = 1 / target_fps
    last_process_time = time.time()
    last_trigger_time = 0
    TRIGGER_THRESHOLD = 25  # pixels

    # Target Locking Variables
    locked_target = None
    lock_start_time = 0
    LOCK_DURATION = 1  # seconds
    TARGET_COOLDOWN = 0.75  # seconds
    LOCK_BREAK_DISTANCE = 100  # pixels
    INITIAL_TARGET_SIDE = None  # Track which side target was first acquired on
    AUTO_FIRE_DELAY = 0.1  # seconds before auto-firing starts
    AUTO_FIRE_INTERVAL = 0.05  # seconds between auto-fires
    last_auto_fire_time = 0  # track last auto-fire time
    is_mouse_down = False  # Track mouse button state
    
    # Variables for aiming prediction based on target movement
    prev_locked = None
    prev_lock_time = time.time()
    tracking_velocity = [0, 0]  # Store target velocity
    
    def fire_once():
        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
        time.sleep(0.05)  # Small delay between down and up
        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
    
    while True:
        current_time = time.time()
        if current_time - last_process_time >= process_interval:
            try:
                frame = frame_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            last_process_time = current_time
            npImg = cp.array([frame])
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

            im = npImg / 255
            im = im.astype(cp.half)
            im = cp.moveaxis(im, 3, 1)
            im = torch.from_numpy(cp.asnumpy(im)).to('cuda')

            results = model(im)
            # Only apply different thresholds if in third person mode
            base_confidence = confidence
            right_side_confidence = confidence * 1.5 if ThirdPerson else confidence

            # Apply different confidence thresholds based on position and game mode
            pred = []
            for det in non_max_suppression(results, base_confidence, base_confidence, 0, False, max_det=10):
                filtered_det = []
                for *xyxy, conf, cls in det:
                    # Only apply position-based threshold in third person mode
                    if ThirdPerson:
                        x_center = (xyxy[0] + xyxy[2]) / 2
                        screen_ratio = x_center / im.shape[3]
                        threshold = right_side_confidence if screen_ratio > 0.5 else base_confidence
                    else:
                        threshold = base_confidence
                    
                    if conf >= threshold:
                        filtered_det.append((*xyxy, conf, cls))
                pred.append(torch.tensor(filtered_det) if filtered_det else torch.empty((0, 6)))

            targets = []
            for i, det in enumerate(pred):
                if len(det):
                    for c in det[:, -1].unique():
                        pass
                    for *xyxy, conf, cls in reversed(det):
                        targets.append((xyxy2xywh(torch.tensor(xyxy).view(1, 4)) /
                                        torch.tensor(im.shape)[[0, 0, 0, 0]]).view(-1).tolist() + [float(conf)])
            targets = pd.DataFrame(targets, columns=['current_mid_x', 'current_mid_y', 'width', "height", "confidence"])
            
            center_screen = [cWidth, cHeight]

            # Target Selection and Locking Logic
            if locked_target is not None:
                if current_time - lock_start_time >= LOCK_DURATION:
                    locked_target = None
                    prev_locked = None
                    INITIAL_TARGET_SIDE = None
                else:
                    if len(targets) > 0:
                        # Adjust break distance based on initial target side
                        current_break_distance = LOCK_BREAK_DISTANCE
                        if INITIAL_TARGET_SIDE == "right":
                            current_break_distance *= 1.5  # Allow more movement for targets that started on right
                        
                        distances = ((targets['current_mid_x'] - locked_target[0])**2 + 
                                   (targets['current_mid_y'] - locked_target[1])**2).apply(np.sqrt)
                        closest_idx = distances.idxmin()
                        
                        if distances[closest_idx] > current_break_distance:
                            locked_target = None
                            prev_locked = None
                            INITIAL_TARGET_SIDE = None
                        else:
                            # Update locked target position with improved prediction
                            xMid = targets.iloc[closest_idx].current_mid_x
                            yMid = targets.iloc[closest_idx].current_mid_y
                            
                            if prev_locked is not None:
                                dt = current_time - prev_lock_time
                                if dt > 0:
                                    # Update velocity with smoothing
                                    new_vx = (xMid - prev_locked[0]) / dt
                                    new_vy = (yMid - prev_locked[1]) / dt
                                    tracking_velocity[0] = 0.7 * tracking_velocity[0] + 0.3 * new_vx
                                    tracking_velocity[1] = 0.7 * tracking_velocity[1] + 0.3 * new_vy
                                    
                                    # Adjust lead time based on movement direction
                                    lead_time = 0.15 if tracking_velocity[0] < 0 else 0.1  # More prediction for targets moving left
                                    pred_x = xMid + tracking_velocity[0] * lead_time
                                    pred_y = yMid + tracking_velocity[1] * lead_time
                                else:
                                    pred_x, pred_y = xMid, yMid
                            else:
                                pred_x, pred_y = xMid, yMid
                            
                            prev_locked = [xMid, yMid]
                            prev_lock_time = current_time
                            locked_target = [pred_x, pred_y]
                            
                            # Movement Logic
                            box_height = targets.iloc[closest_idx].height
                            headshot_offset = box_height * (0.3 if headshot_mode else 0.25)
                            
                            error_x = int(pred_x - cWidth)
                            error_y = int((pred_y - headshot_offset) - cHeight)
                            move_x = int(error_x * 0.3 * aaMovementAmp)
                            move_y = int(error_y * 0.3 * aaMovementAmp)
                            
                            if win32api.GetKeyState(0x14):
                                win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, move_x, move_y, 0, 0)
                            last_mid_coord = [pred_x, pred_y]
            
            # If no locked target, find new target
            if locked_target is None and len(targets) > 0:
                if len(targets) == 1:
                    target = None
                    if ThirdPerson:
                        # Third person specific logic
                        if targets.iloc[0].current_mid_x > cWidth:
                            target = targets.iloc[0]
                            INITIAL_TARGET_SIDE = "right"
                    else:
                        # First person logic - accept targets from both sides
                        target = targets.iloc[0]
                        INITIAL_TARGET_SIDE = "right" if targets.iloc[0].current_mid_x > cWidth else "left"
                else:
                    # For multiple targets
                    targets = targets.sort_values(by='current_mid_y', ascending=True)
                    if ThirdPerson:
                        # Filter for right side targets in third person
                        right_side_targets = targets[targets['current_mid_x'] > cWidth]
                        if not right_side_targets.empty:
                            target = right_side_targets.iloc[0]
                            INITIAL_TARGET_SIDE = "right"
                        else:
                            target = None
                    else:
                        # Use any target in first person
                        target = targets.iloc[0]
                        INITIAL_TARGET_SIDE = "right" if target.current_mid_x > cWidth else "left"

                if target is not None:
                    locked_target = [target.current_mid_x, target.current_mid_y]
                    prev_locked = [target.current_mid_x, target.current_mid_y]
                    prev_lock_time = current_time
                    lock_start_time = current_time
                    tracking_velocity = [0, 0]  # Reset velocity tracking
                    
                    # Movement Logic
                    box_height = target.height
                    headshot_offset = box_height * (0.28 if headshot_mode else 0.1)
                    
                    error_x = target.current_mid_x - cWidth
                    error_y = (target.current_mid_y - headshot_offset) - cHeight
                    move_x = int(error_x * 0.3 * aaMovementAmp)
                    move_y = int(error_y * 0.3 * aaMovementAmp)
                    
                    if win32api.GetKeyState(0x14):
                        win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, move_x, move_y, 0, 0)
                    last_mid_coord = [target.current_mid_x, target.current_mid_y]

            # Enhanced Trigger Logic
            if locked_target is not None:
                current_distance = ((locked_target[0] - cWidth) ** 2 + (locked_target[1] - cHeight) ** 2) ** 0.5
                lock_duration = current_time - lock_start_time
                caps_lock_enabled = win32api.GetKeyState(0x14)  # Check Caps Lock state
                
                # Check if we should auto-fire (only when Caps Lock is enabled)
                if caps_lock_enabled and lock_duration >= AUTO_FIRE_DELAY:
                    if current_time - last_auto_fire_time >= AUTO_FIRE_INTERVAL:
                        fire_once()
                        last_auto_fire_time = current_time
                # Regular close-range trigger logic (only when Caps Lock is enabled)
                elif caps_lock_enabled and current_distance < TRIGGER_THRESHOLD and (current_time - last_trigger_time) >= TARGET_COOLDOWN:
                    fire_once()
                    last_trigger_time = current_time

            if visuals:
                dispImg = cp.asnumpy(npImg[0])
                for i in range(len(targets)):
                    halfW = round(targets["width"][i] / 2)
                    halfH = round(targets["height"][i] / 2)
                    midX = targets['current_mid_x'][i]
                    midY = targets['current_mid_y'][i]
                    (startX, startY, endX, endY) = int(midX + halfW), int(midY + halfH), int(midX - halfW), int(midY - halfH)
                    idx = 0
                    label = "{}: {:.2f}%".format("Human", targets["confidence"][i] * 100)
                    cv2.rectangle(dispImg, (startX, startY), (endX, endY), COLORS[idx], 2)
                    yLoc = startY - 15 if startY - 15 > 15 else startY + 15
                    cv2.putText(dispImg, label, (startX, yLoc), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
                cv2.imshow('Live Feed', dispImg)
                if (cv2.waitKey(1) & 0xFF) == ord('q'):
                    exit()

            count += 1
            if (time.time() - sTime) > 1:
                if cpsDisplay:
                    print("CPS: {}".format(count))
                count = 0
                sTime = time.time()
        # end while iteration
    return last_mid_coord

def main():
    # External Function for running the game selection menu (gameSelection.py)
    camera, cWidth, cHeight = gameSelection.gameSelection()

    # Loading Yolo5 Small AI Model
    model = DetectMultiBackend('yolov5x640.engine', device=torch.device(
        'cuda'), dnn=False, data='', fp16=True)
    stride, names, pt = model.stride, model.names, model.pt

    # Initialize frame queue
    frame_queue = queue.Queue(maxsize=1)  # Queue for frames

    # Start the capture thread
    capture_thread = threading.Thread(target=capture_frames, args=(camera, frame_queue), daemon=True)
    capture_thread.start()

    # Initialize last_mid_coord
    last_mid_coord = None

    # Start the processing thread
    process_thread = threading.Thread(target=process_frames, args=(frame_queue, model, names, confidence, cWidth, cHeight, last_mid_coord), daemon=True)
    process_thread.start()

    # Keep the main thread alive until the exit key is pressed
    while win32api.GetAsyncKeyState(ord(aaQuitKey)) == 0:
        #time.sleep(0.01)
        True

    camera.stop()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        traceback.print_exception(e)
        print("ERROR: " + str(e))
        print("Ask @Wonder for help in our Discord in the #ai-aimbot channel ONLY: https://discord.gg/rootkitorg")