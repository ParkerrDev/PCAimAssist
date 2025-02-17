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
import asyncio  # NEW
import psutil
import os

# Could be do with
# from config import *
# But we are writing it out for clarity for new devs
from config import (
    aaMovementAmp, useMask, maskHeight, maskWidth,
    aaQuitKey, confidence, headshot_mode, cpsDisplay,
    visuals, screenShotWidth, screenShotHeight,
    ThirdPerson
)
import gameSelection

def capture_frame_from_obs(ws, width, height):
    try:
        # Get current scene and screenshot at the desired dimensions (640x640)
        current_scene = ws.get_current_program_scene()
        screenshot = ws.get_source_screenshot(
            name=current_scene.current_program_scene_name,
            img_format="jpg",
            width=width,
            height=height,
            quality=100
        )
        img_data = screenshot.image_data
        if img_data.startswith("data:image/jpg;base64,"):
            img_data = img_data.replace("data:image/jpg;base64,", "")
        padding = len(img_data) % 4
        if padding:
            img_data += "=" * (4 - padding)
        img_bytes = base64.b64decode(img_data)
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Only resize if dimensions do not match exactly
        if img.shape[1] != width or img.shape[0] != height:
            img = cv2.resize(img, (width, height))
        
        # Combine color conversion (BGR to RGB) and normalization (0-255 to 0-1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        
        return img
    except Exception as e:
        print(f"Error capturing frame from OBS: {e}")
        print("Make sure you have a scene selected in OBS")
        return None

async def main():
    # Set process priority to high
    try:
        process = psutil.Process(os.getpid())
        process.nice(psutil.HIGH_PRIORITY_CLASS)
    except Exception as e:
        print(f"Failed to set process priority: {e}")

    # Attempt to enable high performance power mode (Windows-specific)
    try:
        if hasattr(win32api, 'SetProcessPowerThrottling'):
            win32api.SetProcessPowerThrottling(win32api.GetCurrentProcess(), win32con.PROCESS_POWER_THROTTLING_EXECUTION_SPEED, 0)
    except Exception as e:
        print(f"Failed to set high performance power mode: {e}")

    ws, cWidth, cHeight = await gameSelection.gameSelection()
    if ws is None:
        print("Failed to initialize OBS connection.")
        return

    model = DetectMultiBackend('yolov5m.engine', device=torch.device('cuda'), dnn=False, data='', fp16=True)
    stride, names, pt = model.stride, model.names, model.pt
    last_mid_coord = None

    # Schedule the asynchronous frame-processing task
    try:
        await gameSelection.process_frames(ws, model, cWidth, cHeight, last_mid_coord)
    finally:
        await ws.disconnect()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        import traceback
        traceback.print_exception(e)
        print("ERROR: " + str(e))