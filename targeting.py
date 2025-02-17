import time
import win32api
import win32con
import numpy as np
import pandas as pd
import asyncio

# Import necessary variables from config
from config import (
    aaMovementAmp, headshot_mode,
    ThirdPerson, xOffset, yOffset
)

def fire_once():
    # Remove sleep between mouse down/up for faster firing
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)

async def move_mouse(x, y):
    win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, x, y, 0, 0)

async def process_targets(targets: pd.DataFrame, cWidth, cHeight,
                    locked_target, lock_start_time, prev_locked, prev_lock_time,
                    tracking_velocity, INITIAL_TARGET_SIDE, last_mid_coord,
                    last_trigger_time, last_auto_fire_time):
    TRIGGER_THRESHOLD = 25  # pixels
    LOCK_DURATION = 1  # seconds
    TARGET_COOLDOWN = 0.75  # seconds
    LOCK_BREAK_DISTANCE = 100  # pixels
    AUTO_FIRE_DELAY = 0.1  # seconds before auto-firing starts
    AUTO_FIRE_INTERVAL = 0.05  # seconds between auto-fires
    current_time = time.time()
    
    # Target Selection and Locking Logic
    if locked_target is not None:
        if current_time - lock_start_time >= LOCK_DURATION:
            locked_target = None
            prev_locked = None
            INITIAL_TARGET_SIDE = None
        else:
            if not targets.empty:
                # Adjust break distance based on initial target side
                current_break_distance = LOCK_BREAK_DISTANCE
                if INITIAL_TARGET_SIDE == "right":
                    current_break_distance *= 1.5  # Allow more movement for targets that started on right
                
                distances = ((targets['current_mid_x'] - locked_target[0])**2 + 
                           (targets['current_mid_y'] - locked_target[1])**2).apply(np.sqrt) # Potential Bottleneck: distance calculation
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
                    
                    if win32api.GetKeyState(0x14): # Potential Bottleneck: win32api calls can be slow
                        await move_mouse(move_x, move_y) # Potential Bottleneck: async mouse movement
                    last_mid_coord = [pred_x, pred_y]
    
    # If no locked target, find new target
    if locked_target is None and not targets.empty:
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
            targets = targets.sort_values(by='current_mid_y', ascending=True) # Potential Bottleneck: sorting
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
            
            if win32api.GetKeyState(0x14): # Potential Bottleneck: win32api calls can be slow
                await move_mouse(move_x, move_y) # Potential Bottleneck: async mouse movement
            last_mid_coord = [target.current_mid_x, target.current_mid_y]

    # Enhanced Trigger Logic
    if locked_target is not None:
        current_distance = ((locked_target[0] - cWidth) ** 2 + (locked_target[1] - cHeight) ** 2) ** 0.5
        lock_duration = current_time - lock_start_time
        caps_lock_enabled = win32api.GetKeyState(0x14)  # Check Caps Lock state # Potential Bottleneck: win32api calls can be slow
        
        # Check if we should auto-fire (only when Caps Lock is enabled)
        if caps_lock_enabled and lock_duration >= AUTO_FIRE_DELAY:
            if current_time - last_auto_fire_time >= AUTO_FIRE_INTERVAL:
                fire_once() # Potential Bottleneck: firing function
                last_auto_fire_time = current_time
        # Regular close-range trigger logic (only when Caps Lock is enabled)
        elif caps_lock_enabled and current_distance < TRIGGER_THRESHOLD and (current_time - last_trigger_time) >= TARGET_COOLDOWN:
            fire_once() # Potential Bottleneck: firing function
            last_trigger_time = current_time
    
    return locked_target, lock_start_time, prev_locked, prev_lock_time, tracking_velocity, INITIAL_TARGET_SIDE, last_mid_coord, last_trigger_time, last_auto_fire_time
