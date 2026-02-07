import cv2
import time
import math
import numpy as np
from ultralytics import YOLO
import config

class AdaptiveVideoAnalytics:
    def __init__(self, source=config.VIDEO_SOURCE):
        
        # --- LOAD MODELS ---
        print(f"Loading Light Model ({config.MODEL_LIGHT})...")
        self.model_light = YOLO(config.MODEL_LIGHT)
        
        print(f"Loading Heavy Model ({config.MODEL_HEAVY})...")
        self.model_heavy = YOLO(config.MODEL_HEAVY) 

        self.cap = cv2.VideoCapture(source)
        
        # --- STATE VARIABLES ---
        self.current_model_name = "Light (v8n)"
        self.current_res = config.RES_HIGH
        self.bandwidth_level = 100 # Starts at 100%
        self.frame_count = 0
        
    def get_simulated_bandwidth(self):
        # SIMULATION: Creates a sine wave that goes up and down over time
        # Period is approx 100 frames (roughly 3-4 seconds)
        val = (math.sin(self.frame_count * 0.05) + 1) / 2 # Returns 0.0 to 1.0
        return int(val * 100) # Returns percentage (0% to 100%)

    def process_stream(self):
        prev_time = 0
        
        # VARIABLES FOR FRAME SKIPPING
        frame_skip_interval = 3  # Process only every 3rd frame
        last_results = None      # Store the last detection result
        
        while True:
            ret, frame = self.cap.read()
            if not ret: break
            self.frame_count += 1
            
            # --- 1. RESIZE (Keep it small for speed) ---
            self.bandwidth_level = self.get_simulated_bandwidth()
            
            if self.bandwidth_level < 40:
                target_width = config.RES_LOW
            else:
                target_width = config.RES_HIGH
            
            # [[[ FIX: UPDATE THE DASHBOARD VARIABLE ]]]
            self.current_res = target_width 
            
            # Standard resizing
            h, w = frame.shape[:2]
            scale = target_width / w
            frame_resized = cv2.resize(frame, (int(w * scale), int(h * scale)))

            # --- 2. FRAME SKIPPING LOGIC ---
            # Only run AI if it's the right frame
            if self.frame_count % frame_skip_interval == 0:
                
                # --- AI PROCESSING ---
                active_model = self.model_light
                self.current_model_name = "YOLOv8n (Light)"
                color_status = (0, 255, 255) 

                results = self.model_light(frame_resized, verbose=False)
                boxes = results[0].boxes
                
                # Logic checks
                num_objects = len(boxes)
                is_complex = num_objects > config.COMPLEXITY_THRESHOLD
                low_conf = False
                
                if num_objects > 0:
                     if boxes.conf.min().item() < config.CONFIDENCE_THRESHOLD:
                         low_conf = True

                # Switch to Heavy?
                if self.bandwidth_level > 60 and (is_complex or low_conf):
                    self.current_model_name = "YOLOv8s (Heavy)" 
                    color_status = (0, 0, 255)
                    results = self.model_heavy(frame_resized, verbose=False)
                
                # Save these results to use for the skipped frames
                last_results = results[0]
                
                # Visualization
                display_frame = results[0].plot()
                
            else:
                # --- SKIPPED FRAME ---
                # Just draw the OLD results on the NEW frame to fake smoothness
                if last_results is not None:
                     display_frame = last_results.plot(img=frame_resized)
                else:
                     display_frame = frame_resized

            # Resize back up for human viewing (so the HUD is readable)
            display_frame = cv2.resize(display_frame, (1080, 720))

            # --- FPS CALCULATION ---
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
            prev_time = curr_time

            # Draw HUD
            # Use 'locals().get' to prevent crashes if variables aren't set yet
            num_objects = locals().get('num_objects', 0)
            is_complex = locals().get('is_complex', False)
            low_conf = locals().get('low_conf', False)
            color_status = locals().get('color_status', (0, 255, 0))

            self.draw_hud(display_frame, fps, num_objects, is_complex, low_conf, color_status)

            cv2.imshow('Auto-Adaptive Analytics', display_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
    def draw_hud(self, img, fps, objs, complex_scene, low_conf, color):
        # Dark Background
        cv2.rectangle(img, (20, 20), (450, 280), (0, 0, 0), -1)
        cv2.rectangle(img, (20, 20), (450, 280), (255, 255, 255), 2)
        
        # 1. Bandwidth Bar (Dynamic)
        cv2.putText(img, f"Network Bandwidth: {self.bandwidth_level}%", (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        bar_color = (0, 255, 0) if self.bandwidth_level > 50 else (0, 0, 255)
        cv2.rectangle(img, (40, 75), (40 + self.bandwidth_level * 3, 95), bar_color, -1)
        
        # 2. Scene Analysis
        cv2.putText(img, f"Scene Objects: {objs}", (40, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        if complex_scene:
            cv2.putText(img, "[!] COMPLEX SCENE DETECTED", (220, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
        
        # 3. Decision Logic
        cv2.putText(img, f"Active Model: {self.current_model_name}", (40, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 3)
        cv2.putText(img, f"Resolution: {self.current_res}p", (40, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(img, f"FPS: {int(fps)}", (40, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)