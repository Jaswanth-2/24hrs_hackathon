import cv2
import time
import math
import numpy as np
import threading
import subprocess
import platform
import re
from ultralytics import YOLO
import config

class AdaptiveVideoAnalytics:
    def __init__(self, source=config.VIDEO_SOURCE):
        print(f"Loading Light Model ({config.MODEL_LIGHT})...")
        self.model_light = YOLO(config.MODEL_LIGHT)

        print(f"Loading Heavy Model ({config.MODEL_HEAVY})...")
        self.model_heavy = YOLO(config.MODEL_HEAVY)

        print("Connecting to video source...")
        self.cap = cv2.VideoCapture(source)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not self.cap.isOpened():
            raise RuntimeError("Cannot open video source")

        # --- STATE VARIABLES ---
        self.current_model_name = "YOLOv8n (Light)"
        self.current_res = config.RES_HIGH
        self.bandwidth_level = 100 
        self.frame_count = 0
        
        # --- START REAL-TIME NETWORK THREAD ---
        self.running = True
        self.network_thread = threading.Thread(target=self.monitor_network_realtime, daemon=True)
        self.network_thread.start()

    def monitor_network_realtime(self):
        """
        Runs in background. Pings Google DNS (8.8.8.8) to measure 'Real' Health.
        """
        # Command depends on OS (Windows use -n, Linux/Mac use -c)
        param = '-n' if platform.system().lower() == 'windows' else '-c'
        command = ['ping', param, '1', '8.8.8.8'] 

        while self.running:
            try:
                # Run ping command
                output = subprocess.check_output(command, stderr=subprocess.STDOUT, universal_newlines=True)
                
                # Extract time using Regex
                if platform.system().lower() == 'windows':
                    match = re.search(r"time=(\d+)", output)
                else:
                    match = re.search(r"time=(\d+\.?\d*)", output)

                if match:
                    latency = float(match.group(1))
                    
                    # LOGIC: < 50ms = 100%, > 500ms = 10%
                    if latency < 20: self.bandwidth_level = 100
                    elif latency < 50: self.bandwidth_level = 90
                    elif latency < 100: self.bandwidth_level = 75
                    elif latency < 200: self.bandwidth_level = 50
                    elif latency < 500: self.bandwidth_level = 30
                    else: self.bandwidth_level = 10
                else:
                    self.bandwidth_level = 0 # Timeout
                    
            except Exception:
                self.bandwidth_level = 0 # No Internet
            
            # Check every 1 second
            time.sleep(1)

    def process_stream(self):
        prev_time = time.time()
        frame_skip_interval = 3
        last_results = None

        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Frame not received")
                break

            self.frame_count += 1
            
            # NOTE: self.bandwidth_level is now updated automatically by the thread!

            # --- ADAPTIVE RESOLUTION LOGIC ---
            target_width = config.RES_LOW if self.bandwidth_level < 40 else config.RES_HIGH
            self.current_res = target_width

            h, w = frame.shape[:2]
            scale = target_width / w
            frame_resized = cv2.resize(frame, (int(w * scale), int(h * scale)))

            if self.frame_count % frame_skip_interval == 0:
                self.current_model_name = "YOLOv8n (Light)"
                color_status = (0, 255, 255)

                results = self.model_light(frame_resized, verbose=False)
                boxes = results[0].boxes
                num_objects = len(boxes)
                
                is_complex = num_objects > config.COMPLEXITY_THRESHOLD
                low_conf = False

                if num_objects > 0 and boxes.conf.min().item() < config.CONFIDENCE_THRESHOLD:
                    low_conf = True

                # --- MODEL SWITCHING LOGIC ---
                if self.bandwidth_level > 60 and (is_complex or low_conf):
                    self.current_model_name = "YOLOv8s (Heavy)"
                    color_status = (0, 0, 255)
                    results = self.model_heavy(frame_resized, verbose=False)

                last_results = results[0]
                display_frame = results[0].plot()

            else:
                if last_results is not None:
                    display_frame = last_results.plot(img=frame_resized)
                else:
                    display_frame = frame_resized

            display_frame = cv2.resize(display_frame, (1080, 720))

            # FPS Calculation
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time) if curr_time != prev_time else 0
            prev_time = curr_time

            # Use locals().get to prevent crashes on startup
            self.draw_hud(
                display_frame, fps,
                locals().get('num_objects', 0),
                locals().get('is_complex', False),
                locals().get('low_conf', False),
                locals().get('color_status', (0, 255, 0))
            )

            cv2.imshow("Real-Time Adaptive Analytics", display_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.running = False # Stop the thread safely
                break

        self.cap.release()
        cv2.destroyAllWindows()

    def draw_hud(self, img, fps, objs, complex_scene, low_conf, color):
        cv2.rectangle(img, (20, 20), (460, 290), (0, 0, 0), -1)
        cv2.rectangle(img, (20, 20), (460, 290), (255, 255, 255), 2)

        cv2.putText(img, f"Bandwidth: {self.bandwidth_level}%", (40, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        bar_color = (0, 255, 0) if self.bandwidth_level > 50 else (0, 0, 255)
        cv2.rectangle(img, (40, 75), (40 + self.bandwidth_level * 3, 95), bar_color, -1)

        cv2.putText(img, f"Objects: {objs}", (40, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

        if complex_scene:
            cv2.putText(img, "COMPLEX SCENE", (200, 130),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

        if low_conf:
            cv2.putText(img, "LOW CONFIDENCE", (40, 165),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        cv2.putText(img, f"Model: {self.current_model_name}", (40, 210),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 3)

        cv2.putText(img, f"Resolution: {self.current_res}p", (40, 245),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.putText(img, f"FPS: {int(fps)}", (40, 275),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)


if __name__ == "__main__":
    app = AdaptiveVideoAnalytics()
    app.process_stream()