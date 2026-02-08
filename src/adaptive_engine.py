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
        print("------------------------------------------------")
        print("  SMART-STREAM AI   ")
        print("------------------------------------------------")
        
        print(f"[LOAD] Loading Light Model ({config.MODEL_LIGHT})...")
        self.model_light = YOLO(config.MODEL_LIGHT)

        print(f"[LOAD] Loading Heavy Model ({config.MODEL_HEAVY})...")
        self.model_heavy = YOLO(config.MODEL_HEAVY)

        print("[CONN] Connecting to video source...")
        self.cap = cv2.VideoCapture(source)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not self.cap.isOpened():
            raise RuntimeError("Cannot open video source")

        self.current_model_name = "YOLOv8n (Light)"
        self.current_res = config.RES_HIGH
        self.bandwidth_level = 100 
        self.frame_count = 0
        
        self.running = True
        self.network_thread = threading.Thread(target=self.monitor_network_realtime, daemon=True)
        self.network_thread.start()
        print("[NET] Network Monitor Started (Background Thread)")

    def monitor_network_realtime(self):
        param = '-n' if platform.system().lower() == 'windows' else '-c'
        command = ['ping', param, '1', '8.8.8.8'] 

        while self.running:
            try:
                output = subprocess.check_output(command, stderr=subprocess.STDOUT, universal_newlines=True)
                
                if platform.system().lower() == 'windows':
                    match = re.search(r"time=(\d+)", output)
                else:
                    match = re.search(r"time=(\d+\.?\d*)", output)

                if match:
                    latency = float(match.group(1))
                    if latency < 20: self.bandwidth_level = 100
                    elif latency < 50: self.bandwidth_level = 90
                    elif latency < 100: self.bandwidth_level = 75
                    elif latency < 200: self.bandwidth_level = 50
                    elif latency < 500: self.bandwidth_level = 30
                    else: self.bandwidth_level = 10
                else:
                    self.bandwidth_level = 0 
            except Exception:
                self.bandwidth_level = 0 
            
            time.sleep(1) 

    def draw_tech_box(self, img, x1, y1, x2, y2, color, label, confidence):
        line_len = int((x2 - x1) * 0.2)
        thickness = 2
        
        cv2.line(img, (x1, y1), (x1 + line_len, y1), color, thickness)
        cv2.line(img, (x1, y1), (x1, y1 + line_len), color, thickness)
        cv2.line(img, (x2, y1), (x2 - line_len, y1), color, thickness)
        cv2.line(img, (x2, y1), (x2, y1 + line_len), color, thickness)
        cv2.line(img, (x1, y2), (x1 + line_len, y2), color, thickness)
        cv2.line(img, (x1, y2), (x1, y2 - line_len), color, thickness)
        cv2.line(img, (x2, y2), (x2 - line_len, y2), color, thickness)
        cv2.line(img, (x2, y2), (x2, y2 - line_len), color, thickness)
        
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, y1 - 20), (x1 + w + 10, y1), color, -1)
        cv2.putText(img, f"{label} {int(confidence*100)}%", (x1 + 5, y1 - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)

    def draw_sci_fi_hud(self, frame, fps, objs, is_complex, model_name, res, bw_level):
        h, w = frame.shape[:2]
        
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (280, h), (10, 10, 10), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        cv2.putText(frame, "SMART-STREAM", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.line(frame, (20, 50), (260, 50), (0, 255, 255), 2)
        
        y = 100
        cyan = (255, 255, 0)
        white = (220, 220, 220)
        
        cv2.putText(frame, f"NETWORK: {bw_level}%", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, cyan, 1)
        cv2.rectangle(frame, (20, y+10), (240, y+25), (50, 50, 50), -1)
        
        bar_w = int(220 * (bw_level / 100))
        bar_col = (0, 255, 0) if bw_level > 50 else (0, 0, 255)
        cv2.rectangle(frame, (20, y+10), (20 + bar_w, y+25), bar_col, -1)
        
        y += 60
        cv2.putText(frame, "AI INFERENCE ENGINE", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, cyan, 1)
        cv2.putText(frame, f">> {model_name}", (20, y+25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, white, 1)
        
        y += 60
        cv2.putText(frame, "LIVE ANALYTICS", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, cyan, 1)
        cv2.putText(frame, f"FPS: {int(fps)}", (20, y+25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, white, 1)
        cv2.putText(frame, f"RES: {res}p", (140, y+25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, white, 1)
        cv2.putText(frame, f"OBJS: {objs}", (20, y+50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, white, 1)

        if is_complex:
             cv2.rectangle(frame, (20, y+70), (240, y+110), (0, 0, 255), -1)
             cv2.putText(frame, "COMPLEX SCENE", (30, y+98), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    def process_stream(self):
        prev_time = time.time()
        frame_skip_interval = 3
        last_results = None
        
        num_objects = 0
        is_complex = False
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Frame not received. Exiting...")
                break

            self.frame_count += 1
            
            target_width = config.RES_LOW if self.bandwidth_level < 40 else config.RES_HIGH
            self.current_res = target_width

            h, w = frame.shape[:2]
            scale = target_width / w
            frame_resized = cv2.resize(frame, (int(w * scale), int(h * scale)))

            if self.frame_count % frame_skip_interval == 0:
                self.current_model_name = "YOLOv8n (Light)"
                
                results = self.model_light(frame_resized, verbose=False)
                boxes = results[0].boxes
                num_objects = len(boxes)
                
                is_complex = num_objects > config.COMPLEXITY_THRESHOLD
                low_conf = False
                if num_objects > 0 and boxes.conf.min().item() < config.CONFIDENCE_THRESHOLD:
                    low_conf = True

                if self.bandwidth_level > 60 and (is_complex or low_conf):
                    self.current_model_name = "YOLOv11s (Heavy)"
                    results = self.model_heavy(frame_resized, verbose=False)

                last_results = results[0]
            
            display_frame = frame_resized.copy()
            
            if last_results is not None:
                for box in last_results.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    
                    label = self.model_light.names[cls] if hasattr(self.model_light, 'names') else "Object"

                    color = (255, 255, 0) 
                    if self.current_model_name == "YOLOv8s (Heavy)":
                        color = (0, 0, 255)
                    
                    self.draw_tech_box(display_frame, x1, y1, x2, y2, color, label, conf)

            display_frame = cv2.resize(display_frame, (1280, 720))

            curr_time = time.time()
            fps = 1 / (curr_time - prev_time) if curr_time != prev_time else 0
            prev_time = curr_time

            self.draw_sci_fi_hud(display_frame, fps, num_objects, is_complex, 
                               self.current_model_name, self.current_res, self.bandwidth_level)

            cv2.imshow("SmartStream AI // TERMINAL", display_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.running = False
                break

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    app = AdaptiveVideoAnalytics()
    app.process_stream()