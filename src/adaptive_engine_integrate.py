import cv2
import time
import threading
import subprocess
import platform
import re
from ultralytics import YOLO
import config


class AdaptiveVideoAnalytics:
    def __init__(self):
        print("------------------------------------------------")
        print("        SMART-STREAM AI SYSTEM")
        print("------------------------------------------------")

        print(f"[LOAD] Loading Light Model ({config.MODEL_LIGHT})")
        self.model_light = YOLO(config.MODEL_LIGHT)

        print(f"[LOAD] Loading Heavy Model ({config.MODEL_HEAVY})")
        self.model_heavy = YOLO(config.MODEL_HEAVY)

        # ===== VIDEO SOURCE STATE =====
        self.current_source = "LOCAL"
        self.cap = cv2.VideoCapture(config.VIDEO_SOURCE_LOCAL)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not self.cap.isOpened():
            raise RuntimeError("Camera not opened")

        self.current_model_name = "YOLOv8n (Light)"
        self.current_res = config.RES_HIGH
        self.bandwidth_level = 100
        self.frame_count = 0
        self.running = True

        # ===== NETWORK MONITOR THREAD =====
        self.network_thread = threading.Thread(
            target=self.monitor_network_realtime, daemon=True
        )
        self.network_thread.start()

        print("[NET] Network Monitor Started")

    # -------------------------------------------------
    # NETWORK MONITOR
    # -------------------------------------------------
    def monitor_network_realtime(self):
        param = "-n" if platform.system().lower() == "windows" else "-c"
        command = ["ping", param, "1", "8.8.8.8"]

        while self.running:
            try:
                output = subprocess.check_output(
                    command, stderr=subprocess.STDOUT, universal_newlines=True
                )

                match = re.search(r"time[=<](\d+\.?\d*)", output)
                if match:
                    latency = float(match.group(1))
                    if latency < 20:
                        self.bandwidth_level = 100
                    elif latency < 50:
                        self.bandwidth_level = 80
                    elif latency < 100:
                        self.bandwidth_level = 60
                    elif latency < 200:
                        self.bandwidth_level = 40
                    else:
                        self.bandwidth_level = 20
                else:
                    self.bandwidth_level = 10
            except:
                self.bandwidth_level = 10

            time.sleep(1)

    # -------------------------------------------------
    # CAMERA SWITCH FUNCTION
    # -------------------------------------------------
    def switch_video_source(self, source_type):
        if source_type == self.current_source:
            return

        print(f"[SWITCH] Camera → {source_type}")

        self.cap.release()

        if source_type == "LOCAL":
            self.cap = cv2.VideoCapture(config.VIDEO_SOURCE_LOCAL)
        else:
            self.cap = cv2.VideoCapture(config.VIDEO_SOURCE_URL)

        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.current_source = source_type

    # -------------------------------------------------
    # DRAW BOX
    # -------------------------------------------------
    def draw_box(self, img, x1, y1, x2, y2, label, conf, color):
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        text = f"{label} {int(conf*100)}%"
        cv2.putText(img, text, (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # -------------------------------------------------
    # MAIN LOOP
    # -------------------------------------------------
    def process_stream(self):
        prev_time = time.time()
        last_results = None

        while True:
            # ===== AUTO CAMERA SWITCH =====
            if self.bandwidth_level > 50:
                self.switch_video_source("LOCAL")
            else:
                self.switch_video_source("URL")

            ret, frame = self.cap.read()
            if not ret:
                print("Frame not received")
                time.sleep(0.2)
                continue

            self.frame_count += 1

            # ===== RESOLUTION ADAPT =====
            target_width = (
                config.RES_LOW if self.bandwidth_level < 40 else config.RES_HIGH
            )
            self.current_res = target_width

            h, w = frame.shape[:2]
            scale = target_width / w
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)))

            # ===== INFERENCE =====
            self.current_model_name = "YOLOv8n (Light)"
            results = self.model_light(frame, verbose=False)
            boxes = results[0].boxes

            is_complex = len(boxes) > config.COMPLEXITY_THRESHOLD
            low_conf = (
                len(boxes) > 0 and boxes.conf.min().item() < config.CONFIDENCE_THRESHOLD
            )

            if self.bandwidth_level > 60 and (is_complex or low_conf):
                self.current_model_name = "YOLOv11s (Heavy)"
                results = self.model_heavy(frame, verbose=False)

            last_results = results[0]

            # ===== DRAW RESULTS =====
            for box in last_results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                label = self.model_light.names[cls]

                color = (0, 255, 0) if self.current_source == "LOCAL" else (0, 0, 255)
                self.draw_box(frame, x1, y1, x2, y2, label, conf, color)

            # ===== FPS =====
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time)
            prev_time = curr_time

            # ===== HUD =====
            cv2.putText(frame, f"FPS: {int(fps)}", (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"SRC: {self.current_source}", (20, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(frame, f"NET: {self.bandwidth_level}%", (20, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(frame, self.current_model_name, (20, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            frame = cv2.resize(frame, (1280, 720))
            cv2.imshow("SmartStream AI", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                self.running = False
                break

        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    app = AdaptiveVideoAnalytics()
    app.process_stream()
