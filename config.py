# --- CONFIGURATION (AUTO-PILOT) ---

# System Settings
VIDEO_SOURCE = 0          # Webcam
TARGET_FPS = 30           # The FPS we want to maintain

# Adaptive Thresholds
COMPLEXITY_THRESHOLD = 3  # If > 3 objects, scene is "Complex"
CONFIDENCE_THRESHOLD = 0.5 # If confidence < 0.5, model is "unsure"

# Resolution Presets
RES_LOW = 320
RES_HIGH = 640

# Model Paths (Ensure you have internet to download these automatically)
# We use v8 for speed and v11 for accuracy
MODEL_LIGHT = 'yolov8n.pt'   
MODEL_HEAVY = 'yolo11s.pt'   # Make sure you have the latest ultralytics pip install