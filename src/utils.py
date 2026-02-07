import cv2

def draw_dashboard(img, fps, mode, model_name, high_bandwidth, saved_counter):
    """
    Draws the stats overlay on the video frame.
    """
    # Background for text
    cv2.rectangle(img, (10, 10), (380, 170), (0, 0, 0), -1)
    
    # Color coding status
    if "IDLE" in mode:
        status_color = (0, 255, 0)   # Green
    elif "LIGHT" in mode:
        status_color = (0, 255, 255) # Yellow
    else:
        status_color = (0, 0, 255)   # Red

    bw_color = (0, 255, 0) if high_bandwidth else (0, 0, 255)

    # Text Info
    cv2.putText(img, f"FPS: {int(fps)}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(img, f"Mode: {mode}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
    cv2.putText(img, f"Model: {model_name}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    
    bw_text = "HIGH" if high_bandwidth else "LOW"
    cv2.putText(img, f"Bandwidth (Press 'b'): {bw_text}", (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, bw_color, 2)
    
    # Efficiency Metric
    cv2.putText(img, f"Heavy Inferences Avoided: {saved_counter}", (20, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)