from src.adaptive_engine import AdaptiveVideoAnalytics
import config

if __name__ == "__main__":
    print("Initializing SmartStream System...")
    print("Press 'q' to quit.") 
    app = AdaptiveVideoAnalytics(source=config.VIDEO_SOURCE)
    app.process_stream()