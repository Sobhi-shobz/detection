from ultralytics import YOLO

def run_detection():
    # 1. Load a pre-trained YOLOv8 Nano model
    model = YOLO('yolov8n.pt')

    # 2. Run detection on a sample image
    # 'save=True' automatically saves the result with bounding boxes
    results = model('https://ultralytics.com/images/bus.jpg', save=True)

    # 3. Explicitly save to a specific filename if needed
    # The default save path is usually 'runs/detect/predict/bus.jpg'
    for result in results:
        result.save(filename='results.jpg')
    
    print("Detection complete! Results saved as 'results.jpg'.")

if __name__ == "__main__":
    run_detection()