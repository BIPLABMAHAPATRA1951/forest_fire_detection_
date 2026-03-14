from ultralytics import YOLO
import cv2
import os

def predict_image(image_path):
    # Best trained model load karo
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                              "runs", "detect", "runs", "forest_fire6", "weights", "best.pt")
    
    model = YOLO(model_path)
    
    # Prediction karo
    results = model.predict(
        source=image_path,
        conf=0.25,          # 25% confidence threshold
        save=True,          # result image save karo
        save_txt=False,
        project="predictions",
        name="output"
    )
    
    # Results print karo
    for result in results:
        boxes = result.boxes
        if len(boxes) == 0:
            print("✅ No fire detected!")
        else:
            print(f"🔥 Detected {len(boxes)} object(s):")
            for box in boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                name = result.names[cls]
                print(f"   → {name}: {conf:.2%} confidence")
    
    return results

if __name__ == '__main__':
    # Test image path yahan daalo
    image_path = input("Image ka path daalo: ")
    
    if os.path.exists(image_path):
        predict_image(image_path)
        print("\nResult 'predictions/output/' folder mein save hua!")
    else:
        print("❌ Image nahi mili! Path check karo.")
