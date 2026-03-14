from ultralytics import YOLO
import os

def main():
    # Load pretrained YOLOv11 model
    model = YOLO("yolo11n.pt")

    # Absolute path use karo
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset", "data.yaml")

    # Train the model
    results = model.train(
        data=data_path,
        epochs=100,          # 100 epochs
        imgsz=640,
        batch=8,
        device=0,            # GPU
        project="runs",
        name="forest_fire",  # auto increment hoga → forest_fire6
        patience=15,         # early stopping
        save=True,
        plots=True,
        workers=4
    )

    print("Training Complete!")
    print(f"Best model saved at: {results.save_dir}")

if __name__ == '__main__':
    main()