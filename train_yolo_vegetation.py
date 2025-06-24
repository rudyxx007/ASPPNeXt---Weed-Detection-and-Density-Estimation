from ultralytics import YOLO
import torch

def main():
    print(torch.cuda.is_available())
    if torch.cuda.is_available():
        print(torch.cuda.get_device_name(0))

    # Path to your YAML file
    yaml_path = r"D:\AAU Internship\CWF-788\YOLO_vegetation_512x384\vegetation.yaml"

    # Load YOLOv8n model
    model = YOLO('yolo12n-seg.yaml').load("yolo12n.pt")  # or 'yolov8n-seg.pt' for segmentation

    # Train the model
    model.train(
        data=yaml_path,
        epochs=100,
        imgsz=512,
        batch=8,
        device=0
    )

if __name__ == "__main__":
    main()