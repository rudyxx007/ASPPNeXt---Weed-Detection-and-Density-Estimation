from ultralytics import YOLO
import os
import cv2
import numpy as np

def mask_to_yolo_labels(mask_folder, label_folder):
    os.makedirs(label_folder, exist_ok=True)
    for mask_file in os.listdir(mask_folder):
        if not mask_file.endswith('.png'):
            continue
        mask_path = os.path.join(mask_folder, mask_file)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        # Find contours
        contours, _ = cv2.findContours((mask > 127).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        h, w = mask.shape
        label_lines = []
        for cnt in contours:
            x, y, bw, bh = cv2.boundingRect(cnt)
            # YOLO format: class x_center y_center width height (all normalized)
            x_center = (x + bw / 2) / w
            y_center = (y + bh / 2) / h
            bw_norm = bw / w
            bh_norm = bh / h
            label_lines.append(f"0 {x_center} {y_center} {bw_norm} {bh_norm}")
        # Write label file
        label_path = os.path.join(label_folder, mask_file.replace('.png', '.txt'))
        with open(label_path, 'w') as f:
            f.write('\n'.join(label_lines))

# Example usage:
mask_to_yolo_labels(
    r"D:\AAU Internship\CWF-788\IMAGE400x300\trainlabel",
    r"D:\AAU Internship\CWF-788\IMAGE400x300\labels\train"
)
model = YOLO('yolov12n.pt')  # Use yolov12n.pt if available
model.train(data='vegetation.yaml', epochs=100, imgsz=300, batch=8)