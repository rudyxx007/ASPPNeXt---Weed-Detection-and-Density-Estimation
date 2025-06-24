import os
import cv2
import shutil

def mask_to_yolo_labels(mask_folder, label_folder):
    os.makedirs(label_folder, exist_ok=True)
    for mask_file in os.listdir(mask_folder):
        if not mask_file.endswith('.png'):
            continue
        mask_path = os.path.join(mask_folder, mask_file)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        contours, _ = cv2.findContours((mask > 127).astype('uint8'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        h, w = mask.shape
        label_lines = []
        for cnt in contours:
            x, y, bw, bh = cv2.boundingRect(cnt)
            x_center = (x + bw / 2) / w
            y_center = (y + bh / 2) / h
            bw_norm = bw / w
            bh_norm = bh / h
            label_lines.append(f"0 {x_center} {y_center} {bw_norm} {bh_norm}")
        label_path = os.path.join(label_folder, mask_file.replace('.png', '.txt'))
        with open(label_path, 'w') as f:
            f.write('\n'.join(label_lines))

def copy_images(src_folder, dst_folder):
    os.makedirs(dst_folder, exist_ok=True)
    for img_file in os.listdir(src_folder):
        if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
            shutil.copy(os.path.join(src_folder, img_file), os.path.join(dst_folder, img_file))

# Set your source base path (change to IMAGE512x384 if you want that size)
src_base = r"D:\AAU Internship\CWF-788\IMAGE512x384"
dst_base = r"D:\AAU Internship\CWF-788\YOLO_vegetation_512x384"

sublits = [
    ("train_new", "trainlabel_new"),
    ("test_new", "testlabel_new"),
    ("validation_new", "validationlabel_new"),
]

for img_split, mask_split in sublits:
    # Prepare images
    copy_images(
        os.path.join(src_base, img_split),
        os.path.join(dst_base, "images", img_split.replace("_new", ""))
    )
    # Prepare labels
    mask_to_yolo_labels(
        os.path.join(src_base, mask_split),
        os.path.join(dst_base, "labels", img_split.replace("_new", ""))
    )

print("✅ YOLO vegetation dataset created at:", dst_base)

yaml_content = f"""train: {dst_base}/images/train
val: {dst_base}/images/validation

nc: 1
names: ['vegetation']
"""

yaml_path = os.path.join(dst_base, "vegetation.yaml")
with open(yaml_path, "w") as f:
    f.write(yaml_content)

print(f"✅ vegetation.yaml created at: {yaml_path}")