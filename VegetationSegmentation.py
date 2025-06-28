import cv2
import numpy as np
import os
from tqdm import tqdm

def vegetation_mask_exg(image):
    img = image.astype(np.float32)
    B, G, R = cv2.split(img)
    exg = 2 * G - R - B
    exg_norm = cv2.normalize(exg, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, mask = cv2.threshold(exg_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return mask

# --- Folder paths ---
input_folder = r"D:\AAU Internship\CWF-788\IMAGE400x300\validation"  # Change to your input folder
output_folder = r"D:\AAU Internship\CWF-788\IMAGE400x300\validation_test"  # Change to your output folder
os.makedirs(output_folder, exist_ok=True)

# --- Process all images in the folder ---
for fname in tqdm(os.listdir(input_folder)):
    if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
        img_path = os.path.join(input_folder, fname)
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        veg_mask = vegetation_mask_exg(img)
        out_path = os.path.join(output_folder, os.path.splitext(fname)[0] + '_vegmask.png')
        cv2.imwrite(out_path, 255-veg_mask)

print("✅ Vegetation masks saved to:", output_folder)