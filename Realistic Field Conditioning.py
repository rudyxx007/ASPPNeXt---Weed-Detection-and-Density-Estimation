import torch as T
import torchvision as TV
import torchaudio as TA

if T.cuda.is_available():
    device=T.device("cuda")
else:
    device=T.device("cpu")

print(device)
import albumentations as A
import cv2
import numpy as np
import os
import random
from tqdm import tqdm

# ---------------------- Define Realistic Augmentation -----------------------
realistic_augmentations = A.Compose([
    A.OneOf([
        A.RandomSunFlare(flare_roi=(0, 0, 1, 0.4), src_radius=250, src_color=(255, 255, 255), p=1.0),
        A.RandomRain(p=1.0, slant_range=[-30, 30], drop_length=30, drop_width=1,
                     drop_color=(200, 200, 200), blur_value=5, brightness_coefficient=0.7, rain_type="default"),
        A.RandomFog(p=1.0, fog_coef_range=[0.3, 0.5], alpha_coef=0.1)
    ], p=1.0)
])

# ---------------------- Folders ---x   x--------------------
folder_prefix = r"D:\AAU Internship\CWF-788\IMAGE512x384"
folders = ["train", "test", "validation"]
new_folders = ["train_new", "test_new", "validation_new"]
mask_folders = ["trainlabel", "testlabel", "validationlabel"]
new_mask_folders = ["trainlabel_new", "testlabel_new", "validationlabel_new"]
img_ext = ".jpg"
mask_ext = ".png"

# ---------------------- Helper Functions -----------------------
def get_image_files(folder, extension):
    return [os.path.join(folder, f) for f in os.listdir(folder)
            if os.path.splitext(f)[1].lower() == extension]

def load_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Failed to load {image_path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def load_mask(mask_path):
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Failed to load {mask_path}")
    return mask

def save_image(path, image_rgb):
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, image_bgr)

def save_mask(path, mask):
    cv2.imwrite(path, mask)

# ---------------------- Main Loop -----------------------
for folder_div, new_folder_div, mask_folder_div, new_mask_folder_div in zip(folders, new_folders, mask_folders, new_mask_folders):
    folder = os.path.join(folder_prefix, folder_div)
    new_folder = os.path.join(folder_prefix, new_folder_div)
    mask_folder = os.path.join(folder_prefix, mask_folder_div)
    new_mask_folder = os.path.join(folder_prefix, new_mask_folder_div)

    if not os.path.isdir(folder) or not os.path.isdir(mask_folder):
        print(f"Skipping invalid folder: {folder} or {mask_folder}")
        continue

    os.makedirs(new_folder, exist_ok=True)
    os.makedirs(new_mask_folder, exist_ok=True)

    print(f"\n📁 Processing folder: {folder_div}")
    image_paths = get_image_files(folder, img_ext)
    image_paths.sort()

    images, masks, names = [], [], []

    # Load original images and masks
    for path in image_paths:
        image = load_image(path)
        basename = os.path.splitext(os.path.basename(path))[0]
        mask_path = os.path.join(mask_folder, basename + mask_ext)
        mask = load_mask(mask_path)

        images.append(image)
        masks.append(mask)
        names.append(basename)

    orig_count = len(images)
    aug_images, aug_masks, aug_names = [], [], []

    # Apply augmentation (image only)
    for i, (image, mask, basename) in enumerate(tqdm(list(zip(images, masks, names)), desc="⚙️ Augmenting")):
        augmented = realistic_augmentations(image=image)['image']
        aug_images.append(augmented)
        aug_masks.append(mask)  # Keep same mask
        aug_names.append(f"{orig_count + i + 1}_image")  # base name, no extension

    # Combine and shuffle (original + augmented)
    all_images = images + aug_images
    all_masks = masks + aug_masks
    all_names = names + aug_names
    combined = list(zip(all_images, all_masks, all_names))
    random.shuffle(combined)

    # Save images and masks with new names
    for i, (img, msk, _) in enumerate(tqdm(combined, desc="💾 Saving")):
        new_name = f"{i + 1}_image"
        save_image(os.path.join(new_folder, new_name + img_ext), img)
        save_mask(os.path.join(new_mask_folder, new_name + mask_ext), msk)

    print(f"✅ Done with '{folder_div}': Total image-mask pairs saved = {len(combined)}")
