import os
import torch as T
import torchvision as TV
import torchaudio as TA
import cv2
import numpy as np
import random
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch import optim
from torch.utils.data import DataLoader, Dataset
import segmentation_models_pytorch as smp
from glob import glob
import albumentations as A
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score, confusion_matrix
from pathlib import Path
print(T.cuda.is_available())
print(T.cuda.device_count())
print(T.cuda.get_device_name(0))
# ---------------------- DEVICE -----------------------
device = T.device("cuda" if T.cuda.is_available() else "cpu")
print(f"Using device: {device}")
# ---------------------- Paths -----------------------
train_images = r"D:\AAU Internship\CWF-788\IMAGE512x384\train_new"
train_masks = r"D:\AAU Internship\CWF-788\IMAGE512x384\trainlabel_new"
validation_images = r"D:\AAU Internship\CWF-788\IMAGE512x384\validation_new"
validation_masks = r"D:\AAU Internship\CWF-788\IMAGE512x384\validationlabel_new"
test_images = r"D:\AAU Internship\CWF-788\IMAGE512x384\test_new"
test_masks = r"D:\AAU Internship\CWF-788\IMAGE512x384\testlabel_new"
# ---------------------- Augmentations -----------------------
train_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.5),
    A.ElasticTransform(p=0.5),
    A.D4(p=1),
    A.ISONoise(color_shift=[0.01, 0.05], intensity=[0.1, 0.5], p=0.5),
    A.RandomBrightnessContrast(brightness_limit=[-0.2, 0.2], contrast_limit=[-0.2, 0.2], p=0.5),
    A.ElasticTransform(alpha=300, sigma=10, interpolation=cv2.INTER_NEAREST, mask_interpolation=cv2.INTER_NEAREST,
                       same_dxdy=True, border_mode=cv2.BORDER_CONSTANT, fill=0, fill_mask=0, p=0.5),
    A.Resize(512, 384),
])

base_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# ---------------------- Dataset Class -----------------------
class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, train_transform=None, base_transform=None, dataset_type="Unknown"):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.train_transform = train_transform
        self.base_transform = base_transform
        self.dataset_type = dataset_type
        self.image_files = sorted(glob(os.path.join(image_dir, "*.jpg")))
        self.mask_files = sorted(glob(os.path.join(mask_dir, "*.png")))
        self._verify_file_pairs()
        
    def _verify_file_pairs(self):
        if len(self.image_files) != len(self.mask_files):
            raise ValueError(f"Mismatched counts in {self.dataset_type} dataset: {len(self.image_files)} images vs {len(self.mask_files)} masks")
            
        for img_path, mask_path in tqdm(zip(self.image_files, self.mask_files), total=len(self.image_files), desc=f"Verifying {self.dataset_type} File Pairs 🔍"):
            img_name = os.path.splitext(os.path.basename(img_path))[0]
            mask_name = os.path.splitext(os.path.basename(mask_path))[0]
            if img_name != mask_name:
                raise ValueError(f"Filename mismatch in {self.dataset_type} dataset: {img_name} vs {mask_name}")

    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):  # <-- FIXED INDENTATION
        # Read image and mask
        img = cv2.cvtColor(cv2.imread(self.image_files[idx]), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.mask_files[idx], cv2.IMREAD_GRAYSCALE)

        mask = (mask > 127).astype(np.uint8)

        original_img = self.base_transform(img)
        original_mask = T.from_numpy(mask).long()

        sample = {
            "original_img": original_img,
            "original_mask": original_mask
        }

        if self.train_transform:
            augmented = self.train_transform(image=img, mask=mask)
            aug_img = augmented["image"]
            aug_mask = augmented["mask"]

            augmented_img = self.base_transform(aug_img)
            augmented_mask = T.from_numpy(aug_mask).long()

            sample["augmented_img"] = augmented_img
            sample["augmented_mask"] = augmented_mask
        else:
            sample["augmented_img"] = original_img
            sample["augmented_mask"] = original_mask

        return sample


# ---------------------- DataLoaders -----------------------
train_dataset = SegmentationDataset(train_images, train_masks, train_transform, base_transform, "Training")
val_dataset = SegmentationDataset(validation_images, validation_masks, train_transform, base_transform, "Validation")
test_dataset = SegmentationDataset(test_images, test_masks, train_transform, base_transform, "Testing")

train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0, pin_memory=True)
val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=True, num_workers=0, pin_memory=True)
test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=True, num_workers=0, pin_memory=True)
# ---------------------- Model -----------------------
CUSTOM_SAVE_ROOT = Path(r"D:\AAU Internship\UNet-Models")
os.makedirs(CUSTOM_SAVE_ROOT, exist_ok=True)

model = smp.Unet(
    encoder="b7",
    encoder_weights="imagenet",
    encoder_depth=4,
    decoder_use_batchnorm='inplace',
#    decoder_attention_type='scse',
    decoder_channels=[256, 128, 64, 32],
    in_channels=3,
    classes=2,
    activation=None,
    center=True,
).to(device)

# ---------------------- Loss Function -----------------------
class FocalTverskyLoss(nn.Module):
    def __init__(self, alpha=0.7, beta=0.3, gamma=0.75, smooth=1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth

    def update_hyperparams_by_epoch(self, epoch):
        steps = epoch // 5
        self.alpha = max(0.4, 0.7 - 0.03*steps)
        self.beta = 1 - self.alpha
        self.gamma = min(1.5, 0.5 + 0.1*steps)

    def forward(self, preds, targets):
        targets_one_hot = F.one_hot(targets, num_classes=preds.shape[1]).permute(0, 3, 1, 2).float()
        probs = preds
        dims = (0, 2, 3)
        TP = T.sum(probs * targets_one_hot, dims)
        FP = T.sum(probs * (1 - targets_one_hot), dims)
        FN = T.sum((1 - probs) * targets_one_hot, dims)
        Tversky = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)
        return T.mean((1 - Tversky) ** self.gamma)

loss_fn = FocalTverskyLoss().to(device)

# ---------------------- Metrics -----------------------
def compute_metrics(preds, targets):
    with T.no_grad():
        pred_labels = T.argmax(preds, dim=1).cpu().numpy().flatten()
        targets = targets.cpu().numpy().flatten()
        ious = []
        for cls in [0, 1]:
            intersection = ((pred_labels == cls) & (targets == cls)).sum()
            union = ((pred_labels == cls) | (targets == cls)).sum()
            ious.append(intersection / (union + 1e-6))
        class_acc = []
        for cls in [0, 1]:
            mask = (targets == cls)
            if mask.sum() > 0:
                class_acc.append((pred_labels[mask] == cls).mean())
        mPA = np.mean(class_acc) * 100
        cm = confusion_matrix(targets, pred_labels)
        TN, FP, FN, TP = cm.ravel()
        return {
            "Accuracy": 100 * accuracy_score(targets, pred_labels),
            "mPA": mPA,
            "Crop IoU": 100 * ious[1],
            "mIoU": 100 * np.mean(ious),
            "Precision": 100 * precision_score(targets, pred_labels, zero_division=0),
            "Recall": 100 * recall_score(targets, pred_labels, zero_division=0),
            "F1-Score": 100 * f1_score(targets, pred_labels, zero_division=0),
            "FNR": 100 * (FN / (FN + TP + 1e-6))
        }

# ---------------------- Training Setup -----------------------
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
MODEL_PATHS = {name: CUSTOM_SAVE_ROOT / f"best_{name.replace(' ', '')}_model.pth" for name in [
    "mPA", "mIoU", "Crop IoU", "Accuracy", "F1-Score", "Precision", "Recall", "FNR"
]}
best_metrics = {k: {"value": -1 if k != "FNR" else float('inf'), "path": v} for k, v in MODEL_PATHS.items()}

# ---------------------- Training & Validation -----------------------
def TrainUNet(model, dataloader, loss_fn, optimizer, epoch):
    model.train()
    running_loss = 0
    all_preds, all_targets = [], []
    loss_fn.update_hyperparams_by_epoch(epoch)
    loop = tqdm(dataloader, desc=f"Epoch {epoch} [Train]", leave=False)

    for batch in loop:
        inputs = batch['augmented_img'].to(device)
        targets = batch['augmented_mask'].to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        all_preds.append(outputs.detach().cpu())
        all_targets.append(targets.detach().cpu())
        loop.set_postfix(loss=loss.item())

    avg_loss = running_loss / len(dataloader)
    metrics = compute_metrics(T.cat(all_preds), T.cat(all_targets))

    T.cuda.empty_cache()
    
    return avg_loss, metrics

def ValidateUNet(model, dataloader, loss_fn):
    model.eval()
    running_loss = 0
    all_preds, all_targets = []
    loop = tqdm(dataloader, desc="Validating", leave=False)

    with T.no_grad():
        for batch in loop:
            inputs = batch['original_img'].to(device)
            targets = batch['original_mask'].to(device)
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            running_loss += loss.item()
            all_preds.append(outputs.detach().cpu())
            all_targets.append(targets.detach().cpu())
            loop.set_postfix(loss=loss.item())

    avg_loss = running_loss / len(dataloader)
    metrics = compute_metrics(T.cat(all_preds), T.cat(all_targets))
    
    T.cuda.empty_cache()

    return avg_loss, metrics

# ---------------------- Main Training -----------------------
num_epochs = 50
for epoch in range(1, num_epochs + 1):
    train_loss, train_metrics = TrainUNet(model, train_dataloader, loss_fn, optimizer, epoch)
    val_loss, val_metrics = ValidateUNet(model, val_dataloader, loss_fn)

    T.cuda.empty_cache()

    for metric_name in best_metrics.keys():
        current_value = val_metrics[metric_name]
        is_better = current_value > best_metrics[metric_name]["value"] if metric_name != "FNR" else current_value < best_metrics[metric_name]["value"]
        if is_better:
            best_metrics[metric_name]["value"] = current_value
            T.save(model.state_dict(), str(best_metrics[metric_name]["path"]))
            print(f"✅ New best {metric_name}: {current_value:.2f}% | Saved to: {best_metrics[metric_name]['path']}")

    print(f"\n📊 Epoch {epoch} Summary:")
    print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
    for k, v in val_metrics.items():
        print(f"{k}: {v:.2f}%")


# ---------------------- Final Report -----------------------
print("\n🎯 === Best Models Summary ===")
for metric_name, data in best_metrics.items():
    print(f"{metric_name}: {data['value']:.2f}% → {data['path']}")

# ---------------------- Testing -----------------------
print("\n🧪 === Testing Saved Models ===")
for metric_name, data in tqdm(best_metrics.items(), desc="Testing Models"):
    model.load_state_dict(T.load(str(data["path"])))
    test_loss, test_metrics = ValidateUNet(model, test_dataloader, loss_fn)
    print(f"\n📌 {metric_name} Model Test Results:")
    for k, v in test_metrics.items():
        print(f"{k}: {v:.2f}%")
