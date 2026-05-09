import os
import glob
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


# ✅ DINOv3 normalization (IMPORTANT)
DINO_MEAN = [0.430, 0.411, 0.296]
DINO_STD  = [0.213, 0.156, 0.143]


class LoveDADataset(Dataset):
    def __init__(
        self,
        data_root,
        split="Train",   # Train / Val
        img_size=384,
        use_augmentation=False   # ❗ baseline = False
    ):
        self.data_root = data_root
        self.split = split
        self.img_size = img_size
        self.use_augmentation = use_augmentation and (split == "Train")

        self.image_paths = []
        self.mask_paths = []

        # ✅ Load Urban + Rural
        for region in ["Urban", "Rural"]:
            img_dir = os.path.join(data_root, split, split, region, "images_png")
            mask_dir = os.path.join(data_root, split, split, region, "masks_png")
            
            if os.path.exists(img_dir):
                images = sorted(glob.glob(os.path.join(img_dir, "*.png")))
                for img_path in images:
                    name = os.path.basename(img_path)
                    mask_path = os.path.join(mask_dir, name)

                    if os.path.exists(mask_path):
                        self.image_paths.append(img_path)
                        self.mask_paths.append(mask_path)

        print(f"{split}: {len(self.image_paths)} samples")

        self._build_transforms()

    def _build_transforms(self):
        # ✅ BASELINE (no heavy aug)
        if self.use_augmentation:
            self.transforms = A.Compose([
                A.Resize(self.img_size, self.img_size),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.Normalize(mean=DINO_MEAN, std=DINO_STD),
                ToTensorV2()
            ])
        else:
            self.transforms = A.Compose([
                A.Resize(self.img_size, self.img_size),
                A.Normalize(mean=DINO_MEAN, std=DINO_STD),
                ToTensorV2()
            ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # --- Load ---
        image = np.array(Image.open(self.image_paths[idx]).convert("RGB"))
        mask = np.array(Image.open(self.mask_paths[idx]))

        # --- Label Fix (CRITICAL) ---
        mask = mask.astype(np.int64)

        # LoveDA: 0 = ignore → 255
        mask[mask == 0] = 255

        # 1–7 → 0–6
        mask[mask != 255] -= 1

        # --- Transform ---
        augmented = self.transforms(image=image, mask=mask)
        image = augmented["image"]
        mask = augmented["mask"].long()

        return image, mask


# ✅ Optional helper (for debugging / visualization)
def decode_mask(mask):
    palette = np.array([
        [0, 0, 0],        # background
        [128, 0, 0],      # building
        [0, 128, 0],      # road
        [128, 128, 0],    # water
        [0, 0, 128],      # barren
        [128, 0, 128],    # forest
        [0, 128, 128],    # agriculture
    ])

    mask = mask.cpu().numpy()
    color = palette[mask]
    return color