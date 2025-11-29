import os
import cv2
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

import albumentations as A

class SegmentationDataset(Dataset):
    def __init__(self, df, img_size=(256,256), transforms=None):
        self.df = df.reset_index(drop=True)
        self.img_size = img_size
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row['image_path']
        mask_path = row['mask_path']

        img = cv2.imread(img_path)[..., ::-1]  # BGR -> RGB
        mask = cv2.imread(mask_path, 0)  # grayscale

        if img is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        if mask is None:
            raise FileNotFoundError(f"Mask not found: {mask_path}")

        img = cv2.resize(img, self.img_size, interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, self.img_size, interpolation=cv2.INTER_NEAREST)

        if self.transforms:
            augmented = self.transforms(image=img, mask=mask)
            img, mask = augmented['image'], augmented['mask']

        img = img.astype('float32') / 255.0
        mask = (mask > 127).astype('float32')  # binarize

        # HWC -> CHW
        img = np.transpose(img, (2,0,1)).copy()
        mask = np.expand_dims(mask, axis=0).copy()

        return img, mask

def get_transforms(train=True, size=(256,256)):
    if train:
        return A.Compose([
            A.Resize(*size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.Rotate(limit=20, p=0.5),
            A.RandomBrightnessContrast(p=0.5),
        ])
    else:
        return A.Compose([A.Resize(*size)])
