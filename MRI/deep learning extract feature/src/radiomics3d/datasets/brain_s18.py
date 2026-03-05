"""Dataset utilities (generic).

This implementation is adapted from an internal prototype and de-identified.
It expects a CSV with columns: `image_path`, `voi_path`.

Notes
-----
- For preprocessing, we apply:
  1) drop invalid range (remove background extent),
  2) random center crop around non-zero mask,
  3) resize to fixed input size,
  4) intensity normalization on non-zero region.
"""

import os
import pandas as pd
import numpy as np
import nibabel as nib
from scipy import ndimage
from torch.utils.data import Dataset


class BrainS18Dataset(Dataset):
    def __init__(self, csv_path: str, input_D: int, input_H: int, input_W: int, phase: str = "train"):
        self.df = pd.read_csv(csv_path)
        if not {"image_path", "voi_path"}.issubset(set(self.df.columns)):
            raise ValueError("CSV must contain columns: image_path, voi_path")

        self.input_D = int(input_D)
        self.input_H = int(input_H)
        self.input_W = int(input_W)
        self.phase = phase

    @staticmethod
    def _nii_to_tensorarray(data: np.ndarray) -> np.ndarray:
        # (D,H,W) -> (1,D,H,W)
        if data.ndim != 3:
            raise ValueError(f"Expected 3D array, got shape={data.shape}")
        z, y, x = data.shape
        out = np.reshape(data, (1, z, y, x)).astype("float32")
        return out

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_path = str(row["image_path"])
        mask_path = str(row["voi_path"])

        if not os.path.isfile(img_path):
            raise FileNotFoundError(f"Missing image: {img_path}")
        if not os.path.isfile(mask_path):
            raise FileNotFoundError(f"Missing mask: {mask_path}")

        img_nii = nib.load(img_path)
        mask_nii = nib.load(mask_path)

        img_arr, mask_arr = self._training_process(img_nii, mask_nii)

        img_arr = self._nii_to_tensorarray(img_arr)
        mask_arr = self._nii_to_tensorarray(mask_arr)

        sample_name = os.path.splitext(os.path.basename(mask_path))[0]
        return sample_name, img_arr, mask_arr

    @staticmethod
    def _drop_invalid_range(volume: np.ndarray, label: np.ndarray | None = None):
        zero_value = volume.flat[0]
        non_zeros = np.where(volume != zero_value)
        max_z, max_h, max_w = np.max(np.array(non_zeros), axis=1)
        min_z, min_h, min_w = np.min(np.array(non_zeros), axis=1)

        if label is None:
            return volume[min_z:max_z, min_h:max_h, min_w:max_w]
        return (
            volume[min_z:max_z, min_h:max_h, min_w:max_w],
            label[min_z:max_z, min_h:max_h, min_w:max_w],
        )

    @staticmethod
    def _random_center_crop(data: np.ndarray, label: np.ndarray):
        targets = np.where(label > 0)
        if len(targets[0]) == 0:
            return data, label

        img_d, img_h, img_w = data.shape
        max_D, max_H, max_W = np.max(np.array(targets), axis=1)
        min_D, min_H, min_W = np.min(np.array(targets), axis=1)
        target_depth, target_height, target_width = np.array([max_D, max_H, max_W]) - np.array([min_D, min_H, min_W])

        rnd = np.random.random
        Z_min = int((min_D - target_depth / 2.0) * rnd())
        Y_min = int((min_H - target_height / 2.0) * rnd())
        X_min = int((min_W - target_width / 2.0) * rnd())

        Z_max = int(img_d - ((img_d - (max_D + target_depth / 2.0)) * rnd()))
        Y_max = int(img_h - ((img_h - (max_H + target_height / 2.0)) * rnd()))
        X_max = int(img_w - ((img_w - (max_W + target_width / 2.0)) * rnd()))

        Z_min = max(0, Z_min); Y_min = max(0, Y_min); X_min = max(0, X_min)
        Z_max = min(img_d, Z_max); Y_max = min(img_h, Y_max); X_max = min(img_w, X_max)

        return data[Z_min:Z_max, Y_min:Y_max, X_min:X_max], label[Z_min:Z_max, Y_min:Y_max, X_min:X_max]

    def _resize(self, data: np.ndarray) -> np.ndarray:
        depth, height, width = data.shape
        scale = [self.input_D / depth, self.input_H / height, self.input_W / width]
        return ndimage.zoom(data, scale, order=0)

    @staticmethod
    def _intensity_normalize(volume: np.ndarray) -> np.ndarray:
        pixels = volume[volume > 0]
        if pixels.size == 0:
            return volume.astype(np.float32)

        mean = pixels.mean()
        std = pixels.std() if pixels.std() > 1e-8 else 1.0
        out = (volume - mean) / std
        noise = np.random.normal(0, 1, size=volume.shape)
        out[volume == 0] = noise[volume == 0]
        return out.astype(np.float32)

    def _training_process(self, img_nii, mask_nii):
        img = img_nii.get_fdata()
        mask = mask_nii.get_fdata()

        img, mask = self._drop_invalid_range(img, mask)
        img, mask = self._random_center_crop(img, mask)

        img = self._resize(img)
        mask = self._resize(mask)

        img = self._intensity_normalize(img)
        return img, mask
