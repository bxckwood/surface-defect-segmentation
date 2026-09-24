import csv
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset


MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def image_to_tensor(image, image_size):
    height, width = image_size
    image = image.convert("RGB").resize((width, height), Image.Resampling.BILINEAR)

    array = np.asarray(image, dtype=np.float32).copy()
    tensor = torch.from_numpy(array).permute(2, 0, 1) / 255.0
    return (tensor - MEAN) / STD


class SurfaceDefectDataset(Dataset):
    """Читает пары изображений и масок из CSV-разбиения."""

    def __init__(self, data_root, split_csv, image_size=(640, 256), train=False):
        self.data_root = Path(data_root)
        self.image_size = tuple(image_size)
        self.train = train

        with Path(split_csv).open(newline="", encoding="utf-8") as file:
            self.rows = list(csv.DictReader(file))
        if not self.rows:
            raise ValueError(f"No image/mask pairs in {split_csv}")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        image_path = self.data_root / row["image"]
        mask_path = self.data_root / row["mask"]

        with Image.open(image_path) as file:
            image = file.convert("RGB")
        with Image.open(mask_path) as file:
            mask = file.convert("L")

        # NEAREST сохраняет бинарные значения разметки при изменении размера.
        height, width = self.image_size
        image = image.resize((width, height), Image.Resampling.BILINEAR)
        mask = mask.resize((width, height), Image.Resampling.NEAREST)

        # Отражения применяются вместе, чтобы маска осталась совмещена с изображением.
        if self.train:
            if random.random() < 0.5:
                image, mask = ImageOps.mirror(image), ImageOps.mirror(mask)
            if random.random() < 0.5:
                image, mask = ImageOps.flip(image), ImageOps.flip(mask)

        image_array = np.asarray(image, dtype=np.float32).copy()
        image_tensor = torch.from_numpy(image_array).permute(2, 0, 1) / 255.0
        image_tensor = (image_tensor - MEAN) / STD

        mask_array = (np.asarray(mask) > 0).astype(np.float32)
        mask_tensor = torch.from_numpy(mask_array.copy()).unsqueeze(0)

        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "image_path": row["image"],
            "mask_path": row["mask"],
        }
