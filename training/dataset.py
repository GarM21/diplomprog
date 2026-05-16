from pathlib import Path
import random

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


class GreenAreaDataset(Dataset):
    """Dataset for paired satellite images and binary vegetation masks."""

    def __init__(
        self,
        root_dir: str | Path,
        split: str,
        image_size: int = 256,
        augment: bool = False,
        mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: tuple[float, float, float] = (0.229, 0.224, 0.225),
    ) -> None:
        self.root_dir = Path(root_dir)
        self.split = split
        self.image_size = image_size
        self.augment = augment
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)

        self.images_dir = self.root_dir / split / "images"
        self.masks_dir = self.root_dir / split / "masks"
        self.samples = self._find_samples()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_path, mask_path = self.samples[index]

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        image, mask = self._resize(image, mask)

        if self.augment:
            image, mask = self._augment(image, mask)

        image_array = np.asarray(image, dtype=np.float32) / 255.0
        image_array = (image_array - self.mean) / self.std
        image_tensor = torch.from_numpy(image_array).permute(2, 0, 1).float()

        mask_array = np.asarray(mask, dtype=np.float32)
        mask_array = (mask_array > 0).astype(np.float32)
        mask_tensor = torch.from_numpy(mask_array).unsqueeze(0).float()

        return image_tensor, mask_tensor

    def _find_samples(self) -> list[tuple[Path, Path]]:
        if not self.images_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {self.images_dir}")
        if not self.masks_dir.exists():
            raise FileNotFoundError(f"Masks directory not found: {self.masks_dir}")

        image_paths = sorted(
            path
            for path in self.images_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        samples: list[tuple[Path, Path]] = []
        missing_masks: list[str] = []

        for image_path in image_paths:
            mask_path = self.masks_dir / image_path.name
            if mask_path.exists():
                samples.append((image_path, mask_path))
            else:
                missing_masks.append(image_path.name)

        if missing_masks:
            preview = ", ".join(missing_masks[:5])
            raise FileNotFoundError(
                f"Masks with matching filenames are missing: {preview}"
            )
        if not samples:
            raise ValueError(f"No image-mask pairs found in split: {self.split}")

        return samples

    def _resize(self, image: Image.Image, mask: Image.Image) -> tuple[Image.Image, Image.Image]:
        size = (self.image_size, self.image_size)
        image = image.resize(size, Image.Resampling.BILINEAR)
        mask = mask.resize(size, Image.Resampling.NEAREST)
        return image, mask

    def _augment(self, image: Image.Image, mask: Image.Image) -> tuple[Image.Image, Image.Image]:
        if random.random() < 0.5:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

        if random.random() < 0.5:
            image = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
            mask = mask.transpose(Image.Transpose.FLIP_TOP_BOTTOM)

        angle = random.uniform(-15.0, 15.0)
        image = image.rotate(angle, resample=Image.Resampling.BILINEAR)
        mask = mask.rotate(angle, resample=Image.Resampling.NEAREST)

        return image, mask
