"""
Dataset utilities for loading a face-image dataset (e.g. CelebA, FFHQ, or any
folder of face crops) for DCGAN training.

Expected layout (torchvision.ImageFolder convention):
    data/
      images/
        img_000001.jpg
        img_000002.jpg
        ...

A single subfolder is enough — ImageFolder just needs >=1 class directory.
If you already have a flat folder of images with no subdirectory, the
FlatImageDataset fallback below handles that directly.
"""

import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as dset


def build_transforms(image_size):
    """
    Resize + center-crop to a square, convert to tensor, and normalize to [-1, 1]
    to match the Generator's Tanh output range.
    """
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])


class FlatImageDataset(Dataset):
    """Fallback dataset for a directory containing images directly (no class subfolders)."""

    IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.files = [
            f for f in sorted(os.listdir(root))
            if f.lower().endswith(self.IMG_EXTENSIONS)
        ]
        if len(self.files) == 0:
            raise RuntimeError(
                f"No images found in '{root}'. Expected files with extensions {self.IMG_EXTENSIONS}."
            )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = os.path.join(self.root, self.files[idx])
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, 0  # dummy label; DCGAN training is unconditional


def get_dataloader(data_dir, image_size, batch_size, num_workers=4):
    """
    Tries torchvision.ImageFolder first (expects >=1 subdirectory with images).
    Falls back to FlatImageDataset if data_dir contains images directly.
    """
    transform = build_transforms(image_size)

    has_subdirs = any(
        os.path.isdir(os.path.join(data_dir, d)) for d in os.listdir(data_dir)
    ) if os.path.isdir(data_dir) else False

    if has_subdirs:
        dataset = dset.ImageFolder(root=data_dir, transform=transform)
    else:
        dataset = FlatImageDataset(root=data_dir, transform=transform)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,  # keeps BatchNorm stable by avoiding a tiny final batch
    )
    return dataloader, dataset
