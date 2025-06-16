import os

import pytorch_lightning as pl
import albumentations as albu
from torch.utils.data import DataLoader
from pathlib import Path

from src.dataset.segmentation import SegmentationDataset


class SegmentationDataModule(pl.LightningDataModule):
    def __init__(self, txt_split, batch_size=16, resize_size=(256, 256)):
        super().__init__()
        self.txt_folder = Path(txt_split)
        self.batch_size = batch_size
        self.resize_size = resize_size
        self.size = resize_size

    def get_training_augmentation(self):
        train_transforms = [
            albu.GaussNoise(p=0.3),
            albu.HorizontalFlip(p=0.5),
            albu.OneOf(
                [
                    albu.RandomBrightnessContrast(p=0.6),
                    albu.RandomGamma(p=0.4),
                ],
                p=0.8,
            ),
            albu.Blur(p=0.5),
            albu.GridDropout(
                ratio=0.3, unit_size_range=(10, 20), fill="random_uniform", p=0.2
            ),
            albu.RandomFog(alpha=0.03, p=0.2),
        ]
        return albu.Compose(train_transforms)

    def load_image_mask_path(self, txt_file: str, train=False):
        with open(self.txt_folder / txt_file, "r") as file:
            # Read the contents of the file
            lines = file.readlines()
        image_path = [path.strip() for path in lines]

        mask_path = [path.replace("/images_", "/masks_").strip() for path in image_path]
        return image_path, mask_path

    def setup(self, stage=None):
        """Setup method to split dataset into train, val, and test"""
        train_paths, train_labels = self.load_image_mask_path("train.txt", train=True)
        val_paths, val_labels = self.load_image_mask_path("val.txt")
        test_paths, test_labels = self.load_image_mask_path("test.txt")
        self.train_dataset = SegmentationDataset(
            train_paths, train_labels, self.get_training_augmentation()
        )
        self.val_dataset = SegmentationDataset(val_paths, val_labels, None)
        self.test_dataset = SegmentationDataset(test_paths, test_labels, None)

    def train_dataloader(self):
        """Return the train dataloader"""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=8,
            persistent_workers=True,
        )

    def val_dataloader(self):
        """Return the validation dataloader"""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=8,
            persistent_workers=True,
        )

    def test_dataloader(self):
        """Return the test dataloader"""
        return DataLoader(
            self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=8
        )
