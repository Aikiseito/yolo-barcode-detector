import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
import torch
import os
import cv2
import numpy as np
import dvc.api
from typing import Optional, Callable, List
from PIL import Image

class YOLODataset(Dataset):
    def __init__(self, data_dir: str, transform: Optional[Callable] = None):
        """
        Args:
            data_dir (str): Directory containing the images and labels.
            transform (Optional[Callable], optional): Optional transform to be applied
                on a sample. Defaults to None.
        """
        self.data_dir = data_dir
        self.image_files = [f for f in os.listdir(data_dir) if f.endswith('.jpg') or f.endswith('.png')]
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.data_dir, self.image_files[idx])
        image = Image.open(img_name).convert("RGB")

        # Load labels (assuming YOLO format: class_id x_center y_center width height)
        label_name = os.path.join(self.data_dir, self.image_files[idx].replace('.jpg', '.txt').replace('.png', '.txt'))
        try:
            labels = np.loadtxt(label_name, delimiter=' ', ndmin=2)  # Ensure at least 2 dimensions
        except OSError:
            labels = np.array([])  # Handle cases where no labels exist

        sample = {'image': image, 'labels': labels}

        if self.transform:
            sample = self.transform(sample)

        return sample


class YOLODetectionDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_dir: str,
        val_dir: str,
        test_dir: str,
        batch_size: int = 32,
        num_workers: int = 8,
        train_transforms: Optional[Callable] = None,
        val_transforms: Optional[Callable] = None,
        test_transforms: Optional[Callable] = None,
    ):
        super().__init__()
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.test_dir = test_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_transforms = train_transforms
        self.val_transforms = val_transforms
        self.test_transforms = test_transforms

    def setup(self, stage: Optional[str] = None):
        """
        Загрузка данных.  В этом месте нужно реализовать скачивание с Google Drive,
        если данные еще не скачаны локально.  DVC должен к этому моменту отслеживать
        эти директории
        """
        # Check if data is already downloaded (DVC pull should handle this)
        if not os.path.exists(self.train_dir) or not os.path.exists(self.val_dir) or not os.path.exists(self.test_dir):
            raise FileNotFoundError(f"Data directories not found.  Make sure to run 'dvc pull' first.")

        self.train_dataset = YOLODataset(self.train_dir, transform=self.train_transforms)
        self.val_dataset = YOLODataset(self.val_dir, transform=self.val_transforms)
        self.test_dataset = YOLODataset(self.test_dir, transform=self.test_transforms)


    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True,  # Helps transfer data to GPU faster
            collate_fn=self.collate_fn  # Custom collate function
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
            collate_fn=self.collate_fn
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
            collate_fn=self.collate_fn
        )

    def collate_fn(self, batch):
        """
        Custom collate function to handle variable-sized labels in YOLO format.
        """
        images = [item['image'] for item in batch]
        labels = [item['labels'] for item in batch]

        # Convert images to tensors
        images = [torch.as_tensor(np.array(image)).float().div(255.0).permute(2, 0, 1) for image in images]  # HWC to CHW, normalize

        # Convert labels to tensors, padding with -1
        max_boxes = max(len(label) for label in labels)
        padded_labels = torch.ones((len(labels), max_boxes, 5)) * -1  # 5 for class_id, x_center, y_center, width, height
        for i, label in enumerate(labels):
            if len(label) > 0:
                padded_labels[i, :len(label)] = torch.as_tensor(label)

        # Stack images and labels
        images = torch.stack(images)
        return images, padded_labels
