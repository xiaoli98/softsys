import lightning as L
import torch
from torch.utils.data import DataLoader, random_split
from dataset import ImageFileDataset
import torch.nn.functional as F
import numpy as np

def collate_fn(batch):
    """
    Custom collate function that pads images to match the largest in the batch.
    
    Handles two modes:
    - Contrastive: batch items are ((img1, img2), label)
    - Standard: batch items are (img, label)
    """
    # Check if this is contrastive learning mode
    is_contrastive = isinstance(batch[0][0], tuple)
    
    if is_contrastive:
        # Unpack contrastive batch
        images1, images2, labels = [], [], []
        for (img1, img2), label in batch:
            images1.append(img1)
            images2.append(img2)
            labels.append(label)
        
        # Stack and pad both views
        images1 = _pad_and_stack(images1)
        images2 = _pad_and_stack(images2)
        
        # Convert labels efficiently
        labels = torch.from_numpy(np.array(labels, dtype=np.float32))
        
        return (images1, images2), labels
    else:
        # Unpack standard batch
        images, labels = zip(*batch)
        
        # Stack and pad images
        images = _pad_and_stack(images)
        
        # Convert labels efficiently
        labels = torch.from_numpy(np.array(labels, dtype=np.float32))
        
        return images, labels

def _pad_and_stack(images):
    """
    Pad images to match the largest dimensions and stack them.
    
    Args:
        images: List of tensors with shape (C, H, W)
    
    Returns:
        Stacked and padded tensor with shape (B, C, H_max, W_max)
    """
    # Find maximum dimensions
    max_h = max(img.shape[-2] for img in images)
    max_w = max(img.shape[-1] for img in images)
    
    # Pad all images to max dimensions
    padded_images = []
    for img in images:
        h, w = img.shape[-2:]
        pad_h = max_h - h
        pad_w = max_w - w
        
        # Pad: (left, right, top, bottom)
        if pad_h > 0 or pad_w > 0:
            padded = F.pad(img, (0, pad_w, 0, pad_h), mode='constant', value=0)
        else:
            padded = img
        
        padded_images.append(padded)
    
    # Stack along batch dimension
    return torch.stack(padded_images, dim=0)

class ImageDataModule(L.LightningDataModule):
    def __init__(self, data_dir, batch_size, image_size=None, num_workers=4, sobel=False):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.image_size = image_size
        self.num_workers = num_workers
        self.sobel = sobel
        
    def setup(self, stage=None, contrastive=True, verbose=False):
        full_dataset = ImageFileDataset(
            self.data_dir, 
            image_size=self.image_size,
            sobel=self.sobel,
            contrastive=contrastive)
        self.num_classes = full_dataset.num_classes
        
        # Split data
        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        
        self.train_dataset, self.val_dataset = random_split(
            full_dataset, 
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42)  # Reproducible splits
        )
        
        if verbose:
            print(f"Dataset split: {train_size} train, {val_size} val")
            print(f"Number of classes: {self.num_classes}")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True, 
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=collate_fn,
            persistent_workers=True if self.num_workers > 0 else False,
            drop_last=True  # Ensures consistent batch sizes for training
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,  # No need to shuffle validation data
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=collate_fn,
            persistent_workers=True if self.num_workers > 0 else False
        )