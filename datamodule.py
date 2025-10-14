import lightning as L
from torch.utils.data import DataLoader, random_split
from dataset import ImageFileDataset

class ImageDataModule(L.LightningDataModule):
    def __init__(self, data_dir, batch_size, image_size, num_workers=4, sobel=False):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.image_size = image_size
        self.num_workers = num_workers
        self.sobel = sobel
        
    def setup(self, stage=None, verbose=False):
        full_dataset = ImageFileDataset(self.data_dir, image_size=self.image_size, sobel=self.sobel)
        self.num_classes = full_dataset.num_classes
        
        # Split data
        train_size = int(0.8 * len(full_dataset))
        val_size = int(0.1 * len(full_dataset))
        test_size = len(full_dataset) - train_size - val_size
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            full_dataset, [train_size, val_size, test_size]
        )
        if verbose:
            print(f"Dataset split: {train_size} train, {val_size} val, {test_size} test")
            print(f"Percentage of labels in train set: {self.train_dataset.dataset.labels_map}")
            print(f"Number of classes: {self.num_classes}")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True, 
            num_workers=self.num_workers,
            pin_memory=True)

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True)

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True)