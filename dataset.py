import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

class ResizeAndPad:
    def __init__(self, output_size, fill=0):
        assert len(output_size) == 2, "Output size must be a tuple of (height, width)"
        self.height = output_size[0]
        self.width = output_size[1]
        if isinstance(fill, int):
            self.fill = (fill, fill, fill)
        else:
            self.fill = tuple(fill)

    def __call__(self, img):
        if isinstance(img, torch.Tensor):
            img = img.permute(1, 2, 0).cpu().numpy()
        h, w = img.shape[:2]
        scale = min(self.height / h, self.width / w)
        new_w, new_h = int(round(w * scale)), int(round(h * scale))
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)

        canvas = np.full((self.height, self.width, 3), self.fill, dtype=resized.dtype)
        y_off = (self.height - new_h) // 2
        x_off = (self.width - new_w) // 2
        canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized
        return canvas

class SobelFilter:
    def __init__(self, ksize=7):
        self.ksize = ksize
    def __call__(self, img):
        # img: numpy array HxWxC
        if isinstance(img, torch.Tensor):
            img = img.permute(1, 2, 0).cpu().numpy()
        img_f = img.astype(np.float32)
        sx = cv2.Sobel(img_f, cv2.CV_32F, 1, 0, ksize=self.ksize)
        sy = cv2.Sobel(img_f, cv2.CV_32F, 0, 1, ksize=self.ksize)
        mag = cv2.magnitude(sx, sy)
        magnitude = cv2.convertScaleAbs(mag)
        return magnitude

class ImageFileDataset(Dataset):
    def __init__(self, data_dir, image_size, sobel=False):
        assert len(image_size) == 2, "image_size must be a tuple of (height, width)"
        self.data_dir = data_dir
        self.image_files = [f for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, f))]
        self.samples = []
        self.labels_map = {}
        self.num_classes = 0
        self._parse_files()
        self.transform = T.Compose([
            SobelFilter() if sobel else T.Lambda(lambda x: x),
            ResizeAndPad(image_size) if image_size else T.Lambda(lambda x: x),
            T.ToTensor(),  # Converts HWC uint8 [0,255] to CHW float [0,1]
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
        ])

    def _parse_files(self):
        temp_labels = set()
        for fname in self.image_files:
            try:
                base = os.path.splitext(fname)[0]
                name_code, label_str = base.rsplit('_', 1)
                self.samples.append((fname, label_str))
                temp_labels.add(label_str)
            except Exception:
                print(f"Warning: Skipping file with incorrect format: {fname}")
        sorted_labels = sorted(temp_labels)
        self.labels_map = {label: i for i, label in enumerate(sorted_labels)}
        self.num_classes = len(sorted_labels)
        print(f"Found {len(self.samples)} samples belonging to {self.num_classes} classes.")
        print(f"Label mapping: {self.labels_map}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_name, label_str = self.samples[idx]
        path = os.path.join(self.data_dir, file_name)
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            print(f"Error loading image {path}")
            return torch.randn(3, 224, 224), -1
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.transform(img)
        label_idx = self.labels_map[label_str]
        return img, label_idx
