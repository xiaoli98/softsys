import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

try:
    import Image
except ImportError:
    from PIL import Image



class Crop2BBox:
    def __init__(self, threshold=0.6):
        self.threshold = threshold
    def __call__(self, img):
        img = cv2.threshold(img, 0.6, 1, cv2.THRESH_BINARY)[1]
        
        img = (img * 255).astype(np.uint8)
        
        contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            print("No white objects found in the image.")
            return None
        largest_contour = max(contours, key=cv2.contourArea)

        x, y, w, h = cv2.boundingRect(largest_contour)
        return img[y:y+h, x:x+w]
    
class ResizeAndPad:
    def __init__(self, output_size, fill=0):
        assert len(output_size) == 2, "Output size must be a tuple of (height, width)"
        self.height = output_size[0]
        self.width = output_size[1]
        self.fill = fill

    def __call__(self, img):
        h, w = img.shape[:2]
        scale = min(self.height / h, self.width / w)
        new_w, new_h = int(round(w * scale)), int(round(h * scale))
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)

        canvas = np.full((self.height, self.width), self.fill, dtype=resized.dtype)
        y_off = (self.height - new_h) // 2
        x_off = (self.width - new_w) // 2
        canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized
        return canvas

class SobelFilter:
    def __init__(self, ksize=7):
        self.ksize = ksize
    def __call__(self, img):
        img_f = img.astype(np.float32)
        sx = cv2.Sobel(img_f, cv2.CV_32F, 1, 0, ksize=self.ksize)
        sy = cv2.Sobel(img_f, cv2.CV_32F, 0, 1, ksize=self.ksize)
        mag = cv2.magnitude(sx, sy)
        magnitude = cv2.convertScaleAbs(mag)
        return magnitude

class Normalize:
    def __call__(self, img):
        img = img.astype(np.float32)
        min_val = np.min(img)
        max_val = np.max(img)
        
        if max_val > min_val:
            img = (img - min_val) / (max_val - min_val)
        else:
            img.fill(0)
        return img
    
class Standardize:
    def __call__(self, img):
        img = (img - img.mean()) / img.std()
        return img

class ImageFileDataset(Dataset):
    def __init__(self, data_dir, image_size, sobel=False, contrastive=True):
        assert len(image_size) == 2, "image_size must be a tuple of (height, width)"
        self.data_dir = data_dir
        self.image_files = [f for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, f))]
        self.samples = []
        self.labels_map = {}
        self.num_classes = 0
        self._parse_files()
        self.contrastive = contrastive
        transforms = [
            Normalize(),
            Standardize(),
            Crop2BBox(),
            # ResizeAndPad(image_size),
            SobelFilter() if sobel else lambda x: x,
            T.ToTensor(),
        ]
        if contrastive:
            transforms.append(T.RandomResizedCrop(size=image_size, scale=(0.2, 1.)))
            transforms.append(T.RandomHorizontalFlip())
            transforms.append(T.RandomRotation(degrees=15))
            
        self.transform = T.Compose(transforms)

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
        img = Image.open(path)
        assert img is not None, f"Failed to load image {path}"
        
        img_np = np.array(img)
        
        if self.contrastive:
            img_transformed = (self.transform(img_np), self.transform(img_np))
        else:
            img_transformed = self.transform(img_np)
        
        label_idx = self.labels_map[label_str]
        return img_transformed, np.array([label_idx], dtype=np.float32)
