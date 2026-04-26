import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import numpy as np


class ClassificationDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        
        self.samples = []
        self.class_to_idx = {}
        
        self.load_dataset()
    
    def load_dataset(self):
        classes = sorted(os.listdir(self.root_dir))
        
        for idx, cls in enumerate(classes):
            cls_path = os.path.join(self.root_dir, cls)
            if  not os.path.isdir(cls_path):
                continue
            
            self.class_to_idx[cls] = idx
            
            for img_name in os.listdir(cls_path):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(cls_path, img_name)
                    self.samples.append((img_path, idx))
        
        print(f"Loaded {len(self.samples)} samples from {self.root_dir}")

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image=np.array(image))["image"]
        
        return image, label


def compute_class_weights(dataset):
    labels = [label for _, label in dataset.samples]
    class_counts = np.bincount(labels)
    
    class_weights = 1.0 / (class_counts + 1e-6)
    class_weights = class_weights / np.sum(class_weights) * len(class_counts)
    
    return torch.tensor(class_weights, dtype=torch.float)


def create_weight_sampler(dataset):
    labels = [label for _, label in dataset.samples]
    class_counts = np.bincount(labels)
    
    class_weights = 1.0 / (class_counts + 1e-6)
    sample_weights = [float(class_weights[label]) for label in labels]
    
    return torch.utils.data.WeightedRandomSampler(
        sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )