import os
import torch
import torchvision
from PIL import Image
from torch.utils.data import Dataset
import glob

class Ped2Dataset(Dataset):
    """
    Dataset class for PED2 (Pedestrian 2) dataset for anomaly detection.
    This is essentially an alias for PED2Dataset in load_dataset.py, for consistency.
    """
    def __init__(self, data_path, is_train=True, transform=None):
        self.transform = transform
        
        # Set root directory based on train/test
        if is_train:
            self.root_dir = os.path.join(data_path, 'training', 'frames')
        else:
            self.root_dir = os.path.join(data_path, 'testing', 'frames')
            
        # Get all image paths
        self.image_paths = []
        
        # Get all subdirectories (Train001, Train002, etc. or Test001, Test002, etc.)
        subdirs = [d for d in os.listdir(self.root_dir) if os.path.isdir(os.path.join(self.root_dir, d))]
        
        for subdir in subdirs:
            subdir_path = os.path.join(self.root_dir, subdir)
            # Get all jpg files in this subdirectory
            frame_paths = sorted(glob.glob(os.path.join(subdir_path, '*.jpg')))
            self.image_paths.extend(frame_paths)
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        # Target is 0 for all images (in self-supervised learning we don't need class labels)
        target = 0
        
        if self.transform:
            if hasattr(self.transform, '__call__'):
                # For regular transform
                image = self.transform(image)
            else:
                # For DataAugmentationSiT and similar transforms that return multiple values
                return self.transform(image), target
                
        return image, target