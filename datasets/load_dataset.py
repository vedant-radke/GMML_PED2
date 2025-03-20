import os
import torch
import torchvision
from PIL import Image
from torch.utils.data import Dataset
import glob

class PED2Dataset(Dataset):
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

def build_dataset(args, is_train, trnsfrm=None, training_mode='finetune'):
    if args.data_set == 'MNIST':
        dataset = torchvision.datasets.MNIST(os.path.join(args.data_location, 'MNIST_dataset'), 
                                   train=is_train, transform=trnsfrm, download=True)
        nb_classes = 10
    elif args.data_set == 'PED2':
        # dataset = PED2Dataset(os.path.join(args.data_location, 'PED2'), 
        #                       is_train=is_train, transform=trnsfrm)
        dataset = PED2Dataset(args.data_location, 
                      is_train=is_train, transform=trnsfrm)

        nb_classes = 1  # For autoencoder/reconstruction tasks, class labels aren't needed
    else:
        raise ValueError(f"Dataset {args.data_set} not supported")
        
    return dataset, nb_classes