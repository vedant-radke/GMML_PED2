from PIL import Image
import torch
import torchvision
from torchvision import transforms
from datasets.datasets_utils import GaussianBlur, Solarization, GMML_drop_rand_patches

class DataAugmentationPed2(object):
    def __init__(self, args):
        # for corruption
        self.drop_perc = args.drop_perc
        self.drop_type = args.drop_type
        self.drop_align = args.drop_align

        # Adapt to the specific needs of PED2 dataset (grayscale images)
        self.rand_resize_flip = transforms.Compose([
            transforms.RandomResizedCrop(args.img_size, scale=(0.25, 1.0), interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5)])
        
        # No need for color jitter for grayscale images, but keeping light augmentation
        self.clean_transfo = transforms.Compose([
            GaussianBlur(0.1),
            transforms.Grayscale(num_output_channels=3),  # Convert to 3 channels for ViT
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        # Stronger augmentation for corruption
        self.corrupt_transfo = transforms.Compose([
            GaussianBlur(1.0),
            Solarization(0.2),
            transforms.Grayscale(num_output_channels=3),  # Convert to 3 channels for ViT
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def __call__(self, image):
        clean_crops = []
        corrupted_crops = []
        masks_crops = []

        # First crop
        im = self.rand_resize_flip(image)
        im_orig = self.clean_transfo(im)
        
        im_corrupted = im_orig.detach().clone()
        im_mask = torch.zeros_like(im_corrupted)
        if self.drop_perc > 0:
            im_corrupted, im_mask = GMML_drop_rand_patches(im_corrupted, max_replace=self.drop_perc, drop_type=self.drop_type, align=self.drop_align)

        clean_crops.append(im_orig)
        corrupted_crops.append(im_corrupted)
        masks_crops.append(im_mask)

        # Second crop
        im = self.rand_resize_flip(image)
        im_orig = self.clean_transfo(im)
        
        im_corrupted = im_orig.detach().clone()
        im_mask = torch.zeros_like(im_corrupted)
        if self.drop_perc > 0:
            im_corrupted, im_mask = GMML_drop_rand_patches(im_corrupted, max_replace=self.drop_perc, drop_type=self.drop_type, align=self.drop_align)

        clean_crops.append(im_orig)
        corrupted_crops.append(im_corrupted)
        masks_crops.append(im_mask)

        return clean_crops, corrupted_crops, masks_crops