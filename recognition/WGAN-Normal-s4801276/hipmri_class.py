import os
import glob
import torch
from torch.utils.data import Dataset
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt


# Root directory for dataset
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "HipMRI") 
# Spatial size of training images. All images will be resized to this
# size using a transformer.
IMAGE_SIZE_H = 128
IMAGE_SIZE_W = 64
PERCENTILE = 99.5 # Clipping outlier pixels

class HipMRIDataset(Dataset):
    def __init__(self, dir=ROOT, seg="train"):
        # Path to files
        path = os.path.join(dir, f"keras_slices_{seg}", "*.nii.gz")

        # Return files matching the path above
        self.img_paths = sorted(glob.glob(path))

    def __getitem__(self, index):
        # Load a single sample, apply transforms and return the image
        img_path = self.img_paths[index]
        img = nib.load(img_path).get_fdata() # Gives NumPy array

        # Center Crop
        h, w = img.shape
        # Round down to nearest int
        top, left = (h - IMAGE_SIZE_H) // 2, (w - IMAGE_SIZE_W) // 2 
        img = img[top:top + IMAGE_SIZE_H, left:left + IMAGE_SIZE_W]

        # Min-Max Scale [-1, 1]
        max_val = np.percentile(img, PERCENTILE)
        img = np.clip(img, 0, max_val)
        # Scale to [-1, 1] for tanh at end of generator
        img = img / max_val * 2 - 1 

        # To Tensor
        img = torch.from_numpy(img).float().unsqueeze(0) # Add the 1 channel dimension (grayscale)

        return img

    def __len__(self):
        return len(self.img_paths)