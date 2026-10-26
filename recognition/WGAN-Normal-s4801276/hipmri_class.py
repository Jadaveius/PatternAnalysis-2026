import os
import torch
import torch.nn.functional as F
import glob
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
import numpy as np

ROOT = "/HipMRI/" # Root directory for dataset
NC = 1 # Number of channels in the training images. Greyscale image is 1
# Spatial size of training images. All images will be resized to this
#   size using a transformer.
IMAGE_SIZE_H = 256
IMAGE_SIZE_W = 128

class HipMRIDataset(Dataset):
    def __init__(self, dir=ROOT, seg="train"):
        # Path to files
        path = os.path.join(dir, f"keras_slices_{seg}", "*.nii.gz")

        # Return files matching the path above
        self.img_paths = sorted(glob.glob(path))

        self.transform = transforms.Compose(
            [transforms.Grayscale(num_output_channels=NC),
             transforms.Resize(IMAGE_SIZE_H, IMAGE_SIZE_W),
             transforms.CenterCrop(IMAGE_SIZE_H, IMAGE_SIZE_W),
             transforms.ToTensor()]
        )