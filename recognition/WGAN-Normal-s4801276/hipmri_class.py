import os
import torch
import torch.nn.functional as F
import glob
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
import numpy as np

ROOT = "/HipMRI/"

class HipMRIDataset(Dataset):
    def __init__(self, dir=ROOT, seg="train"):
        # Path to files
        path = os.path.join(dir, f"keras_slices_{seg}", "*.nii.gz")

        # Return files matching the path above
        self.img_paths = sorted(glob.glob(path))