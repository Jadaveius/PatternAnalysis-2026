import os
import glob
import torchio as tio
from torchvision import transforms
from torch.utils.data import Dataset
import nibabel as nib



ROOT = "/HipMRI/" # Root directory for dataset
NC = 1 # Number of channels in the training images. Greyscale image is 1
# Spatial size of training images. All images will be resized to this
#   size using a transformer.
IMAGE_SIZE_H = 128
IMAGE_SIZE_W = 64

class HipMRIDataset(Dataset):
    def __init__(self, dir=ROOT, seg="train"):
        # Path to files
        path = os.path.join(dir, f"keras_slices_{seg}", "*.nii.gz")

        # Return files matching the path above
        self.img_paths = sorted(glob.glob(path))

        # Resizing is necessary because data audit found some training images were
        # of size 256x144 while all other data was 256x128
        self.transform = transforms.Compose(
            [transforms.Grayscale(num_output_channels=NC),
             transforms.Resize(IMAGE_SIZE_H, IMAGE_SIZE_W),
             transforms.CenterCrop(IMAGE_SIZE_H, IMAGE_SIZE_W),
             transforms.ToTensor()]
        )

    def __getitem__(self, index):
        # Load a single sample, apply transforms and return the image
        img_path = self.img_paths[index]
        img = nib.load(img_path)
        data = img.get_fdata()
        img = self.transform(img)

        return img

    def __len__(self):
        return len(self.img_paths)