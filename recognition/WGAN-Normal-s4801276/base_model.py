import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from hipmri_class import HipMRIDataset

# NOTES - For myself to be removed/moved to README
# No fully connected or pooling layers are used

# Generator starts with 100-dimensional latent vector, reshapes into 3D and does convlutions to 64x64
# Discriminator starts at 64x64, does convolutions and then binary logistic regression

# Discriminator is same as generator but opposite

# Use ReLU activation in generator for all layers except output with is Tanh
# Use Leaky ReLu activation in all discriminator layers

# Discriminator is HIGH when x is training data and LOW when x is generated

# PARAMETERS - Based on PyTorch DCGAN Tutorial
dataroot = "/HipMRI/" # Root directory for dataset
workers = 1 # Number of workers for dataloader
batch_size = 128 # Batch size during training
# Spatial size of training images. All images will be resized to this
#   size using a transformer.
image_size = 64
nc = 1 # Number of channels in the training images. Greyscale image is 1
nz = 100 # Size of z latent vector (i.e. size of generator input)
ngf = 64 # Size of feature maps in generator
ndf = 64 # Size of feature maps in discriminator
num_epochs = 5 # Number of training epochs
lr = 0.0002 # Learning rate for optimizers
beta1 = 0.5 # Beta1 hyperparameter for Adam optimizers
ngpu = 1 # Number of GPUs available. Use 0 for CPU mode.

# DCGAN paper says model weights are randomly initialised from normal distribution
# with mean = 0 and stdev = 0.02
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)



# RUNNING MODEL

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load datasets
train_set = HipMRIDataset(seg="train")
val_set = HipMRIDataset(seg="validate")

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                          num_workers=workers)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True,
                          num_workers=workers)