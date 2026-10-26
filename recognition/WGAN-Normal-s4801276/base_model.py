import time
import torch

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