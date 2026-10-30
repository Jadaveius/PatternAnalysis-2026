"""
Source code for training, validating, testing and saving DCGAN or WGAN-GP models
for HipMRI 2D slices. Results were produced with pre-structure scripts (see commits)
and moved into the correct format.
"""

import os
import torch

# Configurations of the different models that have been tested in this project
CONFIGS = {
    "dcgan": dict(model="dcgan", epochs=50, lr=0.0002, betas=(0.5, 0.999), nc=1, nz=100,
                  ngf=64, ndf=64, downsample=False),
    "wgan_gp": dict(model="wgan_gp", epochs=250, lr=0.0001, betas=(0.0, 0.9), nc=1, nz=100,
                  ngf=64, ndf=64, downsample=False),
    "wgan_gp_extend": dict(model="wgan_gp_extend", epochs=1000, lr=0.0001, betas=(0.0, 0.9), nc=1, nz=100,
                      ngf=64, ndf=64, downsample=False),
    "wgan_gp_wide": dict(model="wgan_gp_wide", epochs=1000, lr=0.0001, betas=(0.0, 0.9), nc=1, nz=128,
                      ngf=128, ndf=64, downsample=True)
}

def save_checkpoint(out_dir, prefix, epoch, state):
    """
    Saves a checkpoint with the current progress of the model, in case of
    failures mid-run or for validating at which point in epochs the model
    performed best.
    """
    state = {"epoch": epoch + 1, **state}
    path = os.path.join(out_dir, "checkpoints", f"{prefix}_epoch_{epoch + 1:03d}.pt")
    torch.save(state, path)

    print(f"Saved checkpoint at epoch {epoch + 1}")