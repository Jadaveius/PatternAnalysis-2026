"""
Source code for training, validating, testing and saving DCGAN or WGAN-GP models
for HipMRI 2D slices. Results were produced with pre-structure scripts (see commits)
and moved into the correct format.
"""

import os
import time
import torch
from pathlib import Path
import torchvision.utils as vutils

BASE = Path(__file__).resolve().parent

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


def setup_out_dir(name):
    """
    Set up the output directory for results of this model training.
    """
    out_dir = BASE / "outputs" / name

    os.makedirs(os.path.join(out_dir, "samples"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "checkpoints"), exist_ok=True)

    return out_dir

def save_sample_grid(netG, noise, out_dir, epoch):
    """
    Saves a sample grid of current generator output for some noise and
    saves the image result. For purpose of viewing training progression
    and creating generator training animations.
    """
    # Sample grid from the fixed noise (same z every time, so grids are comparable)
    with torch.no_grad():
        # Noise SHOULD BE SAME (seeded) for all epochs of training
        fake = netG(noise).cpu()

    grid = vutils.make_grid(fake, nrow=8, padding=2, normalize=True, value_range=(-1, 1))
    vutils.save_image(grid, os.path.join(out_dir, "samples", f"epoch_{epoch + 1:03d}.png"))

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

def profile_generation(netG, nz, device):
    """
    Will return the generation (inference) time for 64 images with
    the generator at its current epoch.
    """
    training = netG.training

    netG.eval()
    with torch.no_grad():
        # Generate noise
        z = torch.randn(64, nz, 1, 1, device=device)
        netG(z)

        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        netG(z)

        if device.type == "cuda":
            torch.cuda.synchronize()
        gen_ms = (time.time() - t0) * 1000

    netG.train(training)
    return gen_ms