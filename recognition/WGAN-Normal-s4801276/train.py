"""
Source code for training, validating, testing and saving DCGAN or WGAN-GP models
for HipMRI 2D slices. Results were produced with pre-structure scripts (see commits)
and moved into the correct format.
"""

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