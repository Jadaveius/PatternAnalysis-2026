"""
Contains helper functions that are used in both train.py as well as
predict.py, mostly to assist with validating already trained models.
"""

import torch
from modules import Generator

def load_generator(path, device, nz, ngf):
    """
    Loads the generator data from a saved model.
    """
    # Get the saved model
    model = torch.load(path, map_location=device)
    print(path, "keys:", list(model.keys()), "epoch:", model["epoch"])

    netG = Generator(ngpu=1, nz=nz, ngf=ngf, nc=1).to(device)
    netG.load_state_dict(model["netG"])
    netG.eval()

    return netG