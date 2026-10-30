"""
Contains helper functions that are used in both train.py as well as
predict.py, mostly to assist with validating already trained models.
"""

import torch
from modules import Generator
from torchmetrics.functional.image import structural_similarity_index_measure as ssim

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

def invert_ssim(netG, ref, nz, steps=500, lr=0.05, z_reg=1e-3, batch=64, seed=0):
    """
    Compute a GAN inversion. For each slice, optimise a latent z so that G(z) approximates
    slice, then compute the paired SSIM between the slice and its reconstruction G(z). SSIM
    is done this way because MRI slices are too varied to get valuable SSIM score
    from finding maximum SSIM against test slices (and would reward memorisation).
    """

    netG.eval()
   
    for p in netG.parameters():
        p.requires_grad_(False) # only z is optimised, not the generator

    torch.manual_seed(seed)

    recons, scores = [], []
    for s in range(0, ref.size(0), batch):
        x = ref[s:s + batch]
        z = torch.randn(x.size(0), nz, 1, 1, device=x.device, requires_grad=True)
        opt = torch.optim.Adam([z], lr=lr)

        for _ in range(steps):
            opt.zero_grad()
            loss = torch.nn.functional.mse_loss(netG(z), x) + z_reg * z.pow(2).mean()
            loss.backward()
            opt.step()

        with torch.no_grad():
            r = netG(z)
            recons.append(r)
            scores.append(ssim(r, x, data_range=2.0, reduction=None))

    for p in netG.parameters():
        p.requires_grad_(True)

    return torch.cat(recons), torch.cat(scores)