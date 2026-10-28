import os
import time
import torch
import torch.nn as nn
from torch.nn import Module
import torch.optim as optim
from torch.utils.data import DataLoader
from hipmri_class import HipMRIDataset
import torchvision.utils as vutils
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
import numpy as np

# NOTES - For myself to be removed/moved to README
# No fully connected or pooling layers are used

# Generator starts with 100-dimensional latent vector, reshapes into 3D and does convlutions to 64x64
# Discriminator starts at 128x64, does convolutions and then binary logistic regression

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
num_epochs = 50 # Number of training epochs
lr = 0.0001 # Learning rate for optimizers
beta1 = 0.0 # Beta1 hyperparameter for Adam optimisers
beta2 = 0.9 # Beta2 hyperparameter for Adaom optimisers
ngpu = 1 # Number of GPUs available. Use 0 for CPU mode.
n_critic = 5 # Number of times critic runs before generator
lambda_gp = 10 # Multiplcative scalar for gradient penalty

OUT_DIR = "outputs/wgan_gp"
os.makedirs(os.path.join(OUT_DIR, "samples"), exist_ok=True)
os.makedirs(os.path.join(OUT_DIR, "checkpoints"), exist_ok=True)
SAVE_EVERY = 5  # epochs

# DCGAN paper says model weights are randomly initialised from normal distribution
# with mean = 0 and stdev = 0.02
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

# GENERATOR
# Latent space vector (z) must be mapped to training images (1 x 128 x 64)
# Strided 2-dim convolutional transpose layers
# MUST batchnorm after conv to be DCGAN

class Generator(Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( nz, ngf * 8, (8, 4), 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. ``(ngf*8) x 8 x 4``
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. ``(ngf*4) x 16 x 8``
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. ``(ngf*2) x 32 x 16``
            nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. ``(ngf) x 64 x 32``
            nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. ``(nc) x 128 x 64``
        )

    def forward(self, input):
        return self.main(input)

# DISCRIMINATOR
# Binary classification network that takes (1 x 128 x 64) input and outputs final score
# Essentially opposite of Generator but uses Conv2D and LeakyReLU
# NO FINAL SIGMOID unlike DCGAN because we want raw score, not probability
# NO BATCH NORM as it creates correlations between samples in the same batch
class Discriminator(Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is ``(nc) x 128 x 64``
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf) x 64 x 32``
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.GroupNorm(1, ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*2) x 32 x 16``
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.GroupNorm(1, ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*4) x 16 x 8``
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.GroupNorm(1, ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*8) x 8 x 4``
            nn.Conv2d(ndf * 8, 1, (8, 4), 1, 0, bias=False),
        )

    def forward(self, input):
        return self.main(input)

# RUNNING MODEL

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load datasets
    train_set = HipMRIDataset(seg="train")
    val_set = HipMRIDataset(seg="validate")

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True)

    # Create the generator
    netG = Generator(ngpu).to(device)
    # Apply the ``weights_init`` function to randomly initialize all weights
    #  to ``mean=0``, ``stdev=0.02``.
    netG.apply(weights_init)

    # Create the discriminator
    netD = Discriminator(ngpu).to(device)
    # Apply the ``weights_init`` function to randomly initialize all weights
    # like this: ``to mean=0, stdev=0.2``.
    netD.apply(weights_init)

    # Create batch of latent vectors that we will use to visualize
    #  the progression of the generator
    fixed_noise = torch.randn(16, nz, 1, 1, device=device)

    # Establish convention for real and fake labels during training
    real_label = 1.
    fake_label = 0.

    # Setup Adam optimizers for both G and D
    optimiserD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, beta2))
    optimiserG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, beta2))

    # Device/param stats since I'm running on my own PC
    gpu_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"
    n_params_G = sum(p.numel() for p in netG.parameters())
    n_params_D = sum(p.numel() for p in netD.parameters())

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    # Training Loop

    # Lists to keep track of progress
    img_list = []
    G_losses = []
    D_losses = []
    W_estimates = []
    GP_values = []
    epoch_times = [] # Time taken for each epoch
    iters = 0

    print("Starting Training Loop...")
    # For each epoch
    for epoch in range(num_epochs):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
    
        # For each batch in the dataloader
        for i, data in enumerate(train_loader, 0):
            
            ############################
            # (1) Update D network: D(G(z)) - D(x) + lambda_gp*(norm(grad(D(x_hat), x_hat)) - 1)^2
            ###########################
            ## Train with all-real batch
            netD.zero_grad()
            # Format batch
            real_cpu = data.to(device)
            b_size = real_cpu.size(0)
            
            ## Train with all-fake batch
            # Generate batch of latent vectors
            noise = torch.randn(b_size, nz, 1, 1, device=device)
            # Generate fake image batch with G
            fake = netG(noise).detach() # Detach means no gradients into G
            
            # Get the Wasserstian loss (for critic)
            epsilon = torch.rand(b_size, 1, 1, 1, device=device) # Get epsilon for each sample
            # Sampled points somewhere between real and fake
            x_hat = (epsilon*real_cpu + (1 - epsilon) * fake).requires_grad_(True)
            # Get gradient of D(x_hat)
            grad = torch.autograd.grad(netD(x_hat).sum(), x_hat, create_graph=True)[0]
            # The gradient penalty portion is (normalised grad - 1) ^ 2
            gp = ((grad.flatten(1).norm(2, dim=1) - 1) ** 2).mean()

            d_real = netD(real_cpu).mean()
            d_fake = netD(fake).mean()
            loss_D = d_fake - d_real + lambda_gp * gp

            # Update D
            loss_D.backward() # Gradients for D
            optimiserD.step()

            W_estimate = (d_real - d_fake).item() # logging for plot later

            W_estimates.append(W_estimate)
            GP_values.append(gp.item())

            # The generator is only updated for every 5 iterations of the critic,
            # so the generator only gets (i_max / n_critic) iterations in an epoch
            if (i + 1) % n_critic == 0:
                ############################
                # (2) Update G network: minimise -D(G(z))
                ###########################
                netG.zero_grad()

                # Generate batch of latent vectors
                noise = torch.randn(b_size, nz, 1, 1, device=device)
                # Get the Wasserstian loss (for generator)
                loss_G = -netD(netG(noise)).mean()
                loss_G.backward()  # Calculate gradients for G
                # Update G
                optimiserG.step()

                G_losses.append(loss_G.item()) # Save Losses for plotting later

            # Output training stats
            if i % 50 == 0:
                print(f"[{epoch}/{num_epochs}][{i}/{len(train_loader)}]  "
                        f"Loss_D: {loss_D.item():.4f}  W: {W_estimate:.4f}  GP: {gp.item():.4f}")

            D_losses.append(loss_D.item()) # Save Losses for plotting later
            iters += 1

        if device.type == "cuda":
            torch.cuda.synchronize()
        epoch_times.append(time.time() - t0)
        print(f"Epoch {epoch + 1} took {epoch_times[-1]:.1f} s")

         # Sample grid from the fixed noise (same z every time, so grids are comparable)
        with torch.no_grad():
            fake = netG(fixed_noise).cpu()
        grid = vutils.make_grid(fake, nrow=8, padding=2, normalize=True, value_range=(-1, 1))
        img_list.append(grid)  # for the animation
        vutils.save_image(grid, os.path.join(OUT_DIR, "samples", f"epoch_{epoch + 1:03d}.png"))

        # Checkpoint every SAVE_EVERY epochs and at the final epoch
        if (epoch + 1) % SAVE_EVERY == 0 or epoch == num_epochs - 1:
            torch.save({
                "epoch": epoch + 1,
                "netG": netG.state_dict(),
                "netD": netD.state_dict(),
                "optG": optimiserG.state_dict(),
                "optD": optimiserD.state_dict(),
                "G_losses": G_losses,
                "D_losses": D_losses,
                "GP_values": GP_values,
                "W_estimates": W_estimates
            }, os.path.join(OUT_DIR, "checkpoints", f"wgan_gp_epoch_{epoch + 1:03d}.pt"))
            print(f"Saved checkpoint at epoch {epoch + 1}")

    # Peak VRAM required by the DCGAN
    peak_vram = torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else float("nan")

    # Generation (inference) time for 64 images
    netG.eval()
    with torch.no_grad():
        z = torch.randn(64, nz, 1, 1, device=device)
        netG(z)                                    # warm-up run
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        netG(z)
        if device.type == "cuda":
            torch.cuda.synchronize()
        gen_ms = (time.time() - t0) * 1000

    summary = (f"Device: {gpu_name}\n"
                f"G params: {n_params_G:,} | D params: {n_params_D:,}\n"
                f"Peak VRAM: {peak_vram:.0f} MB\n"
                f"Mean epoch time: {sum(epoch_times) / len(epoch_times):.1f} s "
                f"(total {sum(epoch_times) / 60:.1f} min for {num_epochs} epochs)\n"
                f"Generate 64 images: {gen_ms:.1f} ms")
    print(summary)

    # Save text summary for later
    with open(os.path.join(OUT_DIR, "resources.txt"), "w") as f:
        f.write(summary)

    # Loss versus training iteration
    plt.figure(figsize=(10,5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses,label="G")
    plt.plot(D_losses,label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("wgan_losses.png", dpi=150)
    plt.close()