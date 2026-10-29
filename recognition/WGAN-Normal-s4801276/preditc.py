import torch
from pathlib import Path
from hipmri_class import HipMRIDataset
from main_model import Generator, nz
from torchmetrics.functional.image import structural_similarity_index_measure as ssim

VALIDATING = 1
NUM_SAMPLES = 256

BASE = Path(__file__).resolve().parent
GEN_PATH = BASE / "outputs" / "wgan_gp" / "checkpoints"

folder_path = GEN_PATH
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Can load the generator data from a saved model
def load_generator(path, key="netG"):
    # Get the saved model
    model = torch.load(path, map_location=device)
    print(path, "keys:", list(model.keys()), "epoch:", model["epoch"])

    netG = Generator(ngpu=1).to(device)
    netG.load_state_dict(model[key])
    netG.eval()

    return netG

# For each generated image find the highest SSIM against a reference slice
# (can be validate or test)
def best_ssim_match(gen, ref, chunk=256):
    best = torch.full((gen.size(0),), -1.0, device=gen.device)

    for n in range(gen.size(0)):
        # For each generated image
        g = gen[n:n + 1]
        # Compare against some reference images
        for s in range(0, ref.size(0), chunk):
            r = ref[s:s + chunk]
            # Get SSIM, data_range is because images are across [-1, 1]
            scores = ssim(g.expand_as(r), r, data_range=2.0, reduction=None)
            # Each generated image is stored with the best found score
            best[n] = torch.maximum(best[n], scores.max())

    return best

if __name__ == "__main__":

    val_set = HipMRIDataset(seg="validate")
    test_set = HipMRIDataset(seg="test")

    if (VALIDATING):
        ref = torch.stack([val_set[i] for i in range(len(val_set))]).to(device)

        # For each checkpoint in the folder, validate
        results = {}
        for file_path in sorted(folder_path.glob("*.pt")):
            netG = load_generator(file_path)
            torch.manual_seed(0) # Same noise for all checkpoints
            
            with torch.no_grad():
                gen = netG(torch.randn(NUM_SAMPLES, nz, 1, 1, device=device))

            scores = best_ssim_match(gen, ref)
            results[file_path.name] = scores.mean().item()
            print(f"{file_path.name}: mean best-match SSIM {results[file_path.name]:.4f}")

        best_model = max(results, key=results.get)
        print(f"\nBest checkpoint: {best_model} (val SSIM {results[best_model]:.4f})")

    # Should only be run once checkpoints have been picked
    else:
        ref = torch.stack([test_set[i] for i in range(len(test_set))]).to(device)