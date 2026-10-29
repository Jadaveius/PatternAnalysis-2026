import torch
import json
import vutils
from pathlib import Path
from hipmri_class import HipMRIDataset
from main_model import Generator, nz
from torchmetrics.functional.image import structural_similarity_index_measure as ssim
from torchmetrics.image.fid import FrechetInceptionDistance

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

def get_sample_grids(name, generator):
    for seed in [1, 2, 3]:
        torch.manual_seed(seed)

        with torch.no_grad():
            samples = generator(torch.randn(32, nz, 1, 1, device=device))

        vutils.save_image(samples, BASE / "outputs" / f"grid_{name}_seed{seed}.png",
                                  nrow=8, padding=2, normalize=True, value_range=(-1, 1))

# Need to change images from [-1, 1] grayscale to [0, 1] RGB
# Repeat gray channel 3 times to have 3 channels for RGB
to_rgb_01 = lambda x: ((x + 1) / 2).clamp(0, 1).repeat(1, 3, 1, 1)

def compute_fid(fid, gen, batch=100):
    fid.reset() # Clear generated features

    for s in range(0, gen.size(0), batch):
        fid.update(to_rgb_01(gen[s:s + batch]), real=False)

    return fid.compute().item()


if __name__ == "__main__":

    val_set = HipMRIDataset(seg="validate")
    test_set = HipMRIDataset(seg="test")

    if (VALIDATING):
        ref = torch.stack([val_set[i] for i in range(len(val_set))]).to(device)

        fid = FrechetInceptionDistance(feature=2048, normalize=True,
                                       reset_real_features=False).to(device)
        for s in range(0, ref.size(0), 100):
            fid.update(to_rgb_01(ref[s:s + 100]), real=True)

        ssim_results = {}
        fid_results = {}

        # For each checkpoint in the folder, validate
        for file_path in sorted(folder_path.glob("*.pt")):
            netG = load_generator(file_path)
            torch.manual_seed(0) # Same noise for all checkpoints
            
            with torch.no_grad():
                gen = netG(torch.randn(NUM_SAMPLES, nz, 1, 1, device=device))

            scores = best_ssim_match(gen, ref)

            ssim_results[file_path.name] = scores.mean().item()
            fid_results[file_path.name] = compute_fid(fid, gen)

            print(f"{file_path.name}: mean best-match SSIM {ssim_results[file_path.name]:.4f}")
            print(f"{file_path.name}: FID {fid_results[file_path.name]:.4f}")

        best_model_ssim = max(ssim_results, key=ssim_results.get) # best SSIM is large
        best_model_fid = min(fid_results, key=fid_results.get) # best FID is small
        print(f"\nBest checkpoint: {best_model_ssim} (val SSIM {ssim_results[best_model_ssim]:.4f})")
        print(f"\nBest checkpoint: {best_model_fid} (val FID {fid_results[best_model_fid]:.4f})")

        with open(BASE / "outputs" / f"val_results_{GEN_PATH.parent.name}.json", "w") as f:
            json.dump({"ssim": ssim_results, "fid": fid_results}, f, indent=2)

    # Should only be run once checkpoints have been picked
    else:
        FINAL_MODELS = {
        "DCGAN": (BASE / "outputs/dcgan/checkpoints/dcgan_epoch_050.pt", "netG"),
        "WGAN-GP standard": (BASE / "outputs/wgan_gp_standard/checkpoints/wgan_gp_epoch_XXX.pt", "netG"),
        "WGAN-GP tuned": (BASE / "outputs/wgan_gp/checkpoints/wgan_gp_epoch_XXX.pt", "<EMA key>"),
        }

        # Same ref and real FID is used for all the models being tested
        ref = torch.stack([test_set[i] for i in range(len(test_set))]).to(device)
        fid = FrechetInceptionDistance(feature=2048, normalize=True,
                                               reset_real_features=False).to(device)
        for s in range(0, ref.size(0), 100):
            fid.update(to_rgb_01(ref[s:s + 100]), real=True)

        test_results = {}
        # For each different model, evaluate it
        for name, (path, key) in FINAL_MODELS.items():
            torch.manual_seed(0) # Same noise as validity testing       
            netG = load_generator(path, key)
            with torch.no_grad():
                gen = netG(torch.randn(NUM_SAMPLES, nz, 1, 1, device=device))

            get_sample_grids(name, gen)
            scores = best_ssim_match(gen, ref)
            fid_score = compute_fid(fid, gen)

            # Save results in dict for log/ease of find later
            test_results[name] = {"checkpoint": path.name,
                                  "ssim_mean": scores.mean().item(),
                                  "ssim_std": scores.std().item(),
                                  "fid": fid_score}
            print(f"{name}: SSIM {scores.mean():.4f} ± {scores.std():.4f} | FID {fid_score:.2f}")

        # Save results from model evaluation
        with open(BASE / "outputs" / "test_results.json", "w") as f:
            json.dump(test_results, f, indent=2)
