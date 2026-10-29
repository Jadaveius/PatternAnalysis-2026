import torch
from hipmri_class import HipMRIDataset
from main_model import Generator, nz
from torchmetrics.functional.image import structural_similarity_index_measure as ssim

VALIDATING = 1

# For each generated image find the highest SSIM against a reference slice
# (can be validate or test)
def best_ssim_match(gen, ref, chunk=256):
    best = torch.full((gen.size(0),), -1.0, device=gen.device)

    for n in range(gen.size(0)):
        # For each generated image
        g = gen[n:n + 1]
        # Compare against some reference images
        for s in range(0, ref.size(), chunk):
            r = ref[s:s + 1]
            # Get SSIM, data_range is because images are across [-1, 1]
            score = ssim(g.expand_as(r), r, data_range=2.0, reduction=None)
            # Each generated image is stored with the best found score
            best[n] = torch.maximum(best[n], score)

    return best

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    val_set = HipMRIDataset(seg="validate")
    test_set = HipMRIDataset(seg="test")

    if (VALIDATING):
        ref = torch.stack([val_set[i] for i in range(len(val_set))]).to(device)

        with torch.no_grad():
            gen = netG(torch.randn(1000, nz, 1, 1, device=device))

        scores = best_ssim_match(gen, ref)
        print(f"Mean best-match SSIM: {scores.mean():.4f} (target > 0.6)")

    # Should only be run once checkpoints have been picked
    else:
        ref = torch.stack([test_set[i] for i in range(len(test_set))]).to(device)