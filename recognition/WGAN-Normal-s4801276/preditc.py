import torch
from torchmetrics.functional.image import structural_similarity_index_measure as ssim

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