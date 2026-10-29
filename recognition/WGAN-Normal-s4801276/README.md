# Synthetic Prostate MRI Generation with WGAN-GP (HipMRI 2D)

One-paragraph summary: what was built, the problem it solves, headline result
(e.g. "WGAN-GP reached SSIM X vs DCGAN Y with no evidence of memorisation"). - ADDED LAST
BASED ON RESULTS OF REPORT (like exec summary)

![Uncurated WGAN-GP samples](images/wgan_gp_final.png)

## Contents
(optional table of contents linking to the sections below)

## 1. Feasibility Review
**Topic 2.4 - Image Generation: Synthetic Medical Cohorts and Privacy**
### 1.1 User Need, Scope and Acceptance Criteria
The purpose of this research is to provide additional medical imaging resources to medical researchers investigating 2D pelvic MRI slices. Through WGAN generation, new slices could be synthesised which replicate realistic tissue structures and remove the risk of violating patient privacy laws such as HIPPA and GDPR. Final intentions for this project are to determine whether the model can produce synthetic images that are safe, structurally sound and could plausibly be used for training. To address the fidelity vs memorisation vs privacy dilemma, testable acceptance criteria of the final model include:
- Structural Similarity Index of greater than 0.6 (SSIM > 0.6) across synthesised cohorts against reference anatomical slices.
- WGAN results are better than those of the baseline model.
- Nearest-neighbour distance audit and latent space interpolation, sample distribution should not be closer than test distribution.
- Peak VRAM < 4 GB and < 6 hours GPU training.
### 1.2 Model Choice and Course Concepts
The chosen model is the WGAN-GP which is Normal difficulty, on HipMRI 2D slices which are already split into test, train and validation sets. The baseline model is a DCGAN which is identical apart from the loss and Lipschitz constraint.
- Must implement Earth-Mover distance. This is part of what makes it a WGAN, providing a continuous distance (differentiable almost everywhere) that allows the critic to be trained towards optimality. We can then train with low likelihood of mode collapse and image generation improving for higher number of iterations.
- Critic must have Lipschitz constraint. Based on the paper this is the other requirement that allows the critic to have a continuous loss function to be optimised.
- To be a WGAN-GP rather than just a WGAN, the loss function for the critic must include a gradient penalty scaled by some lambda (this penalty directly incorporates the Lipschitz constraint).
- Starting parameters for the system are based on those noted in the WGAN-GP paper, α=0.0001,λ=10,m=64,n_CRITIC=5 where these are learning rate, gradient penalty lambda, batch size and number of critic iterations per generator iteration, respectively.
- Builds directly on course concepts involving DCGANs and WGANs (both covered in Week 9 lecture content) as well as loss/activation functions, hyperparameters and model evaluation methods.
### 1.3 Preliminary Feasibility Evidence
An initial data audit was completed using `data_audit.py`, which can be summarised into this table:
<div align="center">

| **Measure** | **Train** | **Validate** | **Test** |
| --- | :---: | :---: | :---: |
| Slices | 11,460 | 660 | 540 |
| Patients | 31 | 4 | 3 |
| Slice Shapes | 256x128 (11,400), 256x144 (60) | 256x128 | 256x128 |
| Intensity Range (Sampled) | 0 – 1,150 | 0 – 565  | 0 – 617 |
| Median Per-Slice | 387.5 | 373 | 424 |
| Near-Empty Slices | 0% | 0% | 0% |

</div>

An analysis of this preliminary data audit can be found in **Section 4: Data**.
### 1.4 Risk, Budget and Fallback
The key technical risks associated with the project are:
- WGAN has a risk of unstable training results if a momentum-based optimiser (Adam) or high learning rates are used, placing some restrictions on model architecture and tuning parameters.
- Rangpur script delays due to high student traffic approaching the due date.
- Chosen parameters exceed GPU memory or provided training time, which may not become apparent until training failure occurs.
- Test set only has 3 patients which limits strong statements surrounding how well generalised the data is.
The next planned experiment is memory/time estimations followed by baseline model (DCGAN) implementation. In the case where proposed risks begin to impact the deliverables of the project, the DCGAN baseline can instead become the main project with a basic autoencoder becoming the new baseline. This is a different normal difficulty option under the same topic.


## 2. How It Works
Exact information about hyperparameters/how to run can be found in **Section 3: Usage**. This section instead summarises the general architecture of the system and the differences between the baseline and real model. For tabular summary of architecture, skip to **Section 2.3**.
### 2.1 DCGAN (Baseline)
For the baseline implementation a DCGAN was created using only the standard hyperparameters from the DCGAN paper. The generator goes through 4 layers of convolution, starting from latent space vector *z* and outputting a 128x64 image, matching the training images (see **Section 4.2** about pre-process downsampling). Batch normalisation is done between convolutions with ReLU activation functions being used until after the final convolution, which is a Tanh activation. The critic follows the same architecture in reverse, taking in a 128x64 image, and outputting a real or fake label. The key difference between the two is in the activation functions, as the critic uses LeakyReLU with a Sigmoid after the final convolution. BCELoss was utilised in the baseline model.
### 2.2 WGAN-GP
WGAN-GP maintains most of the same architecture as the DCGAN in its generator and critic, with the main differences being in the critic, which no longer has a final Sigmoid activation function and uses group normalisation instead of batch. We no longer use the sigmoid because we want the raw critic score rather than the probability. As for the normalisation, WGAN-GP uses gradient penalty, which is based on one image at a time, but batch normalisation normalises each image by the mean and variance of the whole batch, which breaks the gradient penalty.

Other major changes between the DCGAN and the WGAN-GP surround the loss function. While DCGAN uses BCELoss, WGAN-GP *must* use Wasserstian loss (which has separate equations for the loss of the critic vs loss of the generator). In addition, the critic loss also includes the gradient penalty component (the GP portion of WGAN-GP). Lipschitz constraint is obeyed through the usage of the gradient penalty, making this model a WGAN.

### 2.3 Architecture
**Generator** (identical in DCGAN/WGAN-GP)

<div align="center">

| Layer | Operation | Output shape | Parameters |
|---|---|---|---|
| Input | Latent vector z ~ N(0, I) | 100 × 1 × 1 | – |
| 1 | ConvTranspose (8×4, stride 1) → BatchNorm → ReLU | 512 × 8 × 4 | 1,639,424 |
| 2 | ConvTranspose (4×4, stride 2) → BatchNorm → ReLU | 256 × 16 × 8 | 2,097,664 |
| 3 | ConvTranspose (4×4, stride 2) → BatchNorm → ReLU | 128 × 32 × 16 | 524,544 |
| 4 | ConvTranspose (4×4, stride 2) → BatchNorm → ReLU | 64 × 64 × 32 | 131,200 |
| 5 | ConvTranspose (4×4, stride 2) → Tanh | **1 × 128 × 64** | 1,024 |
| | | **Total** | **4,393,856** |

</div>

**Critic** (different in DCGAN/WGAN-GP)

<div align="center">

| Layer | Operation | Output shape | Parameters |
|---|---|---|---|
| Input | Real or generated slice | 1 × 128 × 64 | – |
| 1 | Conv (4×4, stride 2) → LeakyReLU(0.2) | 64 × 64 × 32 | 1,024 |
| 2 | Conv (4×4, stride 2) → Norm* → LeakyReLU(0.2) | 128 × 32 × 16 | 131,328 |
| 3 | Conv (4×4, stride 2) → Norm* → LeakyReLU(0.2) | 256 × 16 × 8 | 524,800 |
| 4 | Conv (4×4, stride 2) → Norm* → LeakyReLU(0.2) | 512 × 8 × 4 | 2,098,176 |
| 5 | Conv (8×4, stride 1) | **1 × 1 × 1** | 16,384 |
| | | **Total** | **2,771,712** |

</div>

\* **Norm** is `BatchNorm2d` in the DCGAN discriminator and `GroupNorm(1, C)` (layer
normalisation) in the WGAN-GP critic. The DCGAN discriminator additionally ends with a
Sigmoid to output a probability; the WGAN-GP critic outputs an unbounded score.

## 3. Usage
### 3.1 Dependencies
All code in this project was run using Python 3.13. Exact dependencies for the different files are provided below:
`data_audit.py`
- os
- re
- nibabel
- numpy
- Counter from collections

`modules.py`
- torch (including nn, optim and utils)
- torchvision.utils

`dataset.py`
- os
- glob
- Dataset from torch.utils
- nibabel
- numpy

`train.py`
- Insert here (must restructure files)

`predict.py`
- json
- torch
- matplotlib
- Path from pathlib
- torchvision.utils
- torchmetrics (functional and image)

### 3.2 Hyperparameters
| Hyperparameter | DCGAN (baseline) | WGAN-GP (standard) | WGAN-GP (tuned) |
|---|---|---|---|
| **Data & architecture** | | | |
| Image size | 1 × 128 × 64 | 1 × 128 × 64 | 1 × 128 × 64 |
| Latent dimension (nz) | 100 | 100 | 100 |
| Generator / critic base width (ngf / ndf) | 64 / 64 | 64 / 64 | 64 / 64 |
| Generator normalisation | BatchNorm | BatchNorm | BatchNorm |
| Discriminator / critic normalisation | BatchNorm | **GroupNorm(1, C) (layer norm)** | GroupNorm(1, C) (layer norm) |
| Discriminator / critic output | Sigmoid (probability) | **Linear (unbounded score)** | Linear (unbounded score) |
| Weight initialisation | N(0, 0.02) | N(0, 0.02) | N(0, 0.02) |
| **Objective** | | | |
| Loss | BCE (non-saturating G loss) | **Wasserstein + gradient penalty** | Wasserstein + gradient penalty |
| Gradient penalty weight (λ) | – | **10** | 10 |
| Critic updates per G update (n_critic) | 1 | **5** | 5 |
| **Optimisation** | | | |
| Optimiser | Adam | Adam | Adam |
| Learning rate (G and D) | 2 × 10⁻⁴ | **1 × 10⁻⁴** | 1 × 10⁻⁴ |
| Adam (β₁, β₂) | (0.5, 0.999) | **(0.0, 0.9)** | (0.0, 0.9) |
| LR schedule | Constant | Constant | **Constant to epoch 125, then linear decay to 0 at epoch 250 (G ___ D)** |
| Generator EMA | No | No | **Yes, decay = ___** |
| Batch size | 128 | 128 | 128 |
| **Training length** | | | |
| Epochs | 50 | **250** | 250 |
| Generator updates per epoch | 90 | **18** | 18 |
| Total generator updates | 4,500 | 4,500 | 4,500 |
| Total critic / discriminator updates | 4,500 | **22,500** | 22,500 |
| Selected checkpoint (by validation ___) | epoch ___ | epoch ___ | epoch ___ |

### 3.3 Reproducing Results
**DCGAN**

**WGAN-GP**

**WGAN-GP Tuned**

Commands for training (each model), evaluation and prediction; seeds; data path.
### 3.4 Example Inputs & Outputs
What predict.py takes and produces.

## 4. Data
### 4.1 Dataset & Data Audit
The provided dataset is already split into separate training, validation and testing sets. In this case, an audit was done on the provided sets to ensure they were appropriately split with usable data (i.e. no patient leakage between sets and no near-empty/dark images) as seen in **Section 1.3**. The audit found no patient leakage between sets as well as 0 near-empty slices in the provided files, suggesting the test/train/split was valid. In addition to validating the dataset's usability for the task, the audit results were used to help determine what kind of pre-processing should be completed on the dataset.
### 4.2 Pre-processing
- **Centre Crop:** In order to avoid dropping any data from the training set, a centre crop was performed to resize the 60 outlying 256x144 images that were found during the data audit.
- **Scaling to [-1, 1]:** Due to the end of the generator architecture using a tanh activation function, intensity values must be min-max scaled to a range of [-1, 1] to be able to pass through it.
- **99.5th-Percentile Clipping:** The data audit revealed the training data featured intensity values up to 1,150, however, the median highest-intensity value seen in training images was only 387.5. Since the data is min-max scaled from [-1, 1], any training scans that feature outlier high-intensity values will define the top of the scaling range for that slice. This would mean that the other lower-intensity pixels would be scaled down and squashed into a narrower, darker band of intensity values. To prevent this, intensity values for images are clipped to the 99.5th percentile.
- **Downsampling to 128x64:** The network architecture in the DCGAN/WGAN papers was designed around a 64x64, so using the full resolution image would require an additional layer in both the generator and critic networks. Additionaly, downsampling reduces the total pixel count of the images, lowering expected memory usage and training time. While the 2:1 aspect ratio is preserved, it is expected that this pre-processing decision will prevent the generator from learning some of the finer details of the data.

### 4.3 Splits
**Train (91%) / Validate (5%) / Test (4%) Split**
- **Train:** For fitting the generator and critic (learning the weights).
- **Validate:** KID verification is done against validation slices, with the intent of helping to tune the WGAN-GP.
- **Test:** Used with the final model, checking SSIM, KID/FID and memorisation audit acceptance criteria are met.

## 5. Experiments
### 5.1 Training Setup
Hyperparameter table for all three models; hardware; epochs and number of
generator updates.
### 5.2 Training Dynamics
DCGAN loss curves (instability spikes) vs WGAN-GP W-estimate and GP curves.
### 5.3 Generated Samples
Uncurated grids per model; progression figure (epochs 1 / 10 / 25 / final).
### 5.4 Tuned WGAN-GP
What was changed, why, and what effect it had (on validation data).

## 6. Results
### 6.1 Quantitative Benchmark
Table: DCGAN | WGAN-GP (paper defaults) | WGAN-GP (tuned): SSIM vs test,
FID/KID, diversity metric; pass/fail against SSIM > 0.6.
### 6.2 Memorisation Audit
Nearest-neighbour distances (generated→train vs test→train), example
nearest-neighbour pairs.
### 6.3 Latent Space Interpolation
Interpolation strip between two latent vectors; smoothness and anatomical
plausibility.
### 6.4 Resource Profiling
Table: parameters, peak VRAM, time per epoch, time per generator update,
generation latency, GPU.

## 7. Failure Analysis
3–5 representative failure cases, each with image, diagnosis and likely cause
(e.g. checkerboard artefacts, incoherent anatomy, partial mode collapse).

## 8. Recommendation
Evidence-based verdict to the project manager: is WGAN-GP worth its extra cost,
is it safe to share (memorisation), and what would be needed before deployment.
### 8.1 Limitations & Future Work
Small test set (3 patients), FID on greyscale MRI, 2D only, etc.

## 9. AI Use Disclosure
What tools were used, for what, and how outputs were verified.

## 10. References
Arjovsky et al. 2017; Gulrajani et al. 2017; Radford et al. 2016;
PyTorch DCGAN tutorial; HipMRI dataset source.