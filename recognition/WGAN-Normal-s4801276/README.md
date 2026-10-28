# Synthetic Prostate MRI Generation with WGAN-GP (HipMRI 2D)

One-paragraph summary: what was built, the problem it solves, headline result
(e.g. "WGAN-GP reached SSIM X vs DCGAN Y with no evidence of memorisation").

![Uncurated WGAN-GP samples](images/wgan_gp_final.png)

## Contents
(optional table of contents linking to the sections below)

## 1. Problem & Motivation
### 1.1 The Engineering Dilemma
Fidelity vs memorisation vs mode collapse: why synthetic medical images must be
realistic AND novel.
### 1.2 Feasibility Review
Your one-page review from the check-off, clearly labelled.

## 2. How It Works
### 2.1 DCGAN (Baseline)
Architecture, BCE loss, why it's the baseline.
### 2.2 WGAN-GP
Wasserstein distance, Lipschitz critic, gradient penalty (with the formula),
n_critic, why BatchNorm is removed from the critic.
### 2.3 Architecture
Figure or table of generator/critic layers and tensor shapes (1×128×64);
what changed from DCGAN and why.

## 3. Data
### 3.1 Dataset & Data Audit
HipMRI 2D slices; audit table (slice counts, patients, shapes, intensities);
patient-level leakage check.
### 3.2 Pre-processing
Centre crop, 99.5th-percentile clipping, scaling to [-1, 1], resize to 128×64,
with justification for each.
### 3.3 Splits
Train / validate / test usage and why: train for training, validate for model
selection and tuning, test only for final evaluation.

## 4. Experiments
### 4.1 Training Setup
Hyperparameter table for all three models; hardware; epochs and number of
generator updates.
### 4.2 Training Dynamics
DCGAN loss curves (instability spikes) vs WGAN-GP W-estimate and GP curves.
### 4.3 Generated Samples
Uncurated grids per model; progression figure (epochs 1 / 10 / 25 / final).
### 4.4 Tuned WGAN-GP
What was changed, why, and what effect it had (on validation data).

## 5. Results
### 5.1 Quantitative Benchmark
Table: DCGAN | WGAN-GP (paper defaults) | WGAN-GP (tuned): SSIM vs test,
FID/KID, diversity metric; pass/fail against SSIM > 0.6.
### 5.2 Memorisation Audit
Nearest-neighbour distances (generated→train vs test→train), example
nearest-neighbour pairs.
### 5.3 Latent Space Interpolation
Interpolation strip between two latent vectors; smoothness and anatomical
plausibility.
### 5.4 Resource Profiling
Table: parameters, peak VRAM, time per epoch, time per generator update,
generation latency, GPU.

## 6. Failure Analysis
3–5 representative failure cases, each with image, diagnosis and likely cause
(e.g. checkerboard artefacts, incoherent anatomy, partial mode collapse).

## 7. Recommendation
Evidence-based verdict to the project manager: is WGAN-GP worth its extra cost,
is it safe to share (memorisation), and what would be needed before deployment.
### 7.1 Limitations & Future Work
Small test set (3 patients), FID on greyscale MRI, 2D only, etc.

## 8. Usage
### 8.1 Dependencies
Exact versions (Python, torch, torchvision, nibabel, numpy, matplotlib…).
### 8.2 Reproducing Results
Commands for training (each model), evaluation and prediction; seeds; data path.
### 8.3 Example Inputs & Outputs
What predict.py takes and produces.

## 9. AI Use Disclosure
What tools were used, for what, and how outputs were verified.

## 10. References
Arjovsky et al. 2017; Gulrajani et al. 2017; Radford et al. 2016;
PyTorch DCGAN tutorial; HipMRI dataset source.