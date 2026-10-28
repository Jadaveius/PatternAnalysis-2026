# Synthetic Prostate MRI Generation with WGAN-GP (HipMRI 2D)

One-paragraph summary: what was built, the problem it solves, headline result
(e.g. "WGAN-GP reached SSIM X vs DCGAN Y with no evidence of memorisation"). - ADDED LAST
BASED ON RESULTS OF REPORT (like exec summary)

![Uncurated WGAN-GP samples](images/wgan_gp_final.png)

## Contents
(optional table of contents linking to the sections below)

## 1. Feasibility Review
Topic 2.4 - Image Generation: Synthetic Medical Cohorts and Privacy
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
| **Measure** | **Train** | **Validate** | **Test** |
| --- | :---: | :---: | :---: |
| Slices | 11,460 | 660 | 540 |
| Patients | 31 | 4 | 3 |
| Slice Shapes | 256x128 (11,400), 256x144 (60) | 256x128 | 256x128 |
| Intensity Range (Sampled) | 0 – 1,150 | 0 – 565  | 0 – 617 |
| Median Per-Slice | 387.5 | 373 | 424 |
| Near-Empty Slices | 0% | 0% | 0% |
An analysis of this preliminary data audit can be found in Section 3: Data.
### 1.4 Risk, Budget and Fallback
The key technical risks associated with the project are:
- WGAN has a risk of unstable training results if a momentum-based optimiser (Adam) or high learning rates are used, placing some restrictions on model architecture and tuning parameters.
- Rangpur script delays due to high student traffic approaching the due date.
- Chosen parameters exceed GPU memory or provided training time, which may not become apparent until training failure occurs.
- Test set only has 3 patients which limits strong statements surrounding how well generalised the data is.
The next planned experiment is memory/time estimations followed by baseline model (DCGAN) implementation. In the case where proposed risks begin to impact the deliverables of the project, the DCGAN baseline can instead become the main project with a basic autoencoder becoming the new baseline. This is a different normal difficulty option under the same topic.


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