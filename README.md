<p align="center">
  <img src="assets/veins_3d.png" width="480" alt="3D cerebral vein segmentation">
</p>

<h1 align="center">VeinSeg</h1>
<p align="center">
  Physics-informed deep learning for cerebral vein segmentation from QSM
</p>

<p align="center">
  <a href="https://huggingface.co/YousifKhoury/VeinSeg"><img src="https://img.shields.io/badge/🤗%20Hugging%20Face-Model-yellow" alt="Hugging Face"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache%202.0-blue" alt="License"></a>
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/python-3.10+-blue" alt="Python"></a>
</p>

---

## Overview

VeinSeg is a 3D deep learning tool for automatic segmentation of cerebral veins from Quantitative Susceptibility Mapping (QSM) or R2* maps. It is trained on multi-site, multi-field-strength data (3T and 7T) across five QSM reconstruction methods (TGV, MEDI, L1, STAR, iLSQR) and R2* maps, and uses a physics-informed training objective combining:

- **Supervised** Dice + Cross-Entropy loss
- **Self-supervised** physics-informed dipole field consistency loss
- **Self-supervised** Frangi vesselness shape loss

## Model Architecture

<p align="center">
  <img src="assets/architecture.png" width="800" alt="VeinSeg architecture">
</p>

The model is a 3D attention U-Net with a single-channel encoder and three key components:

**Prior Attention Gating** — only the primary image (QSM or R2*) enters the encoder. It is modulated as `img · (1 + gate)`. During training the gate is computed from the local field and Frangi vesselness priors, so they shape what the encoder learns; at inference no priors are needed and the gate reduces to its learned constant value.

**Feature-wise Linear Modulation (FiLM)** — domain embeddings for reconstruction method (TGV / MEDI / L1 / STAR / iLSQR / R2*) and field strength (7T / 3T) condition every encoder stage via learned scale and shift, enabling the same model to generalise across acquisition protocols without retraining.

**R2\* adapter** — a residual conv block after the first encoder stage that is active only for R2* inputs, giving that contrast dedicated capacity without changing the QSM pathway.

### Inputs

A single image: a QSM susceptibility map (ppm) or an R2* map (1/s). No local field or Frangi map is required.

### Output
- Binary vein mask (argmax)
- Vein probability map (softmax)

---

## Installation

VeinSeg requires PyTorch. Install your preferred version first (see [pytorch.org](https://pytorch.org) for CUDA-specific instructions), then:

```bash
pip install veinseg-qsm
```

Download the model weights (run once only):

```bash
veinseg-install /path/to/models/dir
```

The checkpoint is downloaded from [Hugging Face](https://huggingface.co/YousifKhoury/VeinSeg) and the path is saved automatically. You can also set the model path manually using:

```bash
export VEINSEG_CHECKPOINT=/models/veinseg/checkpoint.pth
```

---

## Usage

```bash
veinseg -i qsm.nii.gz -r tgv -f 7t -o mask.nii.gz -p prob.nii.gz
```

### Required arguments

| Flag | Description |
|---|---|
| `-i` | QSM susceptibility map (`.nii` / `.nii.gz`, ppm), or R2* map (1/s) with `-r r2star` — the only input needed |
| `-r` | Reconstruction method: `tgv` \| `medi` \| `l1` \| `star` \| `ilsqr` \| `r2star` |
| `-f` | MRI field strength: `7t` \| `3t` |
| `-o` | Output binary vein mask |
| `-p` | Output vein probability map |

### Optional arguments

| Flag | Default | Description |
|---|---|---|
| `--threshold` | `0.5` | Probability threshold for binary mask |
| `--step-size` | `0.5` | Sliding window overlap as fraction of patch |
| `--no-tta` | off | Disable test-time augmentation (mirroring) |
| `--device` | `auto` | `auto` \| `cpu` \| `cuda` |

### Examples

```bash
# QSM
veinseg -i qsm.nii.gz -r medi -f 3t -o mask.nii.gz -p prob.nii.gz

# R2*
veinseg -i r2star.nii.gz -r r2star -f 3t -o mask.nii.gz -p prob.nii.gz

# CPU inference, no test-time augmentation (faster)
veinseg -i qsm.nii.gz -r tgv -f 7t -o mask.nii.gz -p prob.nii.gz \
        --device cpu --no-tta
```

---

## Supported methods

| Flag | Method |
|---|---|
| `tgv` | Total Generalised Variation |
| `medi` | Morphology Enabled Dipole Inversion |
| `l1` | L1-regularised |
| `star` | STAR-QSM |
| `ilsqr` | Iterative LSQR |
| `r2star` | R2* map |

---

## Model weights

Weights are hosted on Hugging Face: [YousifKhoury/VeinSeg](https://huggingface.co/YousifKhoury/VeinSeg)

Downloaded automatically on first use by `veinseg-install`.
