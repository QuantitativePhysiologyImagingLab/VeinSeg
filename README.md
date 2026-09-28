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

VeinSeg is a 3D deep learning tool for automatic segmentation of cerebral veins from Quantitative Susceptibility Mapping (QSM). It is trained on multi-site, multi-field-strength data (3T and 7T) across five QSM reconstruction methods (TGV, MEDI, L1, STAR, iLSQR) and R2* maps and uses a physics-informed training objective combining:

- **Supervised** Dice + Cross-Entropy loss
- **Self-supervised** physics-informed dipole field consistency loss
- **Self-supervised** Frangi vesselness shape loss

## Model Architecture

<p align="center">
  <img src="assets/architecture.png" width="800" alt="VeinSeg architecture">
</p>

The model is a 3D attention U-Net with a single-channel encoder and three key components:

**Prior Attention Gating** — only the primary image (QSM or R2*) enters the encoder. It is modulated as `img · (1 + gate)`, where the gate is computed from the local field and Frangi prior channels, allowing the network to focus on regions with high vesselness response.

**Feature-wise Linear Modulation (FiLM)** — domain embeddings for reconstruction method (TGV / MEDI / L1 / STAR / iLSQR / R2*) and field strength (7T / 3T) condition every encoder stage via learned scale and shift, enabling the same model to generalise across acquisition protocols without retraining.

**R2\* adapter** — a residual conv block after the first encoder stage that is active only for R2* inputs, giving that contrast dedicated capacity without changing the QSM pathway.

### Inputs

| Channel | Content | Units |
|---|---|---|
| 0 | QSM susceptibility map (or R2* map) — encoder input | ppm (1/s) |
| 1 | Local field (measured or dipole-computed from QSM; zero-filled for R2* if not given) — gate prior | ppm |
| 2 | Frangi vesselness of channel 0, zeroed within 5 mm of the brain edge — gate prior | [0, 1] |

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
| `-i` | QSM susceptibility map (`.nii` / `.nii.gz`, ppm), or R2* map with `-r r2star` |
| `-r` | Reconstruction method: `tgv` \| `medi` \| `l1` \| `star` \| `ilsqr` \| `r2star` |
| `-f` | MRI field strength: `7t` \| `3t` |
| `-o` | Output binary vein mask |
| `-p` | Output vein probability map |

### Optional arguments

| Flag | Default | Description |
|---|---|---|
| `--local-field PATH` | — | Measured background-removed local field instead of dipole-computed. Accepts Hz or ppm (auto-detected). |
| `--local-field-units` | `auto` | Force units: `hz` \| `ppm` \| `auto` |
| `--b0 X Y Z` | `0 0 1` | B0 direction in world/scanner axes |
| `--frangi-erode-mm` | `5` | Zero the Frangi prior within this distance (mm) of the brain edge |
| `--threshold` | `0.5` | Probability threshold for binary mask |
| `--step-size` | `0.5` | Sliding window overlap as fraction of patch |
| `--no-tta` | off | Disable test-time augmentation (mirroring) |
| `--device` | `auto` | `auto` \| `cpu` \| `cuda` |
| `--out-field PATH` | — | Save local field channel used (ppm) |
| `--out-frangi PATH` | — | Save Frangi vesselness channel |

### Examples

```bash
# Basic — dipole field computed automatically from QSM
veinseg -i qsm.nii.gz -r tgv -f 7t -o mask.nii.gz -p prob.nii.gz

# With measured local field (Romeo output in Hz — auto-detected)
veinseg -i qsm.nii.gz -r medi -f 7t -o mask.nii.gz -p prob.nii.gz \
        --local-field bgrm_field.nii.gz

# Inspect intermediate channels going into the model
veinseg -i qsm.nii.gz -r tgv -f 7t -o mask.nii.gz -p prob.nii.gz \
        --out-field dipole_field.nii.gz --out-frangi frangi.nii.gz

# CPU inference
veinseg -i qsm.nii.gz -r tgv -f 7t -o mask.nii.gz -p prob.nii.gz \
        --device cpu
```

---

## Supported QSM methods

| Flag | Method |
|---|---|
| `tgv` | Total Generalised Variation |
| `medi` | Morphology Enabled Dipole Inversion |
| `l1` | L1-regularised |
| `star` | STAR-QSM |
| `ilsqr` | Iterative LSQR |
| `r2star` | R2* map (not QSM; pass `--local-field` if available) |

---

## Model weights

Weights are hosted on Hugging Face: [YousifKhoury/VeinSeg](https://huggingface.co/YousifKhoury/VeinSeg)

Downloaded automatically on first use by `veinseg-install`.
