# VeinSeg

Physics-informed deep learning for cerebral vein segmentation from QSM or R2*.

[Hugging Face](https://huggingface.co/YousifKhoury/VeinSeg) | [GitHub](https://github.com/YousifKhoury/VeinSeg)

---

## Overview

VeinSeg segments cerebral veins from Quantitative Susceptibility Mapping (QSM) or R2* maps, using only the image as input. It supports multi-site, multi-field-strength data (3T and 7T) across five QSM reconstruction methods (TGV, MEDI, L1, STAR, iLSQR) and R2* maps using a physics-informed training objective.

---

## Installation

Install PyTorch first ([pytorch.org](https://pytorch.org)), then:

```bash
pip install veinseg-qsm
```

Download the model weights (~290 MB, once only):

```bash
veinseg-install /path/to/models/dir
```

On shared HPC clusters, set:

```bash
export VEINSEG_CHECKPOINT=/shared/models/veinseg/checkpoint.pth
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

## License

Apache 2.0
