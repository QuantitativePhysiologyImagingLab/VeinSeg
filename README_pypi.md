# VeinSeg

Physics-informed deep learning for cerebral vein segmentation from QSM.

[Hugging Face](https://huggingface.co/YousifKhoury/VeinSeg) | [GitHub](https://github.com/YousifKhoury/VeinSeg)

---

## Overview

VeinSeg segments cerebral veins from Quantitative Susceptibility Mapping (QSM). It supports multi-site, multi-field-strength data (3T and 7T) across five QSM reconstruction methods (TGV, MEDI, L1, STAR, iLSQR) using a physics-informed training objective.

---

## Installation

Install PyTorch first ([pytorch.org](https://pytorch.org)), then:

```bash
pip install veinseg
```

Download the model weights (~600 MB, once only):

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
| `-i` | QSM susceptibility map (`.nii` / `.nii.gz`, ppm) |
| `-r` | QSM reconstruction method: `tgv` \| `medi` \| `l1` \| `star` \| `ilsqr` |
| `-f` | MRI field strength: `7t` \| `3t` |
| `-o` | Output binary vein mask |
| `-p` | Output vein probability map |

### Optional arguments

| Flag | Default | Description |
|---|---|---|
| `--local-field PATH` | — | Measured background-removed local field (skips dipole computation) |
| `--local-field-units` | `auto` | `hz` \| `ppm` \| `auto` |
| `--b0 X Y Z` | `0 0 1` | B0 direction in world/scanner axes |
| `--threshold` | `0.5` | Probability threshold for binary mask |
| `--step-size` | `0.5` | Sliding window overlap as fraction of patch |
| `--no-tta` | off | Disable test-time augmentation |
| `--device` | `auto` | `auto` \| `cpu` \| `cuda` |
| `--out-field PATH` | — | Save local field channel used (ppm) |
| `--out-frangi PATH` | — | Save Frangi vesselness channel |

### Examples

```bash
# Dipole field computed automatically from QSM
veinseg -i qsm.nii.gz -r tgv -f 7t -o mask.nii.gz -p prob.nii.gz

# With measured local field (Romeo output in Hz — auto-detected)
veinseg -i qsm.nii.gz -r medi -f 7t -o mask.nii.gz -p prob.nii.gz \
        --local-field bgrm_field.nii.gz

# Inspect intermediate channels
veinseg -i qsm.nii.gz -r tgv -f 7t -o mask.nii.gz -p prob.nii.gz \
        --out-field dipole_field.nii.gz --out-frangi frangi.nii.gz
```

---

## License

Apache 2.0
