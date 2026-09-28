#!/usr/bin/env python3
"""
VeinSeg: cerebral vein segmentation from QSM or R2*.

Pipeline:
  1. Load the single input image (QSM or R2*)
  2. Run nnUNetPredictor (identical preprocessing + sliding window to training)
  3. Save binary mask and probability map

The network's prior gate (trained from local field + Frangi) falls back to its
learned constant value when no priors are supplied, so only the image is needed.
"""
import os
import sys
import tempfile

# Silence nnUNet path warnings — we don't use its file system layout
os.environ.setdefault("nnUNet_raw",          "/tmp/veinseg_nnunet")
os.environ.setdefault("nnUNet_preprocessed", "/tmp/veinseg_nnunet")
os.environ.setdefault("nnUNet_results",      "/tmp/veinseg_nnunet")

import numpy as np
import nibabel as nib
import torch

from veinseg._checkpoint import get_checkpoint
from veinseg.model_arch   import PriorGatedSingleChannelUNetInfer

METHOD_TO_IDX = {"tgv": 0, "medi": 1, "l1": 2, "star": 3, "ilsqr": 4, "r2star": 5}


def _print_help_and_exit():
    print("""veinseg — cerebral vein segmentation from QSM or R2*

usage:
  veinseg -i IMAGE -r METHOD -f FIELD -o BINARY -p PROB [options]

required:
  -i IMAGE      QSM susceptibility map (.nii / .nii.gz, ppm), or R2* map (1/s)
                when -r r2star
  -r METHOD     Reconstruction method: tgv | medi | l1 | star | ilsqr | r2star
  -f FIELD      MRI field strength: 3t | 7t
  -o BINARY     Output binary vein mask (.nii.gz)
  -p PROB       Output vein probability map (.nii.gz)

optional:
  --threshold T              Binarization threshold, 0-1 (default: 0.5)
  --step-size N              Sliding window step as fraction of patch (default: 0.5)
  --no-tta                   Disable test-time augmentation (mirroring)
  --device MODE              auto | cpu | cuda  (default: auto)

examples:
  veinseg -i qsm.nii.gz    -r tgv    -f 7t -o mask.nii.gz -p prob.nii.gz
  veinseg -i r2star.nii.gz -r r2star -f 3t -o mask.nii.gz -p prob.nii.gz

note:
  Download the model checkpoint (~290 MB) once with: veinseg-install <dir>
  or point to one with: export VEINSEG_CHECKPOINT=/path/to/checkpoint.pth
""")
    sys.exit(0)


def main():
    if len(sys.argv) == 1 or any(a in sys.argv for a in ("-h", "--help")):
        _print_help_and_exit()

    import argparse
    ap = argparse.ArgumentParser(prog="veinseg", add_help=False)
    ap.add_argument("-i",  required=True, metavar="IMAGE")
    ap.add_argument("-r",  required=True, choices=list(METHOD_TO_IDX.keys()))
    ap.add_argument("-f",  required=True, choices=["3t", "7t"])
    ap.add_argument("-o",  required=True, metavar="BINARY")
    ap.add_argument("-p",  required=True, metavar="PROB")
    ap.add_argument("--step-size",  type=float, default=0.5)
    ap.add_argument("--no-tta",     action="store_true")
    ap.add_argument("--device",     default="auto",
                    choices=["auto", "cpu", "cuda"])
    ap.add_argument("--threshold",  type=float, default=0.5)
    args = ap.parse_args()

    device = _pick_device(args.device)
    print(f"[veinseg] device: {device}")

    # ---- load image ----
    print(f"[veinseg] loading {args.i}")
    img_nii = nib.load(args.i)
    img     = np.nan_to_num(img_nii.get_fdata(dtype=np.float32),
                            nan=0., posinf=0., neginf=0.)
    print(f"[veinseg] shape: {img.shape}  "
          f"spacing: {' x '.join(f'{z:.3f}' for z in img_nii.header.get_zooms()[:3])} mm")

    # ---- load checkpoint ----
    checkpoint_path = get_checkpoint()
    print(f"[veinseg] loading checkpoint from {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    from nnunetv2.utilities.plans_handling.plans_handler import PlansManager

    # Only the primary channel is used: its normalization is plans channel 0
    dataset_json = dict(ckpt["init_args"]["dataset_json"])
    dataset_json["channel_names"] = {"0": dataset_json["channel_names"]["0"]}

    plans_manager         = PlansManager(ckpt["init_args"]["plans"])
    configuration_manager = plans_manager.get_configuration(
                                ckpt["init_args"]["configuration"])

    model = PriorGatedSingleChannelUNetInfer(
        out_channels=2,
        patch_size=tuple(configuration_manager.patch_size),
        deep_supervision=True, num_domains=6, domain_embed_dim=32,
    )
    model.load_state_dict(ckpt["network_weights"], strict=True)
    model.default_domain_idx = METHOD_TO_IDX[args.r.lower()]
    model.default_field_idx  = 0 if args.f.lower() == "7t" else 1
    model.to(device).eval()
    print(f"[veinseg] method={args.r} (domain={model.default_domain_idx})  "
          f"field={args.f} (field_idx={model.default_field_idx})")

    # ---- nnUNetPredictor (identical to training-time inference) ----
    predictor = nnUNetPredictor(
        tile_step_size=args.step_size,
        use_gaussian=True,
        use_mirroring=not args.no_tta,
        perform_everything_on_device=True,
        device=device,
        verbose=True,
    )
    predictor.manual_initialization(
        network=model,
        plans_manager=plans_manager,
        configuration_manager=configuration_manager,
        parameters=[model.state_dict()],
        dataset_json=dataset_json,
        trainer_name=ckpt["trainer_name"],
        inference_allowed_mirroring_axes=ckpt["inference_allowed_mirroring_axes"],
    )

    # ---- write temp files -> predict_from_files -> read back ----
    print("[veinseg] running inference ...")
    with tempfile.TemporaryDirectory() as tmpdir:
        ch0_path  = os.path.join(tmpdir, "case_0000.nii.gz")
        out_trunc = os.path.join(tmpdir, "case")

        nib.save(nib.Nifti1Image(img, img_nii.affine, img_nii.header), ch0_path)

        predictor.predict_from_files(
            list_of_lists_or_source_folder=[[ch0_path]],
            output_folder_or_list_of_truncated_output_files=[out_trunc],
            save_probabilities=True,
            overwrite=True,
            num_processes_preprocessing=1,
            num_processes_segmentation_export=1,
        )

        seg_nii   = nib.load(out_trunc + ".nii.gz")
        seg_shape = tuple(seg_nii.header.get_data_shape())

        probs_npz = np.load(out_trunc + ".npz")
        prob_key  = "probabilities" if "probabilities" in probs_npz else "arr_0"
        prob = probs_npz[prob_key][1].T.astype(np.float32)  # (Z,Y,X) -> (X,Y,Z)

        if prob.shape != seg_shape:
            from scipy.ndimage import zoom as ndimage_zoom
            zf   = tuple(s / p for s, p in zip(seg_shape, prob.shape))
            prob = ndimage_zoom(prob, zf, order=1).clip(0., 1.).astype(np.float32)

    binary = (prob >= args.threshold).astype(np.float32)
    _save_nii(binary, seg_nii, args.o)
    _save_nii(prob,   seg_nii, args.p)
    print(f"[veinseg] done.\n  binary:      {args.o}\n  probability: {args.p}")


def _pick_device(choice):
    if choice == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(choice)


def _save_nii(arr, ref_nii, path):
    out = nib.Nifti1Image(arr.astype(np.float32), ref_nii.affine, ref_nii.header)
    out.set_data_dtype(np.float32)
    nib.save(out, path)
    print(f"  saved {path}")


if __name__ == "__main__":
    main()
