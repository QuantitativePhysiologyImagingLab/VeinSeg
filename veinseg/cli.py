#!/usr/bin/env python3
"""
VeinSeg: physics-informed cerebral vein segmentation from QSM.

Pipeline:
  1. Compute dipole local field from QSM (or use provided measured field)
  2. Compute Frangi vesselness from QSM
  3. Run nnUNetPredictor (identical preprocessing + sliding window to training)
  4. Save binary mask and probability map
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
from veinseg.model_arch   import PriorGatedUNetWithAttentionInfer
from veinseg.frangi       import frangi_3d
from veinseg.dipole_conv  import (dipole_field_from_chi_xyz,
                                   voxel_size_from_affine,
                                   b0_dir_from_image_affine)

FRANGI_SIGMAS = (0.1, 0.5, 0.8, 1.0)
METHOD_TO_IDX = {"tgv": 0, "medi": 1, "l1": 2, "star": 3, "ilsqr": 4}


def _print_help_and_exit():
    print("""veinseg — physics-informed cerebral vein segmentation from QSM

usage:
  veinseg -i QSM -r METHOD -f FIELD -o BINARY -p PROB [options]

required:
  -i QSM        QSM susceptibility map (.nii / .nii.gz, ppm)
  -r METHOD     QSM reconstruction method: tgv | medi | l1 | star | ilsqr
  -f FIELD      MRI field strength: 3t | 7t
  -o BINARY     Output binary vein mask (.nii.gz)
  -p PROB       Output vein probability map (.nii.gz)

optional:
  --local-field PATH          Measured background-removed local field
                              (skips dipole computation — for best accuracy)
  --local-field-units auto|hz|ppm
                              Units of --local-field (default: auto-detect)
  --b0 X Y Z                 B0 direction in world/scanner axes (default: 0 0 1)
  --threshold T              Binarization threshold, 0-1 (default: 0.5)
  --step-size N              Sliding window step as fraction of patch (default: 0.5)
  --no-tta                   Disable test-time augmentation (mirroring)
  --device MODE              auto | cpu | cuda  (default: auto)

channel outputs (save intermediate model inputs for inspection):
  --out-field PATH           Save local field channel used (ppm)
  --out-frangi PATH          Save Frangi vesselness channel ([0,1])

examples:
  veinseg -i qsm.nii.gz -r tgv  -f 7t -o mask.nii.gz -p prob.nii.gz
  veinseg -i qsm.nii.gz -r medi -f 7t -o mask.nii.gz -p prob.nii.gz \\
          --local-field bgrm_field.nii.gz
  veinseg -i qsm.nii.gz -r tgv  -f 7t -o mask.nii.gz -p prob.nii.gz \\
          --out-field field.nii.gz --out-frangi frangi.nii.gz

note:
  The model checkpoint (~600 MB) is downloaded automatically from Hugging Face
  on first use and cached at ~/.cache/veinseg/checkpoint.pth.
""")
    sys.exit(0)


def main():
    if len(sys.argv) == 1 or any(a in sys.argv for a in ("-h", "--help")):
        _print_help_and_exit()

    import argparse
    ap = argparse.ArgumentParser(prog="veinseg", add_help=False)
    ap.add_argument("-i",  required=True, metavar="QSM")
    ap.add_argument("-r",  required=True, choices=list(METHOD_TO_IDX.keys()))
    ap.add_argument("-f",  required=True, choices=["3t", "7t"])
    ap.add_argument("-o",  required=True, metavar="BINARY")
    ap.add_argument("-p",  required=True, metavar="PROB")
    ap.add_argument("--local-field",       default=None, metavar="PATH")
    ap.add_argument("--local-field-units", default="auto",
                    choices=["auto", "hz", "ppm"])
    ap.add_argument("--b0", nargs=3, type=float, default=[0., 0., 1.],
                    metavar=("X", "Y", "Z"))
    ap.add_argument("--step-size",  type=float, default=0.5)
    ap.add_argument("--no-tta",     action="store_true")
    ap.add_argument("--device",     default="auto",
                    choices=["auto", "cpu", "cuda"])
    ap.add_argument("--threshold",  type=float, default=0.5)
    ap.add_argument("--out-field",  default=None, metavar="PATH")
    ap.add_argument("--out-frangi", default=None, metavar="PATH")
    args = ap.parse_args()

    device = _pick_device(args.device)
    print(f"[veinseg] device: {device}")

    # ---- load QSM ----
    print(f"[veinseg] loading {args.i}")
    qsm_nii = nib.load(args.i)
    qsm     = np.nan_to_num(qsm_nii.get_fdata(dtype=np.float32),
                             nan=0., posinf=0., neginf=0.)
    vx, vy, vz = voxel_size_from_affine(qsm_nii.affine)
    print(f"[veinseg] shape: {qsm.shape}  spacing: {vx:.3f} x {vy:.3f} x {vz:.3f} mm")

    b0_img = b0_dir_from_image_affine(qsm_nii.affine, np.array(args.b0))
    qsm_t  = torch.from_numpy(qsm).float()

    # ---- channel 1: local field (ppm) ----
    if args.local_field:
        print(f"[veinseg] loading local field from {args.local_field}")
        lf_nii = nib.load(args.local_field)
        ch1    = np.nan_to_num(lf_nii.get_fdata(dtype=np.float32),
                               nan=0., posinf=0., neginf=0.)
        units = args.local_field_units
        if units == "auto":
            abs_p995 = float(np.percentile(np.abs(ch1[ch1 != 0]), 99.5)) \
                       if (ch1 != 0).any() else 0.
            units = "hz" if abs_p995 > 5.0 else "ppm"
            print(f"[veinseg] local field abs-p99.5={abs_p995:.4f} "
                  f"-> auto-detected: {units}")
        HZ_TO_PPM = {"7t": 1.0 / (42.5774 * 7.0), "3t": 1.0 / (42.5774 * 3.0)}
        if units == "hz":
            ch1 = ch1 * HZ_TO_PPM[args.f.lower()]
            print(f"[veinseg] converted Hz -> ppm "
                  f"(factor={HZ_TO_PPM[args.f.lower()]:.6f})")
        if ch1.shape != qsm.shape:
            from scipy.ndimage import zoom as ndimage_zoom
            zf  = tuple(q / l for q, l in zip(qsm.shape, ch1.shape))
            print(f"[veinseg] resampling local field {ch1.shape} -> {qsm.shape}")
            ch1 = ndimage_zoom(ch1, zf, order=1).astype(np.float32)
    else:
        print("[veinseg] computing dipole local field from QSM ...")
        ch1 = dipole_field_from_chi_xyz(
            qsm_t, (vx, vy, vz), b0_img, return_units="ppm"
        ).numpy()

    # ---- channel 2: Frangi vesselness ----
    print("[veinseg] computing Frangi vesselness ...")
    qsm_z = (qsm_t - qsm_t.mean()) / (qsm_t.std() + 1e-8)
    with torch.no_grad():
        V, _ = frangi_3d(qsm_z.unsqueeze(0).unsqueeze(0),
                         sigmas=FRANGI_SIGMAS,
                         alpha=0.2, beta=0.3, c=5.0,
                         bright_vessels=True)
    ch2 = V[0, 0].numpy()
    print(f"[veinseg] Frangi max={ch2.max():.4f}  nonzero={(ch2 > 0).sum()}")

    if args.out_field:
        _save_nii(ch1, qsm_nii, args.out_field)
        print(f"[veinseg] saved field  -> {args.out_field}")
    if args.out_frangi:
        _save_nii(ch2, qsm_nii, args.out_frangi)
        print(f"[veinseg] saved Frangi -> {args.out_frangi}")

    # ---- load checkpoint (downloads if needed) ----
    checkpoint_path = get_checkpoint()
    print(f"[veinseg] loading checkpoint from {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    from nnunetv2.utilities.plans_handling.plans_handler import PlansManager

    plans_manager         = PlansManager(ckpt["init_args"]["plans"])
    configuration_manager = plans_manager.get_configuration(
                                ckpt["init_args"]["configuration"])

    model = PriorGatedUNetWithAttentionInfer(
        in_channels=3, out_channels=2,
        patch_size=tuple(configuration_manager.patch_size),
        deep_supervision=False, num_domains=5, domain_embed_dim=32,
    )
    model.load_state_dict(ckpt["network_weights"], strict=False)
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
        dataset_json=ckpt["init_args"]["dataset_json"],
        trainer_name=ckpt["trainer_name"],
        inference_allowed_mirroring_axes=ckpt["inference_allowed_mirroring_axes"],
    )

    # ---- write temp files -> predict_from_files -> read back ----
    print("[veinseg] running inference ...")
    with tempfile.TemporaryDirectory() as tmpdir:
        ch0_path  = os.path.join(tmpdir, "case_0000.nii.gz")
        ch1_path  = os.path.join(tmpdir, "case_0001.nii.gz")
        ch2_path  = os.path.join(tmpdir, "case_0002.nii.gz")
        out_trunc = os.path.join(tmpdir, "case")

        nib.save(nib.Nifti1Image(qsm, qsm_nii.affine, qsm_nii.header), ch0_path)
        nib.save(nib.Nifti1Image(ch1, qsm_nii.affine, qsm_nii.header), ch1_path)
        nib.save(nib.Nifti1Image(ch2, qsm_nii.affine, qsm_nii.header), ch2_path)

        predictor.predict_from_files(
            list_of_lists_or_source_folder=[[ch0_path, ch1_path, ch2_path]],
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
