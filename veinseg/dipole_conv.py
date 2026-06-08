#!/usr/bin/env python3
import argparse
import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F

# ---------- helpers ----------

def voxel_size_from_affine(aff: np.ndarray):
    # mm/voxel from column norms of the 3x3 part
    R = aff[:3, :3]
    sx, sy, sz = np.linalg.norm(R[:, 0]), np.linalg.norm(R[:, 1]), np.linalg.norm(R[:, 2])
    return float(sx), float(sy), float(sz)

def b0_dir_from_image_affine(aff: np.ndarray, b0_world=(0.0, 0.0, 1.0)):
    """
    Convert B0 direction given in world/scanner axes to image axes using the affine.
    We solve R * c ≈ b0_world for c, where R is the 3x3 from the affine.
    """
    R = aff[:3, :3]
    c, *_ = np.linalg.lstsq(R, np.asarray(b0_world, dtype=np.float64), rcond=None)
    c = c / (np.linalg.norm(c) + 1e-30)
    return c.astype(np.float32)

def dipole_field_from_chi_xyz(
    chi_xyz: torch.Tensor,     # (X,Y,Z) float32/64 in ppm
    voxel_size_xyz,            # (sx,sy,sz) mm
    b0_dir_xyz,                # (3,) unit vector in image (x,y,z) axes
    return_units="ppm",        # 'ppm' | 'hz' | 'rad'
    B0_T: float = None,        # Tesla (needed for 'hz'/'rad')
    TE_s: float = None,        # seconds (needed for 'rad')
    use_double: bool = False
):
    """
    Δb_ppm = F^-1{ F{χ_ppm} * D(k) },  D(k) = 1/3 - [(k·b0)^2 / |k|^2], D(0)=0
    """
    assert chi_xyz.ndim == 3, "chi must be (X,Y,Z)"
    device = chi_xyz.device
    work_dtype = torch.float64 if use_double else torch.float32
    chi = chi_xyz.to(work_dtype)

    # voxel spacing & grids (cycles/mm)
    sx, sy, sz = map(float, voxel_size_xyz)
    X, Y, Z = chi.shape
    kx = torch.fft.fftfreq(X, d=sx, device=device, dtype=work_dtype)
    ky = torch.fft.fftfreq(Y, d=sy, device=device, dtype=work_dtype)
    kz = torch.fft.fftfreq(Z, d=sz, device=device, dtype=work_dtype)
    KX, KY, KZ = torch.meshgrid(kx, ky, kz, indexing='ij')

    b0 = torch.as_tensor(b0_dir_xyz, dtype=work_dtype, device=device)
    b0 = b0 / (b0.norm() + 1e-30)

    kb = b0[0]*KX + b0[1]*KY + b0[2]*KZ
    k2 = KX*KX + KY*KY + KZ*KZ
    eps = torch.finfo(work_dtype).eps
    D = (1.0/3.0) - (kb*kb) / (k2 + eps)
    D[0, 0, 0] = 0.0  # DC

    Chi_k = torch.fft.fftn(chi)
    field_ppm = torch.fft.ifftn(Chi_k * D).real

    if return_units == "ppm":
        return field_ppm.to(chi_xyz.dtype)

    # unit conversions
    if B0_T is None:
        raise ValueError("B0_T is required for return_units != 'ppm'")
    gamma_hz_per_T = 42.577478518e6  # proton gyromagnetic ratio (Hz/T)
    field_hz = gamma_hz_per_T * B0_T * (field_ppm * 1e-6)
    if return_units == "hz":
        return field_hz.to(chi_xyz.dtype)
    if return_units == "rad":
        if TE_s is None:
            raise ValueError("TE_s is required for return_units 'rad'")
        phase_rad = (2.0 * np.pi) * TE_s * field_hz
        return phase_rad.to(chi_xyz.dtype)
    raise ValueError("return_units must be one of {'ppm','hz','rad'}")

def save_like(ref_nii: nib.Nifti1Image, data_xyz: np.ndarray, out_path: str):
    out = nib.Nifti1Image(data_xyz.astype(np.float32), ref_nii.affine, ref_nii.header)
    # ensure header data type sane
    out.set_data_dtype(np.float32)
    nib.save(out, out_path)

# ---------- main ----------

def main():
    ap = argparse.ArgumentParser(
        description="Dipole convolution: compute local field from susceptibility (χ, ppm)."
    )
    ap.add_argument("chi_path", help="Input χ (ppm) NIfTI")
    ap.add_argument("out_path", help="Output NIfTI (ppm by default)")
    ap.add_argument("--units", choices=["ppm", "hz", "rad"], default="ppm",
                    help="Output units (default: ppm)")
    ap.add_argument("--B0", type=float, default=None,
                    help="B0 in Tesla (needed for --units hz/rad)")
    ap.add_argument("--TE", type=float, default=None,
                    help="Echo time in seconds (needed for --units rad)")
    ap.add_argument("--b0-world", type=float, nargs=3, default=(0.0, 0.0, 1.0),
                    help="B0 direction in world/scanner axes (default: 0 0 1)")
    ap.add_argument("--dtype64", action="store_true",
                    help="Use float64 for convolution (slower, more stable)")
    args = ap.parse_args()

    # load χ
    chi_nii = nib.load(args.chi_path)
    chi = chi_nii.get_fdata().astype(np.float32)  # (X,Y,Z) in nib (x,y,z)
    chi = np.nan_to_num(chi, nan=0.0, posinf=0.0, neginf=0.0)

    # voxel size & B0 in image axes
    sx, sy, sz = voxel_size_from_affine(chi_nii.affine)
    b0_img = b0_dir_from_image_affine(chi_nii.affine, args.b0_world)

    print(chi_nii.affine)
    print(b0_img)

    # torch tensors
    dev = torch.device("cpu")
    chi_t = torch.from_numpy(chi).to(dev)

    field_t = dipole_field_from_chi_xyz(
        chi_xyz=chi_t,
        voxel_size_xyz=(sx, sy, sz),
        b0_dir_xyz=b0_img,
        return_units=args.units,
        B0_T=args.B0,
        TE_s=args.TE,
        use_double=args.dtype64
    )

    field = field_t.cpu().numpy()
    save_like(chi_nii, field, args.out_path)

    # small print
    print(f"Saved {args.units} field to: {args.out_path}")
    print(f"voxel size (mm): {sx:.4f}, {sy:.4f}, {sz:.4f}")
    print(f"B0 (image axes): {b0_img}")

if __name__ == "__main__":
    main()