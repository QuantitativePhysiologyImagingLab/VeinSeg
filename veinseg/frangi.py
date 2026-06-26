import argparse
import numpy as np
import nibabel as nib

import torch
import torch.nn.functional as F
from torch import Tensor

from typing import Tuple, List

from typing import Sequence, Tuple, Union
from typing import Optional

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

# -----------------------------
# 1D Gaussian & its derivatives
# -----------------------------
def _gaussian_1d(sigma: float, radius: Optional[int] = None, device=None, dtype=None) -> Tensor:
    if radius is None:
        radius = max(1, int(torch.ceil(torch.tensor(3*sigma)).item()))
    x = torch.arange(-radius, radius+1, device=device, dtype=dtype)
    g = torch.exp(-0.5 * (x / sigma) ** 2)
    g = g / g.sum()
    return g

def _gaussian_1d_first_deriv(sigma: float, radius: Optional[int] = None, device=None, dtype=None) -> Tensor:
    # d/dx of Gaussian (unnormalized), normalized to zero mean
    g = _gaussian_1d(sigma, radius, device, dtype)
    x = torch.arange(-(g.numel()//2), g.numel()//2 + 1, device=device, dtype=dtype)
    dg = -(x/(sigma**2)) * g
    dg = dg - dg.mean()  # zero-sum, helps stability
    return dg

def _gaussian_1d_second_deriv(sigma: float, radius: Optional[int] = None, device=None, dtype=None) -> Tensor:
    # d^2/dx^2 of Gaussian (LoG component)
    g = _gaussian_1d(sigma, radius, device, dtype)
    x = torch.arange(-(g.numel()//2), g.numel()//2 + 1, device=device, dtype=dtype)
    d2g = ((x**2)/(sigma**4) - 1/(sigma**2)) * g
    d2g = d2g - d2g.mean()  # zero-sum
    return d2g

# -----------------------------
# Separable 3D convolution
# -----------------------------
def _sep_conv3d(x: Tensor, kx: Tensor, ky: Tensor, kz: Tensor, groups: int = 1) -> Tensor:
    """
    Apply separable conv with reflect padding along z (depth), y (height), x (width).
    x: (B, C, D, H, W); k*: (K,)
    """
    B, C, D, H, W = x.shape
    device, dtype = x.device, x.dtype

    def conv1d_along(x: Tensor, k: Tensor, dim: int) -> Tensor:
        # dim: 0->z, 1->y, 2->x (spatial axes within (D,H,W))
        pad = [0, 0, 0, 0, 0, 0]  # (W_left, W_right, H_top, H_bottom, D_front, D_back)
        ksz = k.numel()
        r = ksz // 2
        if dim == 2:     # x/width
            pad[0] = pad[1] = r
            k3 = k.view(1, 1, 1, 1, ksz)
        elif dim == 1:   # y/height
            pad[2] = pad[3] = r
            k3 = k.view(1, 1, 1, ksz, 1)
        else:            # z/depth
            pad[4] = pad[5] = r
            k3 = k.view(1, 1, ksz, 1, 1)

        x = F.pad(x, pad, mode='reflect')
        weight = k3.to(device=device, dtype=dtype).repeat(C, 1, 1, 1, 1)
        return F.conv3d(x, weight, groups=C)

    x = conv1d_along(x, kz, 0)
    x = conv1d_along(x, ky, 1)
    x = conv1d_along(x, kx, 2)
    return x

# -----------------------------
# Hessian (scale-normalized)
# -----------------------------
def _hessian_3d(I: Tensor, sigma: float) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """
    Compute scale-normalized second-order partials at scale sigma.
    I: (B,1,D,H,W)
    Returns: I_xx, I_yy, I_zz, I_xy, I_xz, I_yz
    """
    device, dtype = I.device, I.dtype
    g  = _gaussian_1d(sigma, device=device, dtype=dtype)
    d1 = _gaussian_1d_first_deriv(sigma, device=device, dtype=dtype)
    d2 = _gaussian_1d_second_deriv(sigma, device=device, dtype=dtype)

    # pure seconds
    I_xx = _sep_conv3d(I, d2, g,  g)
    I_yy = _sep_conv3d(I, g,  d2, g)
    I_zz = _sep_conv3d(I, g,  g,  d2)

    # mixed seconds: d^2/dxdy = d/dx(d/dy)
    I_xy = _sep_conv3d(I, d1, d1, g)
    I_xz = _sep_conv3d(I, d1, g,  d1)
    I_yz = _sep_conv3d(I, g,  d1, d1)

    # Lindeberg normalization for 2nd-order derivatives
    s2 = sigma**2
    I_xx = s2*I_xx; I_yy = s2*I_yy; I_zz = s2*I_zz
    I_xy = s2*I_xy; I_xz = s2*I_xz; I_yz = s2*I_yz
    return I_xx, I_yy, I_zz, I_xy, I_xz, I_yz

# -----------------------------
# Frangi 3D (single scale)
# -----------------------------
def _frangi_3d_single(
    I: Tensor,
    sigma: float,
    alpha: float = 0.5,
    beta: float = 0.5,
    c: float = 15.0,
    bright_vessels: bool = True
) -> Tuple[Tensor, Tensor]:
    """
    Single-scale vesselness and axis.
    I: (B,1,D,H,W) normalized
    Returns: V (B,1,D,H,W) in [0,1], axis e_vec (B,3,D,H,W)
    """
    B, C, D, H, W = I.shape
    device, dtype = I.device, I.dtype

    I_xx, I_yy, I_zz, I_xy, I_xz, I_yz = _hessian_3d(I, sigma)

    H11, H22, H33 = I_xx, I_yy, I_zz
    H12, H13, H23 = I_xy, I_xz, I_yz

    # Stack to (B,1,D,H,W,3,3)
    H = torch.stack([
        torch.stack([H11, H12, H13], dim=-1),
        torch.stack([H12, H22, H23], dim=-1),
        torch.stack([H13, H23, H33], dim=-1),
    ], dim=-2).contiguous()

    # Flatten -> (B, D*H*W, 3, 3)
    B, _, D, H_, W_, _, _ = H.shape
    H = H.view(B, 1, D, H_, W_, 3, 3).flatten(1, 4)

    evals, evecs = torch.linalg.eigh(H)

    # Sort by |lambda|
    idx = torch.argsort(evals.abs(), dim=-1, stable=True)
    evals = torch.gather(evals, -1, idx)
    evecs = torch.gather(evecs, -1, idx.unsqueeze(-2).expand(-1, -1, 3, -1))

    l1, l2, l3 = evals[...,0], evals[...,1], evals[...,2]
    e_axis = evecs[...,0]

    eps = 1e-12
    Ra = (l2.abs() / (l3.abs() + eps))
    Rb = (l1.abs() / torch.sqrt(l2.abs() * l3.abs() + eps))
    S  = torch.sqrt((l1**2 + l2**2 + l3**2).clamp_min(eps))

    # Vesselness
    expRa = 1.0 - torch.exp(-(Ra**2) / (2*alpha**2))  # suppress sheets
    expRb = torch.exp(-(Rb**2) / (2*beta**2))
    expS  = 1.0 - torch.exp(-(S**2) / (2*c**2))
    V = expRa * expRb * expS

    if bright_vessels:
        mask = ((l2 < 0) & (l3 < 0)).float()
    else:
        mask = ((l2 > 0) & (l3 > 0)).float()

    V = V * mask
    V = V.view(B, 1, D, H_, W_).clamp(0, 1)

    e_axis = e_axis.view(B, D, H_, W_, 3).permute(0, 4, 1, 2, 3)
    e_axis = e_axis / (e_axis.norm(dim=1, keepdim=True).clamp_min(1e-8))

    return V, e_axis

# -----------------------------
# Multiscale Frangi 3D
# -----------------------------
@torch.no_grad()
def frangi_3d(
    I: Tensor,
    sigmas: Sequence[float] = (0.6, 0.9, 1.2, 1.8),
    alpha: float = 0.5,
    beta: float = 0.5,
    c: float = 15.0,
    bright_vessels: bool = True,
    return_scale: bool = False
) -> Union[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor, Tensor]]:
    """
    Multiscale 3D Frangi vesselness.

    Args:
        I: (B,1,D,H,W) image (recommend z-score per volume).
        sigmas: Gaussian scales in voxel units (span expected vessel radii).
        alpha, beta, c: Frangi parameters.
        bright_vessels: True for bright-on-dark tubes; False for dark-on-bright.
        return_scale: if True, also return the winning sigma map.

    Returns:
        V: (B,1,D,H,W) vesselness in [0,1]
        axis: (B,3,D,H,W) principal (tube) axis unit vector
        s_map (optional): (B,1,D,H,W) argmax scale per voxel
    """
    assert I.ndim == 5 and I.shape[1] == 1, "I must be (B,1,D,H,W)"
    B, _, D, H, W = I.shape

    V_best = torch.zeros_like(I)
    axis_best = torch.zeros(B, 3, D, H, W, device=I.device, dtype=I.dtype)
    s_map = torch.zeros_like(I)

    for s in sigmas:
        V_s, axis_s = _frangi_3d_single(I, s, alpha, beta, c, bright_vessels)
        better = V_s > V_best
        V_best = torch.where(better, V_s, V_best)
        axis_best = torch.where(better.expand_as(axis_best), axis_s, axis_best)
        s_map = torch.where(better, torch.full_like(V_s, float(s)), s_map)

    if return_scale:
        return V_best, axis_best, s_map
    return V_best, axis_best


def save_like(tensor, ref_nii, out_path):
    tensor = tensor.detach().cpu()
    if tensor.ndim == 5:
        tensor = tensor[0, 0]
    elif tensor.ndim == 4:
        tensor = tensor[0]
    nii = nib.Nifti1Image(tensor.numpy().astype(np.float32), ref_nii.affine, ref_nii.header)
    nib.save(nii, out_path)
    print(f"Saved {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("chi_path", help="Input χ (ppm) NIfTI")
    ap.add_argument("out_path", help="Output NIfTI (ppm by default)")
    ap.add_argument("--units", choices=["ppm", "hz", "rad"], default="ppm")
    ap.add_argument("--B0", type=float, default=None)
    ap.add_argument("--TE", type=float, default=None)
    ap.add_argument("--b0-world", type=float, nargs=3, default=(0.0, 0.0, 1.0))
    ap.add_argument("--dtype64", action="store_true")
    args = ap.parse_args()

    chi_nii = nib.load(args.chi_path)
    chi = chi_nii.get_fdata().astype(np.float32)
    chi = np.nan_to_num(chi, nan=0.0, posinf=0.0, neginf=0.0)

    sx, sy, sz = voxel_size_from_affine(chi_nii.affine)
    b0_img = b0_dir_from_image_affine(chi_nii.affine, args.b0_world)

    print(chi_nii.affine)
    print(b0_img)

    dev = torch.device("cpu")
    chi_t = torch.from_numpy(chi).to(dev).unsqueeze(0).unsqueeze(0)

    vol = (chi_t - chi_t.mean(dim=(2,3,4), keepdim=True)) / (chi_t.std(dim=(2,3,4), keepdim=True) + 1e-8)

    V_pos, axis_neg, s_map_neg = frangi_3d(
        vol,
        sigmas=(0.1, 0.5, 0.8, 1.0),
        alpha=0.2, beta=0.3, c=5.0,
        bright_vessels=True,
        return_scale=True
    )

    save_like(V_pos, chi_nii, args.out_path)
    print(f"Saved {args.units} field to: {args.out_path}")
    print(f"voxel size (mm): {sx:.4f}, {sy:.4f}, {sz:.4f}")
    print(f"B0 (image axes): {b0_img}")

if __name__ == "__main__":
    main()
