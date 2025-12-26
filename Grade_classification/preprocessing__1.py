#!/usr/bin/env python3
from pathlib import Path
import re
import os
import json
import gc
import numpy as np
import pandas as pd
import nibabel as nib
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
from matplotlib.patches import Rectangle
from typing import List, Tuple, Optional, Dict, Any
from scipy.ndimage import uniform_filter
import matplotlib.pyplot as plt
from collections import Counter
from scipy.ndimage import rotate, map_coordinates, gaussian_filter
from numpy.lib.format import open_memmap
from tqdm.auto import tqdm

# --- EDIT THESE ---
#ASL_DIR = Path("~/ad_MMSE_corr/ASL_project/raw_ASL_vols").expanduser()
#EXCEL   = Path("~/ad_MMSE_corr/ASL_project/UCSF-PDGM-metadata_v2.xlsx").expanduser()
ASL_DIR = Path("~/programs/new_folder/raw").expanduser()
EXCEL   = Path("~/programs/new_folder/UCSF-PDGM-metadata_v2.xlsx").expanduser()

FNAME_RE = re.compile(r'^(UCSF-PDGM-(\d{4}))_ASL\.nii\.gz$', re.IGNORECASE)

_ID_RE = re.compile(r'(UCSF-PDGM-\d{4})', re.IGNORECASE)

def _subject_id_from_path(p: str) -> str:
    m = _ID_RE.search(os.path.basename(p))
    return m.group(1) if m else Path(p).stem

def _center_of_mass_top_pct(arr: np.ndarray, pct: float = 10.0) -> Tuple[float, float, float]:
    """
    Center of mass of the top `pct`% brightest voxels.
    Works for 3D or 4D (takes max over last axis for 4D).
    Returns (cx, cy, cz) in voxel index space (floats).
    """
    a = np.asarray(arr, dtype=float)
    if a.ndim == 4:
        a = a.max(axis=-1)  # brightness proxy across time/PLD
    # threshold at percentile
    thr = np.percentile(a, 100.0 - pct)
    mask = a >= thr
    if not np.any(mask):
        # fallback: use global max voxel
        zyx = np.unravel_index(np.argmax(a), a.shape)
        return float(zyx[0]), float(zyx[1]), float(zyx[2])
    idx = np.argwhere(mask)  # (N, 3) in (x,y,z) index order
    # simple (unweighted) COM over selected brightest voxels
    cx, cy, cz = idx.mean(axis=0)
    return float(cx), float(cy), float(cz)

def _square_pixels(side_mm: float, zooms: Tuple[float, float, float]) -> Tuple[float, float]:
    """
    Convert a square side length in mm to half-side in pixels along x and y.
    zooms: (dx_mm, dy_mm, dz_mm)
    """
    dx, dy, _ = float(zooms[0]), float(zooms[1]), float(zooms[2])
    half_side_px_x = (side_mm / 2.0) / dx
    half_side_px_y = (side_mm / 2.0) / dy
    return half_side_px_x, half_side_px_y

def save_tumor_center_axials(
    X: List[np.ndarray],
    paths: List[str],
    out_dir: str = "tumor_center_axials",
    select: Optional[int] = 10,       # None -> use all
    side_mm: float = 100.0,           # cube side in mm (applied along x,y,z)
    top_pct: float = 10.0,            # brightest-voxel percentage for COM
    randomize: bool = False,
    save: bool = True,                # if False: no figures saved
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """
    For `select` scans:
      1) Find COM of the top `top_pct`% brightest voxels.
      2) Build a 3D cube (x0,x1,y0,y1,z0,z1) centered at COM with side = `side_mm`
         converted via voxel spacing; cube is clipped to volume bounds.
      3) Return metadata for each scan (coords only). Optionally save axial/coronal/sagittal PNGs.

    Returns: a list of dicts per selected scan:
      {
        'subject_id': str,
        'path': str,
        'spacing_mm': (dx,dy,dz),
        'shape': (nx,ny,nz),
        'com_vox': (cx,cy,cz),          # float voxel coords
        'com_idx': (x,y,z),             # int rounded indices
        'bbox_vox': (x0,x1,y0,y1,z0,z1) # int ranges, half-open [x0,x1), etc.
      }
    """
    assert len(X) == len(paths), "X and paths must have same length"
    n = len(X)

    rng = np.random.default_rng(seed)
    idxs = np.arange(n)
    if randomize:
        rng.shuffle(idxs)
    if select is not None:
        idxs = idxs[:int(select)]

    outp = Path(out_dir)
    if save:
        outp.mkdir(parents=True, exist_ok=True)

    results: List[Dict[str, Any]] = []

    for i in idxs:
        arr = X[i]
        p = paths[i]
        sid = _subject_id_from_path(p)

        # spacing (mm)
        dx, dy, dz = nib.load(p).header.get_zooms()[:3]

        # choose 3D volume to analyze/display
        vol3d = arr[..., 0] if arr.ndim == 4 else arr
        nx, ny, nz = vol3d.shape

        # COM of brightest voxels (voxel coords, floats)
        cx, cy, cz = _center_of_mass_top_pct(arr, pct=top_pct)

        # Integer indices (clamped)
        x = int(round(cx)); y = int(round(cy)); z = int(round(cz))
        x = max(0, min(x, nx - 1))
        y = max(0, min(y, ny - 1))
        z = max(0, min(z, nz - 1))

        # Cube half-sides in *pixels*
        half_x = (side_mm / 2.0) / dx
        half_y = (side_mm / 2.0) / dy
        half_z = (side_mm / 2.0) / dz

        # Build half-open voxel ranges [x0,x1), clipped to bounds; keep COM centered
        x0 = int(np.floor(cx - half_x)); x1 = int(np.ceil(cx + half_x))
        y0 = int(np.floor(cy - half_y)); y1 = int(np.ceil(cy + half_y))
        z0 = int(np.floor(cz - half_z)); z1 = int(np.ceil(cz + half_z))

        x0 = max(0, x0); y0 = max(0, y0); z0 = max(0, z0)
        x1 = min(nx, x1); y1 = min(ny, y1); z1 = min(nz, z1)

        # Ensure non-empty after clipping
        if x1 <= x0: x0, x1 = max(0, x-1), min(nx, x+1)
        if y1 <= y0: y0, y1 = max(0, y-1), min(ny, y+1)
        if z1 <= z0: z0, z1 = max(0, z-1), min(nz, z+1)

        results.append({
            "subject_id": sid,
            "path": p,
            "spacing_mm": (float(dx), float(dy), float(dz)),
            "shape": (nx, ny, nz),
            "com_vox": (float(cx), float(cy), float(cz)),
            "com_idx": (int(x), int(y), int(z)),
            "bbox_vox": (int(x0), int(x1), int(y0), int(y1), int(z0), int(z1)),
            "side_mm": float(side_mm),
            "top_pct": float(top_pct),
        })

        if not save:
            continue  # skip rendering/saving

        # ---- Axial view (x-y at fixed z) ----
        sl_ax = vol3d[:, :, z]
        half_x_ax = min((x1 - x0)/2, cx, nx - 1 - cx)
        half_y_ax = min((y1 - y0)/2, cy, ny - 1 - cy)
        left_ax   = cx - half_x_ax
        bottom_ax = cy - half_y_ax
        width_ax  = 2 * half_x_ax
        height_ax = 2 * half_y_ax

        plt.figure(figsize=(6, 6))
        plt.imshow(sl_ax.T, origin="lower", cmap="gray",
                   extent=[0, nx, 0, ny], aspect="equal")
        plt.gca().add_patch(Rectangle((left_ax, bottom_ax), width_ax, height_ax,
                                      fill=False, edgecolor="red", linewidth=2.0))
        plt.title(f"{sid} | Axial z={z} | {int(side_mm)}mm")
        plt.xlabel("x (vox)"); plt.ylabel("y (vox)")
        plt.tight_layout()
        plt.savefig(outp / f"{sid}_axial_z{z}_{int(side_mm)}mm.png", dpi=160)
        plt.close()

        # ---- Coronal view (x-z at fixed y) ----
        sl_cor = vol3d[:, y, :]
        half_x_c = min((x1 - x0)/2, cx, nx - 1 - cx)
        half_z_c = min((z1 - z0)/2, cz, nz - 1 - cz)
        left_cor   = cx - half_x_c
        bottom_cor = cz - half_z_c
        width_cor  = 2 * half_x_c
        height_cor = 2 * half_z_c

        plt.figure(figsize=(6, 6))
        plt.imshow(sl_cor.T, origin="lower", cmap="gray",
                   extent=[0, nx, 0, nz], aspect="equal")
        plt.gca().add_patch(Rectangle((left_cor, bottom_cor), width_cor, height_cor,
                                      fill=False, edgecolor="red", linewidth=2.0))
        plt.title(f"{sid} | Coronal y={y} | {int(side_mm)}mm")
        plt.xlabel("x (vox)"); plt.ylabel("z (vox)")
        plt.tight_layout()
        plt.savefig(outp / f"{sid}_coronal_y{y}_{int(side_mm)}mm.png", dpi=160)
        plt.close()

        # ---- Sagittal view (y-z at fixed x) ----
        sl_sag = vol3d[x, :, :]
        half_y_s = min((y1 - y0)/2, cy, ny - 1 - cy)
        half_z_s = min((z1 - z0)/2, cz, nz - 1 - cz)
        left_sag   = cy - half_y_s
        bottom_sag = cz - half_z_s
        width_sag  = 2 * half_y_s
        height_sag = 2 * half_z_s

        plt.figure(figsize=(6, 6))
        plt.imshow(sl_sag.T, origin="lower", cmap="gray",
                   extent=[0, ny, 0, nz], aspect="equal")
        plt.gca().add_patch(Rectangle((left_sag, bottom_sag), width_sag, height_sag,
                                      fill=False, edgecolor="red", linewidth=2.0))
        plt.title(f"{sid} | Sagittal x={x} | {int(side_mm)}mm")
        plt.xlabel("y (vox)"); plt.ylabel("z (vox)")
        plt.tight_layout()
        plt.savefig(outp / f"{sid}_sagittal_x{x}_{int(side_mm)}mm.png", dpi=160)
        plt.close()

    if save:
        print(f"[OK] Saved {len(idxs)*3} PNGs to: {outp.resolve()}")

    return results

def _norm_excel_id(x):
    """Normalize Excel ID to 'UCSF-PDGM-XXX' (3 digits)."""
    if pd.isna(x):
        return None
    s = str(x).strip()
    # Try to extract 3-4 trailing digits
    m = re.search(r'(\d{1,4})$', s)
    if not m:
        return None
    d = m.group(1)[-3:]  # take last 3 digits
    return f"UCSF-PDGM-{d.zfill(3)}"

def load_asl_and_labels(asl_dir=ASL_DIR, excel_path=EXCEL, max_scans=None):
    # 1) Collect files and derive 3-digit IDs to match Excel
    paths, ids_4d, ids_3d = [], [], []
    for p in sorted(Path(asl_dir).glob("*_ASL.nii.gz")):
        if max_scans is not None and len(paths) >= max_scans:
            break  # stop early
        m = FNAME_RE.match(p.name)
        if not m:
            continue
        full4 = m.group(1)
        digits4 = m.group(2)
        if digits4 in EXCLUDE_IDS:
            continue
        id3 = f"UCSF-PDGM-{digits4[-3:]}"
        paths.append(str(p))
        ids_4d.append(full4)
        ids_3d.append(id3)

    # 2) Load labels from Excel
    df = pd.read_excel(excel_path)
    df.columns = df.columns.astype(str).str.strip()

    # Assume exact column names as specified
    id_col = "ID"
    grade_col = "WHO CNS Grade"

    # Normalize Excel IDs to UCSF-PDGM-XXX and build dict
    df["_norm_id3"] = df[id_col].map(_norm_excel_id)
    label_map = dict(zip(df["_norm_id3"], df[grade_col]))

    # 3) Load images -> NumPy; gather labels aligned with files
    X, y, ids = [], [], []
    for p, id4, id3 in zip(paths, ids_4d, ids_3d):
        img = nib.load(p)
        arr = img.get_fdata(dtype=np.float32)  # NumPy array
        X.append(arr)
        # label can be float/int/NaN in Excel; keep as int if possible else np.nan
        lbl = label_map.get(id3, np.nan)
        try:
            lbl = int(float(lbl))
        except Exception:
            lbl = np.nan
        y.append(lbl)
        ids.append(id4)  # keep 4-digit version as canonical

    return X, y, ids, paths

def _moving_average(x, k: int):
    if k <= 1:
        return x
    k = int(k)
    if k % 2 == 0:  # make odd
        k += 1
    pad = k // 2
    xpad = np.pad(x, (pad, pad), mode='edge')
    kernel = np.ones(k, dtype=float) / k
    return np.convolve(xpad, kernel, mode='valid')

def plot_train_hist_outlines(
    volumes,
    labels=None,
    bins: int = 256,
    remove_background: bool = True,
    clip_percentiles=(1, 99),
    max_voxels: int = 1_000_000,
    smooth: int = 7,
    title: str = "ASL Train Intensity Distributions (Background Removed)"
):
    plt.figure(figsize=(10, 6))
    for i, arr in enumerate(volumes):
        a = np.asarray(arr, dtype=float).ravel()

        # Subsample for speed
        if a.size > max_voxels:
            idx = np.random.randint(0, a.size, size=max_voxels, dtype=np.int64)
            a = a[idx]

        # Remove background (mode)
        if remove_background:
            vals, counts = np.unique(np.round(a, 6), return_counts=True)
            bg_val = vals[counts.argmax()]
            a = a[a != bg_val]

        # Clip extremes
        lo, hi = np.percentile(a, clip_percentiles)
        a = a[(a >= lo) & (a <= hi)]

        # Histogram
        hist, edges = np.histogram(a, bins=bins, density=True)
        centers = 0.5 * (edges[:-1] + edges[1:])

        # Smooth
        if smooth > 1:
            hist = np.convolve(hist, np.ones(smooth)/smooth, mode="same")

        lbl = None
        if labels is not None and i < len(labels):
            lbl = str(labels[i])

        plt.plot(centers, hist, linewidth=1.2, alpha=0.9, label=lbl)

    plt.xlabel("Intensity")
    plt.ylabel("Density")
    if labels is not None and len(labels) <= 20:
        plt.legend(frameon=False)
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.title(title)
    plt.tight_layout()
    plt.savefig("histogram.png")
    plt.close()

def strip_background(arr, round_decimals=6):
    vals, counts = np.unique(np.round(arr.ravel(), round_decimals), return_counts=True)
    bg_val = vals[counts.argmax()]
    return arr[arr != bg_val]

def plot_train_histograms(volumes, bins=100, clip_percentiles=(0.5, 99.5)):
    plt.figure(figsize=(10, 6))
    for arr in volumes:
        a = np.asarray(arr, dtype=float).ravel()
        # remove background = most frequent value
        vals, counts = np.unique(np.round(a, 6), return_counts=True)
        bg_val = vals[counts.argmax()]
        a = a[a != bg_val]
        # clip
        lo, hi = np.percentile(a, clip_percentiles)
        a = a[(a >= lo) & (a <= hi)]
        # plain histogram
        plt.hist(a, bins=bins, density=True, alpha=0.4)
    plt.xlabel("Intensity")
    plt.ylabel("Density")
    plt.title("Histograms of Train Volumes (background removed)")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig("histogram.png")
    plt.close()

def background_ranges_with_overlay(
    volumes,
    paths=None,
    round_decimals=6,
    save_images=False,
    out_dir="bg_overlay",
    alpha=0.35,
    max_scans=None,
    return_masks=False,
    return_ranges=False,
    save_masks_nifti=False,
    mask_dir="bg_masks_nifti"
):
    if save_images:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
    if save_masks_nifti:
        Path(mask_dir).mkdir(parents=True, exist_ok=True)

    results = []
    masks_out = [] if return_masks else None

    scans_idx = range(len(volumes)) if max_scans is None else range(min(max_scans, len(volumes)))
    for i in scans_idx:
        arr = volumes[i]
        vol3d = arr[..., 0] if arr.ndim == 4 else arr
        nx, ny, nz = vol3d.shape

        flat = vol3d.ravel()
        rounded = np.round(flat, round_decimals)
        uniq, counts = np.unique(rounded, return_counts=True)
        mode_r = uniq[counts.argmax()]

        bg_mask = (np.round(vol3d, round_decimals) == mode_r)   # bool (nx,ny,nz)
        bg_vals = flat[rounded == mode_r]
        lo, hi = float(bg_vals.min()), float(bg_vals.max())
        results.append((lo, hi))

        # optional return of masks
        if return_masks:
            masks_out.append(bg_mask)

        if save_masks_nifti:
            if paths is None:
                raise ValueError("paths are required to save NIfTI masks.")
            img = nib.load(paths[i])
            mask_img = nib.Nifti1Image(bg_mask.astype(np.uint8), img.affine, img.header)
            sid = _subject_id_from_path(paths[i]) if paths else f"scan_{i:04d}"
            nib.save(mask_img, Path(mask_dir) / f"{sid}_bgmask.nii.gz")

        # optional overlays
        if save_images:
            from matplotlib import pyplot as plt
            sid = _subject_id_from_path(paths[i]) if paths else f"scan_{i:04d}"
            mx, my, mz = nx // 2, ny // 2, nz // 2
            tissue = vol3d[~bg_mask]
            vmin, vmax = (np.percentile(tissue, (1, 99)) if tissue.size >= 10
                          else np.percentile(flat, (1, 99)))

            # axial
            plt.figure(figsize=(5,5))
            plt.imshow(vol3d[:, :, mz].T, origin="lower", cmap="gray",
                       extent=[0, nx, 0, ny], vmin=vmin, vmax=vmax, aspect="equal")
            plt.imshow(bg_mask[:, :, mz].T, origin="lower", cmap="Reds",
                       extent=[0, nx, 0, ny], alpha=alpha, aspect="equal")
            plt.title(f"{sid} | Axial z={mz} | bg≈[{lo:.6g},{hi:.6g}]")
            plt.tight_layout()
            plt.savefig(Path(out_dir) / f"{sid}_axial_bg.png", dpi=150)
            plt.close()

            # coronal
            plt.figure(figsize=(5,5))
            plt.imshow(vol3d[:, my, :].T, origin="lower", cmap="gray",
                       extent=[0, nx, 0, nz], vmin=vmin, vmax=vmax, aspect="equal")
            plt.imshow(bg_mask[:, my, :].T, origin="lower", cmap="Reds",
                       extent=[0, nx, 0, nz], alpha=alpha, aspect="equal")
            plt.title(f"{sid} | Coronal y={my}")
            plt.tight_layout()
            plt.savefig(Path(out_dir) / f"{sid}_coronal_bg.png", dpi=150)
            plt.close()

            # sagittal
            plt.figure(figsize=(5,5))
            plt.imshow(vol3d[mx, :, :].T, origin="lower", cmap="gray",
                       extent=[0, ny, 0, nz], vmin=vmin, vmax=vmax, aspect="equal")
            plt.imshow(bg_mask[mx, :, :].T, origin="lower", cmap="Reds",
                       extent=[0, ny, 0, nz], alpha=alpha, aspect="equal")
            plt.title(f"{sid} | Sagittal x={mx}")
            plt.tight_layout()
            plt.savefig(Path(out_dir) / f"{sid}_sagittal_bg.png", dpi=150)
            plt.close()

    if return_masks and return_ranges:
        return (results, masks_out)
    if return_masks:
        return masks_out
    elif return_ranges:
        return results
    else:
        print("Returned nothing.")

def augment_dataset(X_train, y_train, *, seed=123):
    """
    Returns (X_train_aug, y_train_aug) where augmented samples are appended.
    - Majority class (label==4): 3 augs per sample (randomly chosen from 4 types)
    - Minority classes (label!=4): 6 augs per sample (2x flip, 2x rotate, 2x elastic)
    - Unlabeled (NaN) samples are NOT augmented; they pass through unchanged.
    """
    rng = np.random.default_rng(seed)
    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train)

    # Helpers ---------------------------------------------------------------
    def _rand_flip(x, rng):
        # Randomly flip along a random non-empty subset of spatial axes
        axes = [0, 1, 2]
        k = rng.integers(1, 4)
        flip_axes = rng.choice(axes, size=k, replace=False)
        x2 = x.copy()
        for ax in flip_axes:
            x2 = np.flip(x2, axis=ax)
        return x2

    def _small_rotation(x, rng, max_deg=10.0):
        # Rotate around a random plane with small random angle in [-max_deg, max_deg]
        planes = [(0,1), (0,2), (1,2)]
        ax0, ax1 = planes[rng.integers(0, len(planes))]
        angle = rng.uniform(-max_deg, max_deg)
        # rotate each channel slice; keep shape (reshape=False), bilinear, nearest padding
        out = np.empty_like(x)
        C = x.shape[-1]
        for c in range(C):
            out[..., c] = rotate(x[..., c], angle=angle, axes=(ax0, ax1),
                                 reshape=False, order=1, mode='nearest')
        return out

    def _elastic_deform(x, rng, alpha=2.0, sigma=8.0):
        """
        Mild elastic deformation (displacement in ~[-alpha, alpha] voxels),
        smoothed with Gaussian(sigma). Applies identically to all channels.
        """
        X, Y, Z, C = x.shape
        # random fields per axis
        dx = rng.normal(0, 1, size=(X, Y, Z))
        dy = rng.normal(0, 1, size=(X, Y, Z))
        dz = rng.normal(0, 1, size=(X, Y, Z))
        dx = gaussian_filter(dx, sigma=sigma) * alpha
        dy = gaussian_filter(dy, sigma=sigma) * alpha
        dz = gaussian_filter(dz, sigma=sigma) * alpha

        # base grid
        gx, gy, gz = np.meshgrid(np.arange(X), np.arange(Y), np.arange(Z), indexing='ij')
        coords = np.array([gx + dx, gy + dy, gz + dz])

        out = np.empty_like(x)
        for c in range(C):
            out[..., c] = map_coordinates(
                x[..., c], coords, order=1, mode='nearest'
            )
        return out

    def _rician_noise(x, rng, sigma=0.02):
        """
        Adds Rician noise channel-wise: y = sqrt((x + n1)^2 + n2^2)
        sigma is relative to per-scan robust scale of x (MAD-based).
        """
        x2 = x.astype(np.float32, copy=True)
        # scale sigma by robust channel-wise magnitude to be contrast-aware
        C = x.shape[-1]
        for c in range(C):
            xc = x2[..., c]
            med = np.median(xc)
            mad = np.median(np.abs(xc - med))
            scale = (1.4826 * mad) if mad > 0 else (np.std(xc) + 1e-6)
            s = sigma * (scale if scale > 1e-6 else 1.0)
            n1 = rng.normal(0.0, s, size=xc.shape)
            n2 = rng.normal(0.0, s, size=xc.shape)
            x2[..., c] = np.sqrt((xc + n1)**2 + (n2**2))
        return x2

    AUG_FUNCS = {
        'flip':    _rand_flip,
        'rotate':  _small_rotation,
        'elastic': _elastic_deform,
        'rician':  _rician_noise,
    }

    # Aug plan --------------------------------------------------------------
    MAJ_LABEL = 4
    MAJ_PER_SAMPLE  = 3
    MIN_PER_SAMPLE  = 8
    MINORITY_PLAN = ['flip', 'flip', 'rotate', 'rotate', 'elastic', 'elastic', 'rician', 'rician']

    # Split labeled/unlabeled
    is_labeled = ~pd.isna(y_train)
    X_lab = X_train[is_labeled]
    y_lab = y_train[is_labeled].astype(int)
    X_unlab = X_train[~is_labeled]
    y_unlab = y_train[~is_labeled]  # NaNs preserved

    # Build outputs starting with originals
    X_out = [X_train]
    y_out = [y_train]

    # Iterate labeled and augment
    for i in range(len(X_lab)):
        x = X_lab[i]
        y = y_lab[i]

        if y == MAJ_LABEL:
            # 3 random picks from all 4 types (with replacement)
            picks = rng.choice(list(AUG_FUNCS.keys()), size=MAJ_PER_SAMPLE, replace=True)
        else:
            # exactly 6: 2x each of selected types (see plan)
            picks = MINORITY_PLAN

        # derive per-aug seeds for determinism
        for j, name in enumerate(picks):
            fn = AUG_FUNCS[name]
            # Deterministic child RNG using SeedSequence (stable across runs)
            op_index = {"flip":0, "rotate":1, "elastic":2, "rician":3}[name]
            ss = np.random.SeedSequence([seed, int(y), i, j, op_index])
            sub_rng = np.random.default_rng(ss.generate_state(1, dtype=np.uint32)[0])

            x_aug = fn(x, sub_rng).astype(x.dtype, copy=False)
            X_out.append(x_aug[None, ...])
            y_out.append(np.array([y]))

    X_train_aug = np.concatenate(X_out, axis=0)
    y_train_aug = np.concatenate(y_out, axis=0)

    return X_train_aug, y_train_aug

# ======================= ASL channel builders (batch-ready) =======================

def _ensure_odd(k: int) -> int:
    return k if (k % 2 == 1) else (k + 1)

def _local_stats_masked(x: np.ndarray, mask: np.ndarray, win: int):
    """
    Local masked mean/std using a cubic window of size 'win' (odd).
    Zeros outside mask do not contribute.
    """
    win = _ensure_odd(int(win))
    vol = float(win ** 3)

    xm     = x * mask
    sum_x  = uniform_filter(xm,     size=win, mode='nearest') * vol
    sum_x2 = uniform_filter(xm * x, size=win, mode='nearest') * vol
    sum_m  = uniform_filter(mask.astype(x.dtype), size=win, mode='nearest') * vol

    eps = 1e-6
    mu   = sum_x / (sum_m + eps)
    ex2  = sum_x2 / (sum_m + eps)
    var  = np.clip(ex2 - mu * mu, 0.0, None)
    std  = np.sqrt(var + eps)

    mu[mask == 0]  = 0.0
    std[mask == 0] = 1.0
    return mu, std

def _center_pad(arr: np.ndarray, target_shape: tuple, pad_value: float = 0.0) -> np.ndarray:
    """Center-pad a 3D or 4D array (last dim = channels allowed) to target (X,Y,Z[,C])."""
    assert arr.ndim in (3, 4)
    is_4d = (arr.ndim == 4)
    if is_4d:
        xyz = arr.shape[:3]
        C   = arr.shape[3]
        assert target_shape[3] == C, "Channel mismatch in _center_pad"
        tgt = target_shape[:3]
    else:
        xyz = arr.shape
        tgt = target_shape

    out = np.full(target_shape, pad_value, dtype=arr.dtype)
    # compute start indices
    starts = [(t - s)//2 for s, t in zip(xyz, tgt)]
    slices_out = tuple(slice(st, st + s) for st, s in zip(starts, xyz))
    if is_4d:
        out[slices_out + (slice(None),)] = arr
    else:
        out[slices_out] = arr
    return out

def make_asl_feature_volume_single(
    cbf: np.ndarray,
    brain_mask: np.ndarray,
    *,
    channels=("raw", "zg", "zl", "log"),
    low_clip_pct: float = 1.0,     # clip ONLY bottom p% inside brain; high tail untouched
    local_win: int = 15,           # voxels (odd); ~15 mm at 1 mm iso
    log_floor: float = 0.0,        # floor before log1p
    dtype=np.float32
) -> np.ndarray:
    """
    Build a (X,Y,Z,C) feature volume for ONE subject from CBF + brain_mask.
    Channels: any ordered subset of {"raw","zg","zl","log"}.
    """
    assert cbf.shape == brain_mask.shape and cbf.ndim == 3
    cbf = np.asarray(cbf, dtype=np.float32)
    M   = (np.asarray(brain_mask) > 0).astype(np.float32)

    # --- raw: clip ONLY the low tail inside brain
    brain_vals = cbf[M > 0]
    if brain_vals.size == 0:
        raise ValueError("Empty brain mask for this subject.")
    lo = np.percentile(brain_vals, low_clip_pct) if (low_clip_pct is not None and low_clip_pct > 0) else None
    if lo is not None:
        raw = np.where(M > 0, np.maximum(cbf, lo), 0.0)
    else:
        raw = np.where(M > 0, cbf, 0.0)

    out_map = {"raw": raw}

    # --- global robust z
    if "zg" in channels:
        med = float(np.median(brain_vals))
        mad = float(np.median(np.abs(brain_vals - med)))
        denom = 1.4826 * mad if mad > 0 else (np.std(brain_vals) + 1e-6)
        zg = (raw - med) / (denom + 1e-6)
        zg[M == 0] = 0.0
        out_map["zg"] = zg

    # --- local z
    if "zl" in channels:
        mu, sd = _local_stats_masked(raw, M, _ensure_odd(local_win))
        zl = (raw - mu) / sd
        zl[M == 0] = 0.0
        out_map["zl"] = zl

    # --- log
    if "log" in channels:
        logcbf = np.log1p(np.maximum(raw, log_floor))
        logcbf[M == 0] = 0.0
        out_map["log"] = logcbf

    # stack
    stacks = [out_map[ch].astype(dtype, copy=False) for ch in channels]
    X4 = np.stack(stacks, axis=-1)  # (X,Y,Z,C)
    return X4

def make_asl_feature_volume_batch(
    vols: list,
    masks: list,
    *,
    channels=("raw", "zg", "zl", "log"),
    low_clip_pct: float = 1.0,
    local_win: int = 15,
    log_floor: float = 0.0,
    dtype=np.float32,
    target_shape: tuple | None = None,  # (X,Y,Z) or (X,Y,Z,C). If None -> pad to per-channel C with XYZ = max dims across subjects
    pad_value: float = 0.0
) -> np.ndarray:
    """
    Build a single (N, X, Y, Z, C) ndarray for the entire dataset.

    - Per subject, computes requested channels with low-tail-only clipping.
    - If shapes differ, center-pads each sample to 'target_shape' (XYZ).
      If target_shape is None, uses max(X), max(Y), max(Z) across subjects.
    """
    assert len(vols) == len(masks) and len(vols) > 0
    N = len(vols)

    # Build one to know C and per-subject shape
    X0 = make_asl_feature_volume_single(
        vols[0], masks[0],
        channels=channels,
        low_clip_pct=low_clip_pct,
        local_win=local_win,
        log_floor=log_floor,
        dtype=dtype
    )
    C = X0.shape[-1]

    # Decide target shape
    if target_shape is None:
        # Pad to the maximum XYZ among subjects
        xs, ys, zs = [vols[0].shape[0]], [vols[0].shape[1]], [vols[0].shape[2]]
        for v in vols[1:]:
            xs.append(v.shape[0]); ys.append(v.shape[1]); zs.append(v.shape[2])
        tgt_xyz = (max(xs), max(ys), max(zs))
    else:
        # allow (X,Y,Z) or (X,Y,Z,C)
        tgt_xyz = target_shape[:3]

    tgt_shape_4d = (*tgt_xyz, C)
    X5 = np.full((N, *tgt_xyz, C), pad_value, dtype=dtype)

    # Place first subject
    X5[0] = _center_pad(X0, tgt_shape_4d, pad_value)

    # Process the rest
    for i in range(1, N):
        Xi = make_asl_feature_volume_single(
            vols[i], masks[i],
            channels=channels,
            low_clip_pct=low_clip_pct,
            local_win=local_win,
            log_floor=log_floor,
            dtype=dtype
        )
        X5[i] = _center_pad(Xi, tgt_shape_4d, pad_value)

    print("Returning array of shape: ", np.shape(X5))
    return X5  # shape = (N, X, Y, Z, C)

def boxes_from_meta(meta):
    """
    Convert the list of dicts returned by save_tumor_center_axials(...)
    into a list of voxel boxes (x0,x1,y0,y1,z0,z1).
    """
    return [tuple(d["bbox_vox"]) for d in meta]

def crop_X5_by_uniform_boxes(X5: np.ndarray, boxes: list) -> np.ndarray:
    """
    Crop a 5D array X5 (N, X, Y, Z, C) with per-subject boxes, assuming
    **all boxes have identical size (cube)**. Returns (N, Xc, Yc, Zc, C).
    Raises if any box size differs from the first.

    boxes[i] = (x0, x1, y0, y1, z0, z1), half-open in voxel indices.
    """
    assert X5.ndim == 5, f"X5 must be (N, X, Y, Z, C); got {X5.shape}"
    N = X5.shape[0]
    assert len(boxes) == N, f"len(boxes)={len(boxes)} must match N={N}"

    # reference cube from the first sample
    x0, x1, y0, y1, z0, z1 = map(int, boxes[0])
    dx, dy, dz = x1 - x0, y1 - y0, z1 - z0
    assert dx > 0 and dy > 0 and dz > 0, f"Invalid first box: {boxes[0]}"

    # pre-allocate output
    C = X5.shape[-1]
    Xc = np.empty((N, dx, dy, dz, C), dtype=X5.dtype)

    # crop each subject, verifying uniform shape
    for i, b in enumerate(boxes):
        bx0, bx1, by0, by1, bz0, bz1 = map(int, b)
        s = (bx1 - bx0, by1 - by0, bz1 - bz0)
        if s != (dx, dy, dz):
            raise ValueError(
                f"Box {i} size {s} != reference {(dx,dy,dz)}. "
                "This usually means your cubic window was clipped by volume bounds. "
                "Either adjust centers so the full cube fits, or switch to the padded cropper."
            )
        Xc[i] = X5[i, bx0:bx1, by0:by1, bz0:bz1, :]

    return Xc  # (N, Xc, Yc, Zc, C)

def invert_bg_to_brain_masks(bg_masks: list) -> list:
    """Invert background masks (list of np.bool_) to brain masks."""
    return [~np.asarray(bm, dtype=bool) for bm in bg_masks]

def boxes_from_meta_aligned(meta: list, ids: list) -> list:
    """
    Return boxes aligned to the order of `ids` (UCSF-PDGM-XXXX).
    Robust even if meta was randomized.
    """
    box_by_id = {d["subject_id"]: tuple(d["bbox_vox"]) for d in meta}
    missing = [sid for sid in ids if sid not in box_by_id]
    if missing:
        raise ValueError(f"Missing boxes for IDs: {missing[:5]}{'...' if len(missing)>5 else ''}")
    return [box_by_id[sid] for sid in ids]

def adjust_boxes_for_padding(boxes: list, vols: list, padded_xyz: tuple) -> list:
    """
    Shift original-coordinate boxes into the center-padded grid used in X_all.
    boxes[i] in original coords; vols[i].shape is original (X_i,Y_i,Z_i);
    padded_xyz is the (X_pad,Y_pad,Z_pad) of X_all.
    """
    Xp, Yp, Zp = padded_xyz
    adj = []
    for b, v in zip(boxes, vols):
        Xi, Yi, Zi = v.shape[:3]
        sx = (Xp - Xi) // 2
        sy = (Yp - Yi) // 2
        sz = (Zp - Zi) // 2
        x0, x1, y0, y1, z0, z1 = map(int, b)
        adj.append((x0+sx, x1+sx, y0+sy, y1+sy, z0+sz, z1+sz))
    return adj

def _class_counts(y):
    y = np.asarray(y)
    if y.dtype.kind == 'f':
        mask = ~pd.isna(y)
    else:
        mask = np.ones(len(y), dtype=bool)
    # cast to Python ints before counting
    vals = [int(v) for v in y[mask].astype(int).tolist()]
    cnt = Counter(vals)
    # ensure keys/values are plain ints (json-friendly)
    return {int(k): int(v) for k, v in sorted(cnt.items())}

def save_split(out_dir, X_train, y_train, ids_train, paths_train,
               X_test, y_test, ids_test, paths_test, notes=""):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    np.save(out_dir / "X_train.npy", X_train)
    np.save(out_dir / "y_train.npy", np.asarray(y_train))
    np.save(out_dir / "ids_train.npy", np.asarray(ids_train))
    np.save(out_dir / "paths_train.npy", np.asarray(paths_train))

    np.save(out_dir / "X_test.npy", X_test)
    np.save(out_dir / "y_test.npy", np.asarray(y_test))
    np.save(out_dir / "ids_test.npy", np.asarray(ids_test))
    np.save(out_dir / "paths_test.npy", np.asarray(paths_test))

    meta = {
        "X_train_shape": tuple(X_train.shape),
        "X_test_shape": tuple(X_test.shape),
        "train_classes": _class_counts(y_train),
        "test_classes": _class_counts(y_test),
        "notes": notes,
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"[OK] Saved augmented split to {out_dir.resolve()}")

def compute_brain_mask_from_mode(vol3d, round_decimals=6):
    flat = vol3d.ravel()
    rounded = np.round(flat, round_decimals)
    uniq, counts = np.unique(rounded, return_counts=True)
    bg_val = uniq[counts.argmax()]
    brain = (np.round(vol3d, round_decimals) != bg_val)
    return brain

def build_and_crop_one(arr, box, channels=("raw","zg","zl","log"), low_clip_pct=1.0, local_win=15):
    vol3d = arr[..., 0] if arr.ndim == 4 else arr
    brain_mask = compute_brain_mask_from_mode(vol3d)

    X4 = make_asl_feature_volume_single(
        vol3d, brain_mask,
        channels=channels,
        low_clip_pct=low_clip_pct,
        local_win=local_win,
        dtype=np.float32
    )

    x0,x1,y0,y1,z0,z1 = map(int, box)
    return X4[x0:x1, y0:y1, z0:z1, :]

def load_paths_and_labels(asl_dir=ASL_DIR, excel_path=EXCEL, max_scans=None):
    paths, ids_4d, ids_3d = [], [], []
    for p in sorted(Path(asl_dir).glob("*_ASL.nii.gz")):
        if max_scans is not None and len(paths) >= max_scans:
            break
        m = FNAME_RE.match(p.name)
        if not m:
            continue
        full4 = m.group(1); digits4 = m.group(2)
        if digits4 in EXCLUDE_IDS:
            continue
        ids_4d.append(full4)
        ids_3d.append(f"UCSF-PDGM-{digits4[-3:]}")
        paths.append(str(p))

    df = pd.read_excel(excel_path)
    df.columns = df.columns.astype(str).str.strip()
    df["_norm_id3"] = df["ID"].map(_norm_excel_id)
    label_map = dict(zip(df["_norm_id3"], df["WHO CNS Grade"]))

    y = []
    for id3 in ids_3d:
        v = label_map.get(id3, np.nan)
        try: v = int(float(v))
        except: v = np.nan
        y.append(v)
    return np.array(y, dtype=float), np.array(ids_4d), np.array(paths)

def compute_box_for_path(path, side_mm=70.0, top_pct=0.01):
    """Load one NIfTI, find COM of top pct%, make a mm-sized cube box in voxel coords."""
    img = nib.load(path)
    dx, dy, dz = img.header.get_zooms()[:3]
    arr = img.get_fdata(dtype=np.float32)
    vol3d = arr[..., 0] if arr.ndim == 4 else arr
    nx, ny, nz = vol3d.shape

    # COM of brightest voxels
    a = vol3d
    thr = np.percentile(a, 100.0 - top_pct)
    mask = a >= thr
    if not np.any(mask):
        zyx = np.unravel_index(np.argmax(a), a.shape)
        cx, cy, cz = map(float, zyx)
    else:
        idx = np.argwhere(mask)
        cx, cy, cz = idx.mean(axis=0)

    # half-sides in voxels from mm
    hx = (side_mm / 2.0) / dx
    hy = (side_mm / 2.0) / dy
    hz = (side_mm / 2.0) / dz

    x0 = max(0, int(np.floor(cx - hx))); x1 = min(nx, int(np.ceil(cx + hx)))
    y0 = max(0, int(np.floor(cy - hy))); y1 = min(ny, int(np.ceil(cy + hy)))
    z0 = max(0, int(np.floor(cz - hz))); z1 = min(nz, int(np.ceil(cz + hz)))
    if x1 <= x0: x0, x1 = max(0, int(round(cx))-1), min(nx, int(round(cx))+1)
    if y1 <= y0: y0, y1 = max(0, int(round(cy))-1), min(ny, int(round(cy))+1)
    if z1 <= z0: z0, z1 = max(0, int(round(cz))-1), min(nz, int(round(cz))+1)
    return (x0, x1, y0, y1, z0, z1)

def compute_brain_mask_fast(vol3d: np.ndarray) -> np.ndarray:
    # fast background removal; avoids huge uniques
    if np.count_nonzero(vol3d == 0) > vol3d.size * 0.05:
        return vol3d != 0
    thr = np.percentile(vol3d, 0.5)
    return vol3d > thr

def build_and_crop_one_from_path(path, box,
                                 channels=("raw","zg","zl","log"),
                                 low_clip_pct=1.0, local_win=15):
    img = nib.load(path)
    vol3d = img.get_fdata(dtype=np.float32)
    vol3d = vol3d[..., 0] if vol3d.ndim == 4 else vol3d
    brain_mask = compute_brain_mask_fast(vol3d)
    X4 = make_asl_feature_volume_single(
        vol3d, brain_mask,
        channels=channels,
        low_clip_pct=low_clip_pct,
        local_win=local_win,
        dtype=np.float32
    )
    x0,x1,y0,y1,z0,z1 = map(int, box)
    return X4[x0:x1, y0:y1, z0:z1, :]

def pad_to_shape(x, tgt_shape, pad_value=0.0):
    out = np.full(tgt_shape, pad_value, dtype=x.dtype)
    in_shape = x.shape
    slices_in = []
    slices_out = []
    for d in range(3):  # X,Y,Z (channels last)
        if in_shape[d] >= tgt_shape[d]:
            si = (in_shape[d] - tgt_shape[d]) // 2
            slices_in.append(slice(si, si + tgt_shape[d]))
            slices_out.append(slice(0, tgt_shape[d]))
        else:
            so = (tgt_shape[d] - in_shape[d]) // 2
            slices_in.append(slice(0, in_shape[d]))
            slices_out.append(slice(so, so + in_shape[d]))
    out[slices_out[0], slices_out[1], slices_out[2], :] = x[slices_in[0], slices_in[1], slices_in[2], :]
    return out

def compute_target_xyz(paths, side_mm=70.0):
    """
    Decide a fixed (X,Y,Z) crop size, in voxels, large enough for all scans,
    based on each scan's header zooms and the desired side length in mm.
    """
    maxX = maxY = maxZ = 0
    for p in paths:
        dx, dy, dz = nib.load(p).header.get_zooms()[:3]
        X = int(np.ceil(side_mm / dx))
        Y = int(np.ceil(side_mm / dy))
        Z = int(np.ceil(side_mm / dz))
        maxX = max(maxX, X)
        maxY = max(maxY, Y)
        maxZ = max(maxZ, Z)
    return (maxX, maxY, maxZ)

################################################################################################################
######                                                                                                     #####
################################################################################################################

EXCLUDE_IDS = [
        "0005", "0013", "0035", "0036", "0050", "0067", "0073", "0075", "0092", "0096", "0159", "0196",
        "0229", "0259", "0264", "0288", "0332", "0379", "0415", "0442", "0474", "0489", "0522", "0526",
]

if __name__ == "__main__":
    # --- 1) Only labels + ids + paths (NO volumes in RAM) ---
    y, ids, paths = load_paths_and_labels(max_scans=None)  # <- use the light loader!

    # --- 2) Split on labels/ids/paths only ---
    is_labeled   = ~pd.isna(y)
    y_lab        = y[is_labeled].astype(int)
    ids_lab      = ids[is_labeled]
    paths_lab    = paths[is_labeled]
    ids_unlab    = ids[~is_labeled]
    paths_unlab  = paths[~is_labeled]

    cnt = Counter(y_lab.tolist())
    rare_classes     = {c for c, k in cnt.items() if k < 2}
    keep_train_mask  = np.isin(y_lab, list(rare_classes))
    split_mask       = ~keep_train_mask
    if rare_classes:
        print(f"[WARN] Classes with <2 samples (kept in TRAIN only): {sorted(rare_classes)}", flush=True)

    ids_split   = ids_lab[split_mask]
    paths_split = paths_lab[split_mask]
    y_split     = y_lab[split_mask]

    ids_rare    = ids_lab[keep_train_mask]
    paths_rare  = paths_lab[keep_train_mask]
    y_rare      = y_lab[keep_train_mask]

    # Stratified split
    desired_test = 40
    if isinstance(desired_test, int):
        test_size = min(desired_test, max(1, len(y_split) - len(set(y_split)))) or 0.2
    else:
        test_size = desired_test

    ids_tr_s, ids_te_s, y_tr_s, y_te_s, paths_tr_s, paths_te_s = train_test_split(
        ids_split, y_split, paths_split,
        test_size=test_size, random_state=42, shuffle=True, stratify=y_split
    )

    # Assemble final TRAIN/TEST lists
    ids_train   = np.concatenate([ids_tr_s,   ids_rare,   ids_unlab])
    paths_train = np.concatenate([paths_tr_s, paths_rare, paths_unlab])
    y_train     = np.concatenate([y_tr_s,     y_rare,     np.full(len(ids_unlab), np.nan)])
    ids_test, paths_test, y_test = ids_te_s, paths_te_s, y_te_s

    # --- 3) Prepare streaming build config ---
    CHANNELS = ("raw","zg","zl","log")
    C        = len(CHANNELS)
    SIDE_MM  = 70.0
    TOP_PCT  = 0.01

    # index maps for fast lookups
    id_to_idx   = {sid: i for i, sid in enumerate(ids)}
    paths_all   = np.array(paths)
    y_all       = np.array(y)
    idx_train   = np.array([id_to_idx[sid] for sid in ids_train])
    idx_test    = np.array([id_to_idx[sid] for sid in ids_test])

    # fixed target crop size (XYZ) based on headers across ALL scans
    tgt_xyz   = compute_target_xyz(paths, side_mm=SIDE_MM)
    tgt_shape = (*tgt_xyz, C)
    print("Target crop shape (XYZC):", tgt_shape, flush=True)

    # --- 4) Build TEST (streamed) ---
    X_test = np.zeros((len(idx_test), *tgt_shape), dtype=np.float32)
    with tqdm(total=len(idx_test), desc="TEST  ", unit="scan") as pbar:
        for k, i in enumerate(idx_test):
            p   = paths_all[i]
            box = compute_box_for_path(p, side_mm=SIDE_MM, top_pct=TOP_PCT)
            xc  = build_and_crop_one_from_path(p, box, channels=CHANNELS, low_clip_pct=1.0, local_win=15)
            if xc.shape != tgt_shape:  # center pad/crop to fixed cube
                xc = pad_to_shape(xc, tgt_shape)
            X_test[k] = xc
            pbar.update(1)

    # --- 5) Build TRAIN (streamed + augment per-scan) ---
    def aug_count(lbl):
        if pd.isna(lbl): return 0
        return 3 if int(lbl) == 4 else 8

    n_total = len(idx_train) + sum(aug_count(y_all[i]) for i in idx_train)
    X_train = np.zeros((n_total, *tgt_shape), dtype=np.float32)
    y_train_out = np.empty((n_total,), dtype=float)


    # before TRAIN loop
    ids_all   = np.array(ids)
    paths_all = np.array(paths)
    
    ids_train_out   = np.empty((n_total,), dtype=object)
    paths_train_out = np.empty((n_total,), dtype=object)

    w = 0
    with tqdm(total=n_total, desc="TRAIN ", unit="sample") as pbar:
        for i in idx_train:
            p   = paths_all[i]
            yi  = y_all[i]
            box = compute_box_for_path(p, side_mm=SIDE_MM, top_pct=TOP_PCT)
            x   = build_and_crop_one_from_path(p, box, channels=CHANNELS, low_clip_pct=1.0, local_win=15)
            if x.shape != tgt_shape:
                x = pad_to_shape(x, tgt_shape)
    
            Xa, ya = augment_dataset(x[None, ...], np.array([yi]), seed=42+i)
            nn = Xa.shape[0]
            X_train[w:w+nn]        = Xa
            y_train_out[w:w+nn]    = ya
            ids_train_out[w:w+nn]   = ids_all[i]
            paths_train_out[w:w+nn] = paths_all[i]
            w += nn
            pbar.update(nn)

    # --- 6) Save ---
    print(f"Train: X={X_train.shape}, labeled={np.sum(~pd.isna(y_train_out))}", flush=True)
    print(f"Test : X={X_test.shape},  labeled={np.sum(~pd.isna(y_test))}", flush=True)
    
    save_split(
        "final_data_aug",
        X_train, y_train_out, ids_train_out, paths_train_out,
        X_test,  y_test,      ids_test,     paths_test,
        notes=f"Streamed build (side={SIDE_MM}mm, channels={CHANNELS}, seed=42)."
    )

