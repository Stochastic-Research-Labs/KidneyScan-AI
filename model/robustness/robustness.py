"""KidneyScan AI — robustness corruptions (TEST ONLY, never for training).

What this file does
-------------------
Make a clean CT volume worse on purpose, so we can check how much a
trained model falls apart on bad input. This is Gap 5 in the plan:
models drop on new hospital data (Raman 2025: tumor Dice 0.85 on
KiTS -> 0.66 on a new hospital set).

Corruptions in this file:
  1. gaussian_noise      — add random noise (bad scanner / low dose).
  2. lower_resolution    — downsample, then upsample back (thick slices).
  3. drop_slices         — drop 10% of slices (missing slices).
  4. intensity_shift     — shift all HU values (different scanner setup).

Rules:
  * Test only. Never use these functions to make training data.
    Training augmentation is a different job (see the model folder).
  * Every function takes a volume + a severity and returns a NEW
    volume. The input is never changed.
  * No invented numbers: severities below are test settings, not results.

Reuse — see REUSE table at the bottom of this file and README.md.
The ideas come from nnU-Net v2 augmentation and MONAI transforms.
We did NOT copy their code. We wrote small NumPy/SciPy versions
here, so the test runs with no torch / MONAI install.

Labels: 0 background, 1 kidney, 2 tumor, 3 cyst (KiTS23).
Volume shape: (depth, height, width), values in Hounsfield Units (HU).
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

__all__ = [
    "SEVERITIES",
    "gaussian_noise",
    "lower_resolution",
    "drop_slices",
    "intensity_shift",
    "corrupt",
]

# Severity levels. A float 0-1 is also accepted by every function
# (0 = no change, 1 = strongest test in this file).
# The HU numbers are test settings chosen for KiTS23 CT, not results.
SEVERITIES = {
    # Gaussian noise: sigma in HU.
    "noise_sigma_hu": {"mild": 10.0, "moderate": 25.0, "severe": 50.0},
    # Lower resolution: downsample factor per axis (z, y, x).
    "downsample_factor": {"mild": 1.5, "moderate": 2.0, "severe": 3.0},
    # Drop slices: fraction of slices dropped along z.
    "drop_fraction": {"mild": 0.05, "moderate": 0.10, "severe": 0.20},
    # Intensity shift: added HU.
    "shift_hu": {"mild": 20.0, "moderate": 50.0, "severe": 100.0},
}


def _resolve(value, table, severity):
    """Turn a severity (name or 0-1 float) into a real test value."""
    if isinstance(severity, str):
        if severity not in table:
            raise ValueError(f"Unknown severity {severity!r}. Use {sorted(table)} or a 0-1 float.")
        return float(table[severity])
    severity = float(severity)
    if not 0.0 <= severity <= 1.0:
        raise ValueError("Float severity must be between 0 and 1.")
    # Linear map: 0 -> mild value, 1 -> severe value.
    keys = ["mild", "moderate", "severe"]
    lo, hi = float(table[keys[0]]), float(table[keys[-1]])
    return lo + severity * (hi - lo)


def _check_volume(volume):
    volume = np.asarray(volume)
    if volume.ndim != 3:
        raise ValueError(f"Volume must be 3D (depth, height, width), got shape {volume.shape}.")
    if volume.size == 0:
        raise ValueError("Volume is empty.")
    return volume.astype(np.float32, copy=True)


def gaussian_noise(volume, severity="moderate", seed=42):
    """Add Gaussian noise. Simulates a noisy / low-dose scan.

    volume: 3D CT array in HU. severity: "mild"/"moderate"/"severe"
    or a 0-1 float. seed: fixed so the test can be repeated.
    Returns a new float32 volume, same shape.
    """
    vol = _check_volume(volume)
    sigma = _resolve(SEVERITIES["noise_sigma_hu"], SEVERITIES["noise_sigma_hu"], severity)
    rng = np.random.default_rng(seed)
    return (vol + rng.normal(0.0, sigma, size=vol.shape).astype(np.float32)).astype(np.float32)


def lower_resolution(volume, severity="moderate"):
    """Downsample, then upsample back to the same shape.

    Simulates thick slices / a low-resolution scanner. The shape
    stays the same, so the model gets the same input size, but the
    fine detail is gone. Uses linear interpolation (order=1).
    """
    vol = _check_volume(volume)
    factor = _resolve(SEVERITIES["downsample_factor"], SEVERITIES["downsample_factor"], severity)
    if factor <= 1.0:
        return vol
    small = ndimage.zoom(vol, 1.0 / factor, order=1)
    back = ndimage.zoom(small, np.array(vol.shape) / np.array(small.shape), order=1)
    # zoom can be off by one voxel; crop or pad to the exact shape.
    back = _match_shape(back, vol.shape)
    return back.astype(np.float32)


def drop_slices(volume, severity="moderate", seed=42):
    """Drop a fraction of slices along z, fill each gap with the
    nearest kept slice. Default (moderate) = 10% of slices.

    Simulates missing slices. Shape stays the same, so the model
    input shape does not change. The dropped slice index list is
    evenly spaced (not random), so the test is repeatable; `seed`
    is kept in the signature for a future random mode.
    """
    vol = _check_volume(volume)
    frac = _resolve(SEVERITIES["drop_fraction"], SEVERITIES["drop_fraction"], severity)
    n = vol.shape[0]
    n_drop = int(round(frac * n))
    if n_drop <= 0:
        return vol
    if n_drop >= n:
        raise ValueError("Drop fraction would remove all slices.")
    # Evenly spaced indices, never the first or last slice.
    drop_idx = np.linspace(1, n - 2, n_drop).round().astype(int)
    drop_idx = np.unique(drop_idx)
    keep_idx = np.array([i for i in range(n) if i not in set(drop_idx.tolist())])
    out = vol.copy()
    for i in drop_idx:
        # Nearest kept slice (the one just before it, else just after).
        before = keep_idx[keep_idx < i]
        after = keep_idx[keep_idx > i]
        src = before[-1] if len(before) else after[0]
        out[i] = vol[src]
    return out.astype(np.float32)


def intensity_shift(volume, severity="moderate"):
    """Add a constant HU shift to the whole volume.

    Simulates a different scanner calibration / contrast setting
    (one reason models drop on new hospital data, Raman 2025).
    This is a shift only, not a rescale: tissue contrast is kept.
    """
    vol = _check_volume(volume)
    shift = _resolve(SEVERITIES["shift_hu"], SEVERITIES["shift_hu"], severity)
    return (vol + shift).astype(np.float32)


def corrupt(volume, kind, severity="moderate", seed=42):
    """Apply one corruption by name. Kinds: noise, lowres, dropslices, shift."""
    kinds = {
        "noise": gaussian_noise,
        "lowres": lower_resolution,
        "dropslices": drop_slices,
        "shift": intensity_shift,
    }
    if kind not in kinds:
        raise ValueError(f"Unknown corruption {kind!r}. Use {sorted(kinds)}.")
    if kind == "lowres":
        return kinds[kind](volume, severity=severity)
    if kind == "shift":
        return kinds[kind](volume, severity=severity)
    return kinds[kind](volume, severity=severity, seed=seed)


def _match_shape(arr, shape):
    """Crop or zero-pad an array to an exact shape."""
    out = np.zeros(shape, dtype=arr.dtype)
    slices = tuple(slice(0, min(a, b)) for a, b in zip(arr.shape, shape))
    out[slices] = arr[slices]
    return out


# ---------------------------------------------------------------------------
# REUSE TABLE — record for every reused code / idea (user rule, 15 Nov 2026)
# ---------------------------------------------------------------------------
# | What we reused | Source + link | License | What we changed and why | How to cite |
# | nnU-Net v2 augmentation idea: Gaussian noise, low-res simulation,
# | intensity shift as robustness tests (not their training code) |
# | https://github.com/MIC-DKFZ/nnUNet | Apache 2.0 |
# | Rewrote as small NumPy/SciPy test-only functions, no torch, no
# | nnU-Net install; severities set for KiTS23 CT HU values; drop-
# | slices is our own test (nnU-Net does not test missing slices). |
# | Cite: Isensee et al., Nature Methods 2021, DOI 10.1038/s41592-020-01008-z |
# | MONAI transforms idea: intensity shift / noise as transforms |
# | https://github.com/Project-MONAI/MONAI | Apache 2.0 |
# | Rewrote as plain functions on NumPy arrays, not MONAI
# | MapTransform classes, so this file has no MONAI dependency. |
# | Cite: MONAI Consortium, https://monai.io (Apache 2.0) |
# | Test design (which corruptions, why): Raman 2025 external drop
# | 0.85 -> 0.66; Scalco 2024 cyst Dice 0.21; BackSplit cyst 0.18 |
# | Raman DOI 10.1016/j.ejrai.2025.100035; Scalco DOI 10.1002/mp.17189;
# | BackSplit arXiv 2511.19394. See docs/literature-review.md. |
# | No code reused from these papers (none was published with them);
# | only the evidence that cyst, small tumors and new-hospital data
# | are the weak points, so the test must score them apart. |
