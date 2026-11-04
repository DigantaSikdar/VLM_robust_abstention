"""
caad/data/imagenet_c.py
=======================
The 19 ImageNet-C corruptions x 5 severities used by Usama et al. (2025),
"Analysing the Robustness of Vision-Language-Models to Common Corruptions", to
build TextVQA-C and GQA-C. Evaluation uses these (not caad.data.corruption) so
numbers are comparable with that paper.

Backed by the reference implementation, the ``imagecorruptions`` package
(Michaelis et al.), with two compatibility shims for current libraries:
  - NumPy 2 removed ``np.float_`` (used by fog's plasma fractal)
  - scikit-image >= 0.21 removed ``gaussian(..., multichannel=True)``
    (used by glass_blur / gaussian_blur)

The package draws from NumPy's *global* RNG (and impulse_noise from
scikit-image's own); ``corrupt_image`` seeds both from the sample identity so a
corrupted image reproduces bit-for-bit.
"""

from __future__ import annotations

import numpy as np

from .corruption import derive_seed

# Paper Sec. 3 grouping (Figures 4-7).
CATEGORIES = {
    "noise":   ["gaussian_noise", "shot_noise", "impulse_noise", "speckle_noise"],
    "blur":    ["defocus_blur", "glass_blur", "motion_blur", "zoom_blur", "gaussian_blur"],
    "weather": ["snow", "frost", "fog", "brightness", "spatter"],
    "digital": ["contrast", "elastic_transform", "pixelate", "jpeg_compression", "saturate"],
}
IMAGENET_C = [c for cs in CATEGORIES.values() for c in cs]
SEVERITIES = (1, 2, 3, 4, 5)

_patched = False


def _load():
    global _patched
    import imagecorruptions
    import imagecorruptions.corruptions as icc
    if not _patched:
        if not hasattr(np, "float_"):
            np.float_ = np.float64
        _gaussian = icc.gaussian

        def gaussian(image, sigma=1, multichannel=None, **kw):
            if multichannel is not None:
                try:
                    return _gaussian(image, sigma=sigma, multichannel=multichannel, **kw)
                except TypeError:
                    kw["channel_axis"] = -1 if multichannel else None
            return _gaussian(image, sigma=sigma, **kw)

        icc.gaussian = gaussian
        _patched = True
    return imagecorruptions


def corrupt_image(image: np.ndarray, name: str, severity: int, *,
                  image_id: str = "", global_seed: int = 0) -> np.ndarray:
    """(H, W, 3) uint8 RGB -> same shape/dtype, corrupted with ``name`` at
    ``severity`` (1..5). Deterministic in (image_id, name, severity, global_seed)."""
    if name not in IMAGENET_C:
        raise ValueError(f"unknown corruption {name!r}; choose from {IMAGENET_C}")
    if severity not in SEVERITIES:
        raise ValueError(f"severity must be 1..5, got {severity}")
    assert image.ndim == 3 and image.shape[-1] == 3, "expect (H,W,3) RGB"
    ic = _load()
    import skimage.util as sku
    random_noise = sku.random_noise

    def seeded_random_noise(*a, **kw):     # impulse_noise: skimage has its own RNG
        if "rng" not in kw and "seed" not in kw:
            try:
                return random_noise(*a, rng=int(np.random.randint(2 ** 31)), **kw)
            except TypeError:                  # skimage < 0.21 spells it `seed`
                return random_noise(*a, seed=int(np.random.randint(2 ** 31)), **kw)
        return random_noise(*a, **kw)

    state = np.random.get_state()
    np.random.seed(derive_seed(image_id, name, severity, global_seed) % (2 ** 32))
    sku.random_noise = seeded_random_noise
    try:
        out = ic.corrupt(image, corruption_name=name, severity=severity)
    finally:
        sku.random_noise = random_noise
        np.random.set_state(state)
    out = np.asarray(out)
    if out.dtype != np.uint8:
        out = np.clip(out, 0, 255).astype(np.uint8)
    assert out.shape == image.shape, f"{name} changed shape {image.shape}->{out.shape}"
    return out
