"""
caad/data/corruption.py
=======================
Image corruption engine.

Design contract (this is what makes spec-replay and the CAAD/GRPO comparison
valid):

  apply_corruption(image, spec) is a PURE FUNCTION.
  All randomness flows from np.random.default_rng(spec.seed).
  Same (image, spec) -> bit-identical output, on any machine.

Corruption families (lighting / camera / occlusion / weather follow the ROVA
paper, Sec 3.1 / Tab. 8; sensor and digital follow ImageNet-C):
  lighting : dusk, night, overexposure, shadow
  camera   : translation, zoom, rotation (one static affine)
  occlusion: static_occlusion
  weather  : fog, rain, snow
  sensor   : gaussian_noise, gaussian_blur, motion_blur
  digital  : jpeg, pixelate

Every corruption preserves (H, W, C) -- required so clean/corrupted images
tokenize to the SAME number of visual tokens (the CAAD teacher scores the
student's rollout on shared prefixes, so prompt lengths must match).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, asdict, field

import cv2
import numpy as np

CORRUPTION_STYLES = [
    "dusk", "night", "overexposure", "shadow",          # lighting
    "translation", "zoom", "rotation",                  # camera
    "static_occlusion",                                 # occlusion
    "fog", "rain", "snow",                              # weather
    "gaussian_noise", "gaussian_blur", "motion_blur",   # sensor / optics
    "jpeg", "pixelate",                                 # digital
]
SEVERITIES = (1, 2, 3, 4, 5)


@dataclass(frozen=True)
class CorruptionSpec:
    image_id: str
    style: str
    severity: int                      # 1..5
    seed: int
    extra: dict = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @staticmethod
    def from_json(s: str) -> "CorruptionSpec":
        return CorruptionSpec(**json.loads(s))


def derive_seed(image_id: str, style: str, severity: int, global_seed: int) -> int:
    """Seed from sample identity: stable across shuffle order / num_workers."""
    key = f"{global_seed}:{image_id}:{style}:{severity}".encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:8], "little")


def make_spec(image_id, style, severity, global_seed):
    return CorruptionSpec(
        image_id=image_id, style=style, severity=int(severity),
        seed=derive_seed(image_id, style, severity, global_seed),
    )


# --------------------------------------------------------------------------- #
# Main entry
# --------------------------------------------------------------------------- #
def apply_corruption(image: np.ndarray, spec: CorruptionSpec) -> np.ndarray:
    """
    image: (H, W, 3) uint8 RGB.  Returns same shape/dtype.
    """
    assert image.ndim == 3 and image.shape[-1] == 3, "expect (H,W,3) RGB"
    if spec.style not in _DISPATCH:
        raise ValueError(f"unknown corruption {spec.style!r}; choose from {CORRUPTION_STYLES}")
    rng = np.random.default_rng(spec.seed)
    sev = spec.severity / 5.0  # normalize 0.2..1.0

    out = _DISPATCH[spec.style](image.astype(np.float32), sev, rng)

    out = np.clip(out, 0, 255).astype(np.uint8)
    assert out.shape == image.shape, (
        f"corruption changed shape {image.shape}->{out.shape}; "
        "this breaks clean/corrupted prompt-length alignment")
    return out


# --------------------------------------------------------------------------- #
# Lighting
# --------------------------------------------------------------------------- #
def _dusk(x, sev, rng):
    gamma = 1.0 + 1.2 * sev
    x = 255.0 * (x / 255.0) ** gamma
    tint = np.array([1.0 + 0.25 * sev, 1.0, 1.0 - 0.25 * sev])   # warm orange
    return x * tint * (0.85 + 0.15 * (1 - sev))


def _night(x, sev, rng):
    x = x * (1.0 - 0.75 * sev)
    x[..., 2] *= 1.0 + 0.15 * sev            # slight blue cast
    noise = rng.normal(0, 6 + 14 * sev, x.shape).astype(np.float32)
    return x + noise                          # sensor noise in the dark


def _overexposure(x, sev, rng):
    x = x * (1.0 + 1.5 * sev) + 40 * sev
    # blown-out bloom around the brightest region
    H, W, _ = x.shape
    cy, cx = rng.integers(H // 4, 3 * H // 4), rng.integers(W // 4, 3 * W // 4)
    yy, xx = np.mgrid[0:H, 0:W]
    r2 = ((yy - cy) ** 2 + (xx - cx) ** 2) / (0.12 * (H * W) * (0.4 + sev))
    bloom = np.exp(-r2)[:, :, None] * 255 * sev
    return x + bloom


def _shadow(x, sev, rng):
    H, W, _ = x.shape
    # smooth random shadow mask
    m = rng.random((max(1, H // 8), max(1, W // 8))).astype(np.float32)
    m = cv2.GaussianBlur(m, (0, 0), 3)
    m = cv2.resize(m, (W, H))
    m = (m > np.quantile(m, 0.6)).astype(np.float32)
    m = cv2.GaussianBlur(m, (0, 0), 15)
    return x * (1.0 - 0.7 * sev * m[..., None])


# --------------------------------------------------------------------------- #
# Camera (one static affine; border replicate keeps shape)
# --------------------------------------------------------------------------- #
def _signed_mag(rng, peak):
    return float(rng.choice([-1.0, 1.0]) * rng.uniform(0.5, 1.0) * peak)


def _warp(x, mat):
    H, W, _ = x.shape
    return cv2.warpAffine(x, mat, (W, H), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)


def _translation(x, sev, rng):
    H, W, _ = x.shape
    dx, dy = _signed_mag(rng, 0.08 * W * sev), _signed_mag(rng, 0.08 * H * sev)
    return _warp(x, np.float32([[1, 0, dx], [0, 1, dy]]))


def _zoom(x, sev, rng):
    H, W, _ = x.shape
    s = 1.0 + abs(_signed_mag(rng, 0.20 * sev))
    return _warp(x, cv2.getRotationMatrix2D((W / 2, H / 2), 0, s))


def _rotation(x, sev, rng):
    H, W, _ = x.shape
    ang = _signed_mag(rng, 12.0 * sev)
    return _warp(x, cv2.getRotationMatrix2D((W / 2, H / 2), ang, 1.0))


# --------------------------------------------------------------------------- #
# Occlusion
# --------------------------------------------------------------------------- #
def _blocks(rng, H, W, sev, n_lo=1, n_hi=4):
    n = rng.integers(n_lo, n_hi + 1)
    rects = []
    for _ in range(n):
        bw = int(W * (0.10 + 0.25 * sev) * (0.5 + rng.random()))
        bh = int(H * (0.10 + 0.25 * sev) * (0.5 + rng.random()))
        x0 = rng.integers(0, max(1, W - bw))
        y0 = rng.integers(0, max(1, H - bh))
        rects.append((x0, y0, bw, bh))
    return rects


def _static_occlusion(x, sev, rng):
    H, W, _ = x.shape
    out = x.copy()
    for (x0, y0, bw, bh) in _blocks(rng, H, W, sev):
        color = rng.integers(0, 60, 3).astype(np.float32)
        out[y0:y0 + bh, x0:x0 + bw] = color
    return out


# --------------------------------------------------------------------------- #
# Weather
# --------------------------------------------------------------------------- #
def _fog(x, sev, rng):
    H, W, _ = x.shape
    fog_color = 235.0
    # spatially-smooth fog density, denser toward the top
    d = rng.random((max(1, H // 8), max(1, W // 8))).astype(np.float32)
    d = cv2.resize(cv2.GaussianBlur(d, (0, 0), 4), (W, H))
    d = d * np.linspace(1.0, 0.55, H)[:, None]
    alpha = np.clip(0.25 + 0.65 * sev * d, 0, 0.95)[..., None]
    return x * (1 - alpha) + fog_color * alpha


def _rain(x, sev, rng):
    H, W, _ = x.shape
    out = x * (1.0 - 0.15 * sev)                       # overcast dimming
    n = max(int(120 * sev * (H * W) / (480 * 640)), 30)
    px = rng.integers(0, W, n)
    py = rng.integers(0, H, n)
    length = 8 + 20 * sev
    for i in range(n):
        cv2.line(out, (int(px[i]), int(py[i])),
                 (int(px[i] - length * 0.3), int(py[i] + length)),
                 (200, 200, 210), 1, cv2.LINE_AA)
    if sev > 0.5:                                      # streak blur in heavy rain
        out = cv2.GaussianBlur(out, (0, 0), 0.6 + sev)
    return out


def _snow(x, sev, rng):
    H, W, _ = x.shape
    out = x * (1.0 - 0.10 * sev) + 15 * sev
    n = max(int(200 * sev * (H * W) / (480 * 640)), 50)
    px = rng.integers(0, W, n)
    py = rng.integers(0, H, n)
    rad = rng.integers(1, 3, n)
    for i in range(n):
        cv2.circle(out, (int(px[i]), int(py[i])), int(rad[i]),
                   (245, 245, 250), -1, cv2.LINE_AA)
    return out


# --------------------------------------------------------------------------- #
# Sensor / optics / digital
# --------------------------------------------------------------------------- #
def _gaussian_noise(x, sev, rng):
    return x + rng.normal(0, 8 + 40 * sev, x.shape).astype(np.float32)


def _gaussian_blur(x, sev, rng):
    return cv2.GaussianBlur(x, (0, 0), 0.5 + 3.0 * sev)


def _motion_blur(x, sev, rng):
    k = int(3 + 20 * sev) | 1                         # odd kernel length
    kernel = np.zeros((k, k), np.float32)
    kernel[k // 2, :] = 1.0
    rot = cv2.getRotationMatrix2D((k / 2 - 0.5, k / 2 - 0.5), float(rng.uniform(0, 180)), 1.0)
    kernel = cv2.warpAffine(kernel, rot, (k, k))
    kernel /= max(kernel.sum(), 1e-6)
    return cv2.filter2D(x, -1, kernel, borderType=cv2.BORDER_REFLECT)


def _jpeg(x, sev, rng):
    quality = int(round(90 - 80 * sev))               # sev 1..5 -> q 74..10
    ok, buf = cv2.imencode(".jpg", np.clip(x, 0, 255).astype(np.uint8),
                           [cv2.IMWRITE_JPEG_QUALITY, quality])
    assert ok, "jpeg encode failed"
    return cv2.imdecode(buf, cv2.IMREAD_UNCHANGED).astype(np.float32)


def _pixelate(x, sev, rng):
    H, W, _ = x.shape
    scale = max(0.08, 1.0 - 0.9 * sev)
    h, w = max(1, int(H * scale)), max(1, int(W * scale))
    small = cv2.resize(x, (w, h), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (W, H), interpolation=cv2.INTER_NEAREST)


_DISPATCH = {
    "dusk": _dusk, "night": _night, "overexposure": _overexposure, "shadow": _shadow,
    "translation": _translation, "zoom": _zoom, "rotation": _rotation,
    "static_occlusion": _static_occlusion,
    "fog": _fog, "rain": _rain, "snow": _snow,
    "gaussian_noise": _gaussian_noise, "gaussian_blur": _gaussian_blur,
    "motion_blur": _motion_blur, "jpeg": _jpeg, "pixelate": _pixelate,
}
assert set(_DISPATCH) == set(CORRUPTION_STYLES)
