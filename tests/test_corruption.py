"""Corruption engine: determinism + the shape-preservation invariant.
Needs OpenCV (skipped where cv2 is unavailable)."""
import numpy as np
import pytest

pytest.importorskip("cv2")

from caad.data.corruption import (CORRUPTION_STYLES, apply_corruption,  # noqa: E402
                                  derive_seed, make_spec)


def _img(seed=0, H=64, W=80):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (H, W, 3), dtype=np.uint8)


@pytest.mark.parametrize("style", CORRUPTION_STYLES)
def test_shape_preserved_and_changed(style):
    img = _img()
    out = apply_corruption(img, make_spec("img0", style, 3, global_seed=0))
    assert out.shape == img.shape and out.dtype == np.uint8
    assert not np.array_equal(out, img), f"{style} left the image unchanged"


@pytest.mark.parametrize("style", CORRUPTION_STYLES)
def test_bit_identical_for_same_spec(style):
    img = _img()
    spec = make_spec("img0", style, 3, global_seed=0)
    assert np.array_equal(apply_corruption(img, spec), apply_corruption(img, spec)), \
        f"{style} not deterministic"


def test_seed_depends_on_identity_not_order():
    # derive_seed must be stable across shuffle order / workers
    s1 = derive_seed("imgA", "fog", 3, 42)
    s2 = derive_seed("imgA", "fog", 3, 42)
    s3 = derive_seed("imgB", "fog", 3, 42)
    assert s1 == s2 and s1 != s3


def test_different_severity_changes_output():
    img = _img()
    lo = apply_corruption(img, make_spec("i", "fog", 1, 0))
    hi = apply_corruption(img, make_spec("i", "fog", 5, 0))
    assert not np.array_equal(lo, hi)


def test_unknown_style_rejected():
    with pytest.raises(ValueError):
        apply_corruption(_img(), make_spec("i", "dynamic_occlusion", 3, 0))
