"""ImageNet-C wrapper (TextVQA-C / GQA-C): all 19 corruptions run on current
NumPy / scikit-image, keep shape, and are deterministic. Skipped without
imagecorruptions."""
import numpy as np
import pytest

pytest.importorskip("imagecorruptions")

from caad.data.imagenet_c import CATEGORIES, IMAGENET_C, corrupt_image  # noqa: E402


def test_nineteen_in_four_categories():
    assert len(IMAGENET_C) == 19 == len(set(IMAGENET_C))
    assert [len(v) for v in CATEGORIES.values()] == [4, 5, 5, 5]


@pytest.mark.parametrize("name", IMAGENET_C)
def test_runs_deterministic_and_keeps_shape(name):
    img = np.random.default_rng(0).integers(0, 256, (48, 64, 3), dtype=np.uint8)
    a = corrupt_image(img, name, 3, image_id="x")
    b = corrupt_image(img, name, 3, image_id="x")
    assert a.shape == img.shape and a.dtype == np.uint8
    assert np.array_equal(a, b), f"{name} not deterministic"


def test_global_rng_restored():
    np.random.seed(123); expected = np.random.rand()
    np.random.seed(123)
    corrupt_image(np.zeros((32, 32, 3), np.uint8), "gaussian_noise", 2)
    assert np.random.rand() == expected
