
import numpy as np
import pytest

from src.data.patches import extract_patch


@pytest.fixture
def synthetic_volume():
    rng = np.random.default_rng(42)

    image = rng.normal(size=(4, 32, 32, 32)).astype(np.float32)
    mask = np.zeros((32, 32, 32), dtype=np.int64)

    mask[12:18, 12:18, 12:18] = 1

    return image, mask


def test_tumor_patch_shape_and_tumor_presence(synthetic_volume):
    image, mask = synthetic_volume

    image_patch, mask_patch = extract_patch(
        image,
        mask,
        patch_size=(16, 16, 16),
        strategy="tumor",
        rng=np.random.default_rng(42)
    )

    assert image_patch.shape == (4, 16, 16, 16)
    assert mask_patch.shape == (16, 16, 16)
    assert np.any(mask_patch > 0)


def test_random_patch_shape(synthetic_volume):
    image, mask = synthetic_volume

    image_patch, mask_patch = extract_patch(
        image,
        mask,
        patch_size=(16, 16, 16),
        strategy="random",
        rng=np.random.default_rng(42)
    )

    assert image_patch.shape == (4, 16, 16, 16)
    assert mask_patch.shape == (16, 16, 16)


def test_invalid_strategy_raises_error(synthetic_volume):
    image, mask = synthetic_volume

    with pytest.raises(ValueError, match="strategy"):
        extract_patch(
            image,
            mask,
            patch_size=(16, 16, 16),
            strategy="invalid"
        )


def test_patch_larger_than_volume_raises_error(synthetic_volume):
    image, mask = synthetic_volume

    with pytest.raises(ValueError, match="larger"):
        extract_patch(
            image,
            mask,
            patch_size=(64, 64, 64),
            strategy="random"
        )


def test_image_and_mask_spatial_alignment(synthetic_volume):
    image, mask = synthetic_volume

    image_patch, mask_patch = extract_patch(
        image,
        mask,
        patch_size=(16, 16, 16),
        strategy="tumor",
        rng=np.random.default_rng(42)
    )

    assert image_patch.shape[1:] == mask_patch.shape


def test_tumor_strategy_rejects_empty_mask(synthetic_volume):
    image, _ = synthetic_volume
    empty_mask = np.zeros((32, 32, 32), dtype=np.int64)

    with pytest.raises(ValueError, match="no tumor voxels"):
        extract_patch(
            image,
            empty_mask,
            patch_size=(16, 16, 16),
            strategy="tumor"
        )
