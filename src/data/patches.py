
import numpy as np


def extract_patch(
    image,
    mask,
    patch_size=(64, 64, 64),
    strategy="tumor",
    rng=None
):
    """
    Extract a 3D MRI patch and its corresponding segmentation mask.

    image: NumPy array with shape (4, D, H, W)
    mask: NumPy array with shape (D, H, W)
    strategy: "tumor" or "random"
    """

    if rng is None:
        rng = np.random.default_rng()

    if image.ndim != 4 or image.shape[0] != 4:
        raise ValueError("Image must have shape (4, D, H, W).")

    if mask.ndim != 3 or image.shape[1:] != mask.shape:
        raise ValueError("Image and mask spatial dimensions must match.")

    if len(patch_size) != 3:
        raise ValueError("patch_size must contain three dimensions.")

    if any(p <= 0 for p in patch_size):
        raise ValueError("Patch dimensions must be positive.")

    if any(p > s for p, s in zip(patch_size, mask.shape)):
        raise ValueError("Patch cannot be larger than the input volume.")

    if strategy not in ("tumor", "random"):
        raise ValueError("strategy must be 'tumor' or 'random'.")

    if strategy == "tumor":
        tumor_voxels = np.argwhere(mask > 0)

        if len(tumor_voxels) == 0:
            raise ValueError("Cannot extract a tumor patch: mask has no tumor voxels.")

        center = tumor_voxels[rng.integers(len(tumor_voxels))]
        starts = [
            max(0, min(int(c) - p // 2, s - p))
            for c, p, s in zip(center, patch_size, mask.shape)
        ]

    else:
        starts = [
            int(rng.integers(0, s - p + 1))
            for p, s in zip(patch_size, mask.shape)
        ]

    slices = tuple(
        slice(start, start + size)
        for start, size in zip(starts, patch_size)
    )

    image_patch = image[(slice(None),) + slices]
    mask_patch = mask[slices]

    return image_patch.copy(), mask_patch.copy()
