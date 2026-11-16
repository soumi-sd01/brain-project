# MRI Preprocessing Protocol

## Intensity Normalization

We use per-patient, per-modality z-score normalization.

For each MRI modality:

1. Identify nonzero voxels.
2. Calculate the mean and standard deviation over nonzero voxels.
3. Normalize nonzero voxels using:

   x_normalized = (x - mean) / standard_deviation

4. Preserve zero-valued background voxels as zero.

### Constant-Intensity Edge Case

If all nonzero voxels in a volume have the same intensity, the standard deviation is zero and z-score normalization cannot be performed.

In this case:
- Set all nonzero voxels to zero.
- Keep background voxels at zero.
- This prevents unnormalized intensity values from entering the model.

This is a defensive handling rule for constant-intensity volumes.

## Validation

A test on BraTS2021_00000 FLAIR produced:

- Normalized nonzero mean: approximately 0
- Normalized nonzero standard deviation: 1
- Background voxels preserved as zero
- Original MRI volume unchanged

An additional unit test verifies that a constant nonzero volume becomes entirely zero after normalization.

This protocol is selected for the project and will be applied consistently during preprocessing. It is not a claim that this method is universally optimal.
