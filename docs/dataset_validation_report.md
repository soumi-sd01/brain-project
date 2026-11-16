# BraTS 2021 Dataset Validation Report

## Dataset
- Dataset: BraTS 2021 Task 1
- Patients checked: 1,251
- Modalities: FLAIR, T1, T1ce, T2
- Segmentation masks: 1,251
- Expected NIfTI files: 6,255

## Validation Results
- Missing or empty files: 0
- Unexpected image dimensions: 0
- Spatial affine mismatches: 0
- Observed segmentation labels: 0, 1, 2, 4
- Invalid segmentation masks: 0
- Empty segmentation masks: 0
- MRI volumes containing NaN or Inf: 0

## Expected Image Dimensions
(240, 240, 155)

## Conclusion
All 1,251 patient folders passed the structural,
header alignment, segmentation label, and MRI
finite-value checks performed in this validation.

These checks do not establish clinical annotation
quality or intensity normalization.
