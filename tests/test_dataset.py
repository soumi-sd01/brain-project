
import numpy as np
import nibabel as nib
import pytest
import torch

from src.data.dataset import BraTSDataset


@pytest.fixture
def sample_dataset(tmp_path):
    data_dir = tmp_path / "data"
    patient_id = "BraTS2021_TEST001"
    patient_dir = data_dir / patient_id
    patient_dir.mkdir(parents=True)

    rng = np.random.default_rng(42)

    for modality in BraTSDataset.MODALITIES:
        volume = rng.uniform(10, 100, size=(8, 8, 8)).astype(np.float32)
        volume[:2, :, :] = 0

        image = nib.Nifti1Image(volume, np.eye(4))
        nib.save(image, patient_dir / f"{patient_id}_{modality}.nii.gz")

    segmentation = np.zeros((8, 8, 8), dtype=np.uint8)
    segmentation[2:4, :, :] = 1
    segmentation[4:6, :, :] = 2
    segmentation[6:8, :, :] = 4

    seg_image = nib.Nifti1Image(segmentation, np.eye(4))
    nib.save(seg_image, patient_dir / f"{patient_id}_seg.nii.gz")

    split_file = tmp_path / "train.txt"
    split_file.write_text(patient_id + "\n")

    return data_dir, split_file, patient_dir


def test_output_shapes_and_types(sample_dataset):
    data_dir, split_file, _ = sample_dataset
    dataset = BraTSDataset(data_dir, split_file)

    sample = dataset[0]

    assert sample["image"].shape == (4, 8, 8, 8)
    assert sample["mask"].shape == (8, 8, 8)
    assert sample["image"].dtype == torch.float32
    assert sample["mask"].dtype == torch.int64


def test_segmentation_label_mapping(sample_dataset):
    data_dir, split_file, _ = sample_dataset
    dataset = BraTSDataset(data_dir, split_file)

    sample = dataset[0]

    assert set(torch.unique(sample["mask"]).tolist()) == {0, 1, 2, 3}


def test_normalization_and_background(sample_dataset):
    data_dir, split_file, _ = sample_dataset
    dataset = BraTSDataset(data_dir, split_file)

    sample = dataset[0]

    for channel in sample["image"]:
        nonzero = channel[channel != 0]

        assert torch.isfinite(channel).all()
        assert torch.allclose(
            nonzero.mean(),
            torch.tensor(0.0),
            atol=1e-5
        )
        assert torch.allclose(
            nonzero.std(unbiased=False),
            torch.tensor(1.0),
            atol=1e-5
        )

        assert torch.all(channel[:2, :, :] == 0)


def test_missing_modality_raises_error(sample_dataset):
    data_dir, split_file, patient_dir = sample_dataset

    missing_file = patient_dir / "BraTS2021_TEST001_t1ce.nii.gz"
    missing_file.unlink()

    dataset = BraTSDataset(data_dir, split_file)

    with pytest.raises(FileNotFoundError):
        dataset[0]


def test_invalid_segmentation_labels_raise_error(sample_dataset):
    data_dir, split_file, patient_dir = sample_dataset

    invalid_segmentation = np.zeros((8, 8, 8), dtype=np.uint8)
    invalid_segmentation[3, 3, 3] = 5

    image = nib.Nifti1Image(invalid_segmentation, np.eye(4))
    nib.save(
        image,
        patient_dir / "BraTS2021_TEST001_seg.nii.gz"
    )

    dataset = BraTSDataset(data_dir, split_file)

    with pytest.raises(ValueError):
        dataset[0]



def test_constant_intensity_volume(sample_dataset):
    data_dir, split_file, patient_dir = sample_dataset
    patient_id = "BraTS2021_TEST001"

    constant_volume = np.full((8, 8, 8), 7.0, dtype=np.float32)
    constant_volume[:2, :, :] = 0

    image = nib.Nifti1Image(constant_volume, np.eye(4))
    nib.save(
        image,
        patient_dir / f"{patient_id}_flair.nii.gz"
    )

    dataset = BraTSDataset(data_dir, split_file)
    sample = dataset[0]

    flair = sample["image"][0]

    assert torch.isfinite(flair).all()
    assert torch.all(flair == 0)
