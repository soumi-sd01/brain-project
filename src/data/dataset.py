
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset
from pathlib import Path


class BraTSDataset(Dataset):
    """BraTS 2021 Dataset for four-class brain tumour segmentation."""

    MODALITIES = ["flair", "t1", "t1ce", "t2"]

    LABEL_MAP = {
        0: 0,
        1: 1,
        2: 2,
        4: 3
    }

    def __init__(
        self,
        data_dir,
        split_file,
        normalize=True,
        missing_modalities=None
    ):
        self.data_dir = Path(data_dir)
        self.normalize = normalize

        self.missing_modalities = set(
            missing_modalities or []
        )

        invalid = self.missing_modalities - set(self.MODALITIES)

        if invalid:
            raise ValueError(
                f"Invalid modality names: {sorted(invalid)}"
            )

        if len(self.missing_modalities) == len(self.MODALITIES):
            raise ValueError("At least one modality must be available.")

        with open(split_file, "r") as f:
            self.patient_ids = [
                line.strip() for line in f if line.strip()
            ]

    def __len__(self):
        return len(self.patient_ids)

    def _load_volume(self, path):
        return np.asarray(
            nib.load(str(path)).dataobj,
            dtype=np.float32
        )

    def _normalize(self, volume):
        mask = volume != 0

        if not np.any(mask):
            return volume

        mean = volume[mask].mean()
        std = volume[mask].std()

        if std > 0:
            volume[mask] = (volume[mask] - mean) / std
        else:
            volume[mask] = 0.0

        return volume

    def __getitem__(self, index):
        patient_id = self.patient_ids[index]
        patient_dir = self.data_dir / patient_id

        loaded_volumes = {}
        reference_shape = None

        for modality in self.MODALITIES:
            if modality in self.missing_modalities:
                continue

            path = patient_dir / f"{patient_id}_{modality}.nii.gz"

            if not path.is_file():
                raise FileNotFoundError(
                    f"Missing modality file: {path}"
                )

            volume = self._load_volume(path)

            if reference_shape is None:
                reference_shape = volume.shape
            elif volume.shape != reference_shape:
                raise ValueError(
                    f"Shape mismatch for {patient_id}, {modality}"
                )

            if self.normalize:
                volume = self._normalize(volume)

            loaded_volumes[modality] = volume

        images = []
        modality_mask = []

        for modality in self.MODALITIES:
            if modality in self.missing_modalities:
                images.append(
                    np.zeros(reference_shape, dtype=np.float32)
                )
                modality_mask.append(False)
            else:
                images.append(loaded_volumes[modality])
                modality_mask.append(True)

        seg_path = patient_dir / f"{patient_id}_seg.nii.gz"

        if not seg_path.is_file():
            raise FileNotFoundError(
                f"Missing segmentation file: {seg_path}"
            )

        segmentation = np.asarray(
            nib.load(str(seg_path)).dataobj
        )

        if segmentation.shape != reference_shape:
            raise ValueError(
                f"Segmentation shape mismatch for {patient_id}"
            )

        unique_labels = np.unique(segmentation)

        if not set(unique_labels.tolist()).issubset(self.LABEL_MAP):
            raise ValueError(
                f"Unexpected segmentation labels: {unique_labels}"
            )

        mapped_segmentation = np.zeros(
            segmentation.shape, dtype=np.int64
        )

        for original, mapped in self.LABEL_MAP.items():
            mapped_segmentation[segmentation == original] = mapped

        image_tensor = torch.from_numpy(
            np.stack(images, axis=0).copy()
        )

        mask_tensor = torch.from_numpy(
            mapped_segmentation.copy()
        )

        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "modality_mask": torch.tensor(
                modality_mask, dtype=torch.bool
            ),
            "patient_id": patient_id
        }
