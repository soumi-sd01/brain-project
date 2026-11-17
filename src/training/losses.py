"""Loss functions for 3D brain tumour segmentation."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceCELoss(nn.Module):
    """Combined multiclass Dice and Cross-Entropy loss."""

    def __init__(self, dice_weight=0.5, ce_weight=0.5, smooth=1e-5):
        super().__init__()

        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.smooth = smooth

    def forward(self, logits, targets):
        """
        Args:
            logits: Predictions (B, C, D, H, W).
            targets: Ground-truth labels (B, D, H, W).

        Returns:
            Combined scalar loss.
        """

        num_classes = logits.shape[1]

        # Cross-Entropy
        ce_loss = F.cross_entropy(logits, targets.long())

        # Convert logits to probabilities
        probabilities = torch.softmax(logits, dim=1)

        # One-hot encode ground truth
        targets_one_hot = F.one_hot(
            targets.long(),
            num_classes=num_classes
        ).permute(0, 4, 1, 2, 3).float()

        # Exclude background
        probabilities = probabilities[:, 1:]
        targets_one_hot = targets_one_hot[:, 1:]

        # Spatial and batch dimensions
        dims = (0, 2, 3, 4)

        intersection = torch.sum(
            probabilities * targets_one_hot,
            dim=dims
        )

        denominator = torch.sum(
            probabilities + targets_one_hot,
            dim=dims
        )

        dice_score = (
            2.0 * intersection + self.smooth
        ) / (
            denominator + self.smooth
        )

        # Include only classes present in ground truth
        target_volume = torch.sum(targets_one_hot, dim=dims)
        present_classes = target_volume > 0

        if present_classes.any():
            dice_loss = 1.0 - dice_score[present_classes].mean()
        else:
            dice_loss = logits.sum() * 0.0

        # Combined loss
        total_loss = (
            self.dice_weight * dice_loss
            + self.ce_weight * ce_loss
        )

        return total_loss
