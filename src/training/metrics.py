"""Evaluation metrics for 3D brain tumour segmentation."""

import torch


def per_class_dice(predictions, targets, num_classes=4, smooth=1e-5):
    """
    Calculate Dice score separately for each class.

    Args:
        predictions: Predicted labels (B, D, H, W).
        targets: Ground-truth labels (B, D, H, W).
        num_classes: Number of segmentation classes.
        smooth: Small constant for numerical stability.

    Returns:
        Dictionary containing Dice score for each class.
    """

    dice_scores = {}

    for class_id in range(num_classes):

        pred_mask = (predictions == class_id)
        target_mask = (targets == class_id)

        intersection = (pred_mask & target_mask).sum().float()

        pred_volume = pred_mask.sum().float()
        target_volume = target_mask.sum().float()

        denominator = pred_volume + target_volume

        if denominator == 0:
            dice = torch.tensor(1.0, device=predictions.device)
        else:
            dice = (
                2.0 * intersection + smooth
            ) / (
                denominator + smooth
            )

        dice_scores[class_id] = dice.item()

    return dice_scores



def region_dice(predictions, targets, smooth=1e-5):
    """
    Calculate Dice for BraTS tumour regions.

    WT = labels 1, 2, 3
    TC = labels 1, 3
    ET = label 3

    Args:
        predictions: Predicted labels (B, D, H, W).
        targets: Ground-truth labels (B, D, H, W).

    Returns:
        Dictionary containing WT, TC and ET Dice scores.
    """

    regions = {
        "WT": [1, 2, 3],
        "TC": [1, 3],
        "ET": [3]
    }

    scores = {}

    for region_name, labels in regions.items():

        pred_mask = torch.zeros_like(
            predictions,
            dtype=torch.bool
        )

        target_mask = torch.zeros_like(
            targets,
            dtype=torch.bool
        )

        for label in labels:
            pred_mask |= (predictions == label)
            target_mask |= (targets == label)

        intersection = (pred_mask & target_mask).sum().float()

        pred_volume = pred_mask.sum().float()
        target_volume = target_mask.sum().float()

        denominator = pred_volume + target_volume

        if pred_volume == 0 and target_volume == 0:
            dice = torch.tensor(1.0, device=predictions.device)

        elif pred_volume == 0 or target_volume == 0:
            dice = torch.tensor(0.0, device=predictions.device)

        else:
            dice = (
                2.0 * intersection + smooth
            ) / (
                denominator + smooth
            )

        scores[region_name] = dice.item()

    return scores
