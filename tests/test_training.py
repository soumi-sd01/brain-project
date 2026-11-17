import torch

from src.training.losses import DiceCELoss
from src.training.metrics import per_class_dice, region_dice


def test_dice_ce_loss_is_finite():
    torch.manual_seed(42)

    logits = torch.randn(2, 4, 8, 8, 8, requires_grad=True)
    targets = torch.randint(0, 4, (2, 8, 8, 8))

    criterion = DiceCELoss()
    loss = criterion(logits, targets)

    assert loss.ndim == 0
    assert torch.isfinite(loss)

    loss.backward()

    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert logits.grad.abs().sum() > 0


def test_loss_handles_absent_class():
    torch.manual_seed(42)

    logits = torch.randn(1, 4, 2, 2, 2, requires_grad=True)

    targets = torch.tensor([
        [[[0, 0], [1, 1]],
         [[2, 2], [0, 1]]]
    ])

    loss = DiceCELoss()(logits, targets)

    assert torch.isfinite(loss)

    loss.backward()

    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_per_class_dice_hand_calculation():
    targets = torch.tensor([[[[0, 0, 1, 1, 2, 2, 0, 0]]]])
    predictions = torch.tensor([[[[0, 0, 1, 2, 2, 2, 0, 0]]]])

    scores = per_class_dice(predictions, targets)

    assert abs(scores[0] - 1.0) < 1e-4
    assert abs(scores[1] - (2 / 3)) < 1e-4
    assert abs(scores[2] - 0.8) < 1e-4
    assert scores[3] == 1.0


def test_region_dice_hand_calculation():
    targets = torch.tensor([[[[0, 1, 1, 2, 2, 3, 3, 0]]]])
    predictions = torch.tensor([[[[0, 1, 2, 2, 2, 3, 0, 0]]]])

    scores = region_dice(predictions, targets)

    assert abs(scores["WT"] - (10 / 11)) < 1e-4
    assert abs(scores["TC"] - (4 / 6)) < 1e-4
    assert abs(scores["ET"] - (2 / 3)) < 1e-4


def test_absent_region_both_empty():
    targets = torch.tensor([[[[0, 1, 2, 0]]]])
    predictions = torch.tensor([[[[0, 1, 2, 0]]]])

    scores = region_dice(predictions, targets)

    assert scores["ET"] == 1.0


def test_absent_region_false_positive():
    targets = torch.tensor([[[[0, 1, 2, 0]]]])
    predictions = torch.tensor([[[[0, 1, 3, 0]]]])

    scores = region_dice(predictions, targets)

    assert scores["ET"] == 0.0
