import torch
from torch import nn


EPSILON = 1e-7


class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0) -> None:
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probabilities = torch.sigmoid(logits)
        probabilities = probabilities.view(probabilities.size(0), -1)
        targets = targets.view(targets.size(0), -1)

        intersection = (probabilities * targets).sum(dim=1)
        union = probabilities.sum(dim=1) + targets.sum(dim=1)
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()


def dice_coefficient(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
) -> float:
    predictions = _binarize_logits(logits, threshold)
    targets = targets.float()
    intersection = (predictions * targets).sum()
    union = predictions.sum() + targets.sum()
    dice = (2.0 * intersection + EPSILON) / (union + EPSILON)
    return dice.item()


def iou_score(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
) -> float:
    predictions = _binarize_logits(logits, threshold)
    targets = targets.float()
    intersection = (predictions * targets).sum()
    union = predictions.sum() + targets.sum() - intersection
    iou = (intersection + EPSILON) / (union + EPSILON)
    return iou.item()


def pixel_accuracy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
) -> float:
    predictions = _binarize_logits(logits, threshold)
    targets = targets.float()
    correct = (predictions == targets).float().sum()
    total = targets.numel()
    return (correct / total).item()


def _binarize_logits(logits: torch.Tensor, threshold: float) -> torch.Tensor:
    return (torch.sigmoid(logits) >= threshold).float()
