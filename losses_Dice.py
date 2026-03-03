# src/losses.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    Soft Dice loss for binary segmentation.
    Input: logits (B,1,H,W)
    Target: y in {0,1}, shape (B,1,H,W)
    """
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)

        probs = probs.view(probs.size(0), -1)
        target = target.view(target.size(0), -1)

        intersection = (probs * target).sum(dim=1)
        union = probs.sum(dim=1) + target.sum(dim=1)

        dice = (2.0 * intersection + self.eps) / (union + self.eps)
        return 1.0 - dice.mean()


class BCEDiceLoss(nn.Module):
    """
    BCEWithLogits + Dice
    """
    def __init__(self, bce_weight: float = 1.0, dice_weight: float = 1.0):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        self.bce_w = bce_weight
        self.dice_w = dice_weight

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss_bce = self.bce(logits, target)
        loss_dice = self.dice(logits, target)
        return self.bce_w * loss_bce + self.dice_w * loss_dice


@torch.no_grad()
def dice_score(logits: torch.Tensor, target: torch.Tensor, thresh: float = 0.5) -> torch.Tensor:
    """
    Dice score for evaluation.
    """
    probs = torch.sigmoid(logits)
    preds = (probs > thresh).float()

    preds = preds.view(preds.size(0), -1)
    target = target.view(target.size(0), -1)

    intersection = (preds * target).sum(dim=1)
    union = preds.sum(dim=1) + target.sum(dim=1)

    dice = (2.0 * intersection) / (union + 1e-6)
    return dice.mean()
