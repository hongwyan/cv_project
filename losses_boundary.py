# src/losses_boundary.py  (REPLACE with this version)
import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        probs = probs.view(probs.size(0), -1)
        target = target.view(target.size(0), -1)

        inter = (probs * target).sum(dim=1)
        union = probs.sum(dim=1) + target.sum(dim=1)
        dice = (2.0 * inter + self.eps) / (union + self.eps)
        return 1.0 - dice.mean()


class WeightedBCEWithLogits(nn.Module):
    """
    Per-pixel weighted BCEWithLogits:
      loss = mean( w * BCE(logits, target) )
    """
    def forward(self, logits: torch.Tensor, target: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        return (weight * bce).mean()


class WeightedBCEDiceLoss(nn.Module):
    def __init__(self, w_bce: float = 1.0, w_dice: float = 1.0):
        super().__init__()
        self.wbce = WeightedBCEWithLogits()
        self.dice = DiceLoss()
        self.w_bce = w_bce
        self.w_dice = w_dice

    def forward(self, logits: torch.Tensor, target: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        lb = self.wbce(logits, target, weight)
        ld = self.dice(logits, target)
        return self.w_bce * lb + self.w_dice * ld
