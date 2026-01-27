# src/losses_combo_signed.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        p = torch.sigmoid(logits)
        p = p.view(p.size(0), -1)
        t = target.view(target.size(0), -1)

        inter = (p * t).sum(dim=1)
        union = p.sum(dim=1) + t.sum(dim=1)
        dice = (2.0 * inter + self.eps) / (union + self.eps)
        return 1.0 - dice.mean()


class WeightedBCEWithLogits(nn.Module):
    def forward(self, logits: torch.Tensor, target: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        return (w * bce).mean()


class SignedBoundaryLoss(nn.Module):
    """
    Classic boundary loss: mean( p * phi )
    phi is signed distance map from GT (outside positive, inside negative).
    IMPORTANT: phi should be normalized/clipped for stability.
    """
    def forward(self, logits: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
        p = torch.sigmoid(logits)
        return (p * phi).mean()


class WeightedBCEDiceSignedBoundaryLoss(nn.Module):
    def __init__(self, w_bce=1.0, w_dice=1.0, w_sboundary=0.001):
        super().__init__()
        self.wbce = WeightedBCEWithLogits()
        self.dice = DiceLoss()
        self.sbd = SignedBoundaryLoss()
        self.w_bce = w_bce
        self.w_dice = w_dice
        self.w_sbd = w_sboundary

    def forward(self, logits, target, w_map, phi):
        lb = self.wbce(logits, target, w_map)
        has_tumor = target.sum(dim=(1, 2, 3)) > 0
        if has_tumor.any():
            logits_pos = logits[has_tumor]
            target_pos = target[has_tumor]
            phi_pos = phi[has_tumor]
            ld = self.dice(logits_pos, target_pos)
            ls = self.sbd(logits_pos, phi_pos)
        else:
            ld = logits.new_tensor(0.0)
            ls = logits.new_tensor(0.0)
        return self.w_bce * lb + self.w_dice * ld + self.w_sbd * ls
