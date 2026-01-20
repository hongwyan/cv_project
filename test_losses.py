# src/test_losses.py
import torch
from losses import DiceLoss, BCEDiceLoss, dice_score

B, H, W = 4, 240, 240
logits = torch.randn(B, 1, H, W)
target = torch.randint(0, 2, (B, 1, H, W)).float()

dice_loss = DiceLoss()
combo_loss = BCEDiceLoss()

print("Dice loss:", dice_loss(logits, target).item())
print("BCE+Dice loss:", combo_loss(logits, target).item())
print("Dice score:", dice_score(logits, target).item())
