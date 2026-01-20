# src/unet.py
# Minimal U-Net for binary segmentation (channels-first).
# Input:  (B, in_ch, H, W)
# Output: (B, out_ch, H, W) logits (no sigmoid)

from __future__ import annotations
import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UNet(nn.Module):
    def __init__(self, in_ch: int = 3, out_ch: int = 1, base_ch: int = 32):
        super().__init__()

        c1, c2, c3, c4, c5 = base_ch, base_ch * 2, base_ch * 4, base_ch * 8, base_ch * 16

        self.enc1 = DoubleConv(in_ch, c1)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = DoubleConv(c1, c2)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = DoubleConv(c2, c3)
        self.pool3 = nn.MaxPool2d(2)

        self.enc4 = DoubleConv(c3, c4)
        self.pool4 = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(c4, c5)

        self.up4 = nn.ConvTranspose2d(c5, c4, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(c4 + c4, c4)

        self.up3 = nn.ConvTranspose2d(c4, c3, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(c3 + c3, c3)

        self.up2 = nn.ConvTranspose2d(c3, c2, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(c2 + c2, c2)

        self.up1 = nn.ConvTranspose2d(c2, c1, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(c1 + c1, c1)

        self.out = nn.Conv2d(c1, out_ch, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.enc1(x)      # (B,c1,H,W)
        x2 = self.enc2(self.pool1(x1))  # (B,c2,H/2,W/2)
        x3 = self.enc3(self.pool2(x2))  # (B,c3,H/4,W/4)
        x4 = self.enc4(self.pool3(x3))  # (B,c4,H/8,W/8)
        x5 = self.bottleneck(self.pool4(x4))  # (B,c5,H/16,W/16)

        u4 = self.up4(x5)                 # (B,c4,H/8,W/8)
        d4 = self.dec4(torch.cat([u4, x4], dim=1))

        u3 = self.up3(d4)                 # (B,c3,H/4,W/4)
        d3 = self.dec3(torch.cat([u3, x3], dim=1))

        u2 = self.up2(d3)                 # (B,c2,H/2,W/2)
        d2 = self.dec2(torch.cat([u2, x2], dim=1))

        u1 = self.up1(d2)                 # (B,c1,H,W)
        d1 = self.dec1(torch.cat([u1, x1], dim=1))

        return self.out(d1)               # logits


if __name__ == "__main__":
    x = torch.randn(4, 3, 240, 240)
    m = UNet(in_ch=3, out_ch=1, base_ch=32)
    y = m(x)
    print(y.shape)
