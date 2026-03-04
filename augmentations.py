import math
import random

import torch
import torch.nn.functional as F


class RandomFlipRotate2D:
    """
    Train-only augmentation for paired image/mask tensors.
    - Random horizontal and vertical flips
    - Random rotation in [-max_degrees, max_degrees]
    """

    def __init__(self, max_degrees: float = 10.0, p_flip: float = 0.5):
        self.max_degrees = float(max_degrees)
        self.p_flip = float(p_flip)

    def __call__(self, image: torch.Tensor, mask: torch.Tensor):
        if random.random() < self.p_flip:
            image = torch.flip(image, dims=(-1,))
            mask = torch.flip(mask, dims=(-1,))

        if random.random() < self.p_flip:
            image = torch.flip(image, dims=(-2,))
            mask = torch.flip(mask, dims=(-2,))

        angle = random.uniform(-self.max_degrees, self.max_degrees)
        if abs(angle) < 1e-6:
            return image, mask

        radians = math.radians(angle)
        c = math.cos(radians)
        s = math.sin(radians)
        theta = torch.tensor(
            [[c, -s, 0.0], [s, c, 0.0]],
            dtype=image.dtype,
            device=image.device,
        ).unsqueeze(0)

        img_b = image.unsqueeze(0)
        mask_b = mask.unsqueeze(0)
        grid = F.affine_grid(theta, img_b.size(), align_corners=False)
        image = F.grid_sample(
            img_b,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        ).squeeze(0)
        mask = F.grid_sample(
            mask_b,
            grid,
            mode="nearest",
            padding_mode="zeros",
            align_corners=False,
        ).squeeze(0)
        return image.contiguous(), mask.contiguous()
