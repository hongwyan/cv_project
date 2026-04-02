from __future__ import annotations

import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover
    matplotlib = None
    plt = None

from augmentations import RandomFlipRotate2D
from boundary_utils_signed import normalize_clip_phi, signed_distance_map_2d
from brats2d_dataset import BraTS2D
from brats2p5d_dataset import BraTS2p5D
from losses_Dice import BCEDiceLoss, DiceOnlyLoss, dice_score
from losses_boundary_combo import BCEDiceSignedBoundaryLoss
from metrics_boundary import hd95_2d
from splits import build_patient_split
from unet import UNet


DEFAULT_ROOT = "../data/BraTS2021_Training_Data"
DEFAULT_MAX_PATIENTS = 100
DEFAULT_RESULTS_ROOT = "result_new"
EXPERIMENT_ORDER = [
    "2d_dice",
    "2d_bce_dice",
    "2d_bce_dice_posweighted",
    "2d_bce_dice_boundary_0.05",
    "2d_bce_dice_boundary_0.1",
    "2d_bce_dice_boundary_0.2",
    "2d_bce_dice_posweighted_boundary_0.05",
    "2d_bce_dice_posweighted_boundary_0.1",
    "2d_bce_dice_posweighted_boundary_0.2",
    "25d_dice",
    "25d_bce_dice",
    "25d_bce_dice_posweighted",
    "25d_bce_dice_boundary_0.01",
    "25d_bce_dice_boundary_0.05",
    "25d_bce_dice_boundary_0.1",
    "25d_bce_dice_boundary_0.2",
    "25d_bce_dice_posweighted_boundary_0.05",
    "25d_bce_dice_posweighted_boundary_0.1",
    "25d_bce_dice_posweighted_boundary_0.2",
]
EXPERIMENT_SPECS = {
    "2d_dice": {"input_mode": "2d", "loss_name": "dice"},
    "2d_bce_dice": {"input_mode": "2d", "loss_name": "bce_dice", "use_pos_weight": False},
    "2d_bce_dice_posweighted": {"input_mode": "2d", "loss_name": "bce_dice", "use_pos_weight": True},
    "2d_bce_dice_boundary_0.05": {"input_mode": "2d", "loss_name": "bce_dice_boundary", "use_pos_weight": False, "w_sboundary": 0.05},
    "2d_bce_dice_boundary_0.1": {"input_mode": "2d", "loss_name": "bce_dice_boundary", "use_pos_weight": False, "w_sboundary": 0.1},
    "2d_bce_dice_boundary_0.2": {"input_mode": "2d", "loss_name": "bce_dice_boundary", "use_pos_weight": False, "w_sboundary": 0.2},
    "2d_bce_dice_posweighted_boundary_0.05": {"input_mode": "2d", "loss_name": "bce_dice_boundary", "use_pos_weight": True, "w_sboundary": 0.05},
    "2d_bce_dice_posweighted_boundary_0.1": {"input_mode": "2d", "loss_name": "bce_dice_boundary", "use_pos_weight": True, "w_sboundary": 0.1},
    "2d_bce_dice_posweighted_boundary_0.2": {"input_mode": "2d", "loss_name": "bce_dice_boundary", "use_pos_weight": True, "w_sboundary": 0.2},
    "25d_dice": {"input_mode": "2p5d", "loss_name": "dice"},
    "25d_bce_dice": {"input_mode": "2p5d", "loss_name": "bce_dice", "use_pos_weight": False},
    "25d_bce_dice_posweighted": {"input_mode": "2p5d", "loss_name": "bce_dice", "use_pos_weight": True},
    "25d_bce_dice_boundary_0.01": {"input_mode": "2p5d", "loss_name": "bce_dice_boundary", "use_pos_weight": False, "w_sboundary": 0.01},
    "25d_bce_dice_boundary_0.05": {"input_mode": "2p5d", "loss_name": "bce_dice_boundary", "use_pos_weight": False, "w_sboundary": 0.05},
    "25d_bce_dice_boundary_0.1": {"input_mode": "2p5d", "loss_name": "bce_dice_boundary", "use_pos_weight": False, "w_sboundary": 0.1},
    "25d_bce_dice_boundary_0.2": {"input_mode": "2p5d", "loss_name": "bce_dice_boundary", "use_pos_weight": False, "w_sboundary": 0.2},
    "25d_bce_dice_posweighted_boundary_0.05": {"input_mode": "2p5d", "loss_name": "bce_dice_boundary", "use_pos_weight": True, "w_sboundary": 0.05},
    "25d_bce_dice_posweighted_boundary_0.1": {"input_mode": "2p5d", "loss_name": "bce_dice_boundary", "use_pos_weight": True, "w_sboundary": 0.1},
    "25d_bce_dice_posweighted_boundary_0.2": {"input_mode": "2p5d", "loss_name": "bce_dice_boundary", "use_pos_weight": True, "w_sboundary": 0.2},
}


@dataclass
class ExperimentConfig:
    name: str
    input_mode: str
    loss_name: str
    root: str = DEFAULT_ROOT
    max_patients: int = DEFAULT_MAX_PATIENTS
    only_tumor_slices: bool = False
    neg_to_pos_ratio: float = 2.0
    cache_volumes: bool = True
    dataset_seed: int = 3
    split_seed: int = 0
    train_frac: float = 0.8
    val_frac: float = 0.1
    batch_size: int = 4
    epochs: int = 30
    lr: float = 1e-3
    weight_decay: float = 1e-4
    base_ch: int = 32
    num_workers: int = 0
    persistent_workers: bool = False
    prefetch_factor: int = 2
    pin_memory: bool = True
    drop_last: bool = True
    metric_threshold: float = 0.7
    results_root: str = DEFAULT_RESULTS_ROOT
    w_sboundary: float = 0.05
    artifact_suffix: str = ""
    use_pos_weight: bool = False

    @property
    def artifact_prefix(self) -> str:
        prefix = f"{self.name}_seed{self.dataset_seed}"
        if self.artifact_suffix:
            prefix = f"{prefix}_{self.artifact_suffix}"
        return prefix


class AugmentedSubset(Dataset):
    def __init__(self, subset: Subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        x, y = self.subset[idx]
        if self.transform is not None:
            x, y = self.transform(x, y)
        return x, y


def parse_dataset_seed(default: int = 3) -> int:
    import argparse

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--dataset-seed", type=int, default=default)
    args, _ = parser.parse_known_args()
    return args.dataset_seed


def make_experiment_config(
    name: str,
    dataset_seed: int = 3,
    results_root: str | None = None,
    w_sboundary: float | None = None,
    artifact_suffix: str = "",
) -> ExperimentConfig:
    if name not in EXPERIMENT_SPECS:
        raise ValueError(f"Unknown experiment: {name}")

    spec = EXPERIMENT_SPECS[name]
    config = ExperimentConfig(
        name=name,
        input_mode=spec["input_mode"],
        loss_name=spec["loss_name"],
        dataset_seed=dataset_seed,
    )
    config.use_pos_weight = spec.get("use_pos_weight", False)
    config.w_sboundary = spec.get("w_sboundary", config.w_sboundary)
    if results_root is not None:
        config.results_root = results_root
    if w_sboundary is not None:
        config.w_sboundary = w_sboundary
    if artifact_suffix:
        config.artifact_suffix = artifact_suffix
    return config


def set_seed(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_dataset(config: ExperimentConfig):
    common = dict(
        root=config.root,
        max_patients=config.max_patients,
        only_tumor_slices=config.only_tumor_slices,
        neg_to_pos_ratio=config.neg_to_pos_ratio,
        cache_volumes=config.cache_volumes,
        seed=config.dataset_seed,
    )
    if config.input_mode == "2d":
        return BraTS2D(**common)
    if config.input_mode == "2p5d":
        return BraTS2p5D(**common)
    raise ValueError(f"Unsupported input_mode: {config.input_mode}")


def build_loaders(dataset, config: ExperimentConfig):
    split = build_patient_split(dataset, seed=config.split_seed, train_frac=config.train_frac, val_frac=config.val_frac)
    train_subset = Subset(dataset, split["train_indices"])
    val_subset = Subset(dataset, split["val_indices"])
    test_subset = Subset(dataset, split["test_indices"])

    # The transform is channel-agnostic: for 2.5D inputs it applies the same flip/rotation
    # to all three slices and the mask, preserving spatial alignment.
    train_ds = AugmentedSubset(train_subset, transform=RandomFlipRotate2D(max_degrees=10.0))
    val_ds = AugmentedSubset(val_subset, transform=None)
    test_ds = AugmentedSubset(test_subset, transform=None)

    loader_kwargs = dict(
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )
    if config.num_workers > 0:
        loader_kwargs["persistent_workers"] = config.persistent_workers
        loader_kwargs["prefetch_factor"] = config.prefetch_factor
    train_dl = DataLoader(train_ds, shuffle=True, drop_last=config.drop_last, **loader_kwargs)
    val_dl = DataLoader(val_ds, shuffle=False, drop_last=False, **loader_kwargs)
    test_dl = DataLoader(test_ds, shuffle=False, drop_last=False, **loader_kwargs)
    return train_dl, val_dl, test_dl, split


def build_model(config: ExperimentConfig, device):
    in_ch = 1 if config.input_mode == "2d" else 3
    return UNet(in_ch=in_ch, out_ch=1, base_ch=config.base_ch).to(device)


def compute_train_pos_weight(train_subset: Subset):
    pos_pixels = 0.0
    neg_pixels = 0.0
    for idx in train_subset.indices:
        _, y = train_subset.dataset[idx]
        pos = float(y.sum().item())
        total = float(y.numel())
        pos_pixels += pos
        neg_pixels += total - pos
    return neg_pixels / max(pos_pixels, 1.0)


def build_criterion(config: ExperimentConfig, device, pos_weight: float = 1.0):
    if config.loss_name == "dice":
        return DiceOnlyLoss().to(device)
    if config.loss_name == "bce_dice":
        return BCEDiceLoss(1.0, 1.0, pos_weight=pos_weight).to(device)
    if config.loss_name == "bce_dice_boundary":
        return BCEDiceSignedBoundaryLoss(
            w_bce=1.0,
            w_dice=1.0,
            w_sboundary=config.w_sboundary,
            pos_w=pos_weight,
        ).to(device)
    raise ValueError(f"Unsupported loss_name: {config.loss_name}")


def _build_aux_tensors(loss_name: str, y: torch.Tensor, device):
    if loss_name in {"dice", "bce_dice"}:
        return {}

    aux = {}
    if loss_name == "bce_dice_boundary":
        y_np = y.detach().cpu().numpy()
        phi = np.stack(
            [normalize_clip_phi(signed_distance_map_2d(y_np[i, 0]), clip=10.0) for i in range(y_np.shape[0])],
            axis=0,
        )
        aux["phi"] = torch.from_numpy(phi).unsqueeze(1).to(device, non_blocking=True)

    return aux


def _select_aux(aux: dict, mask: torch.Tensor):
    return {name: value[mask] for name, value in aux.items()}


def compute_loss(config: ExperimentConfig, criterion, logits: torch.Tensor, y: torch.Tensor, aux: dict):
    if config.loss_name in {"dice", "bce_dice"}:
        return criterion(logits, y)
    if config.loss_name == "bce_dice_boundary":
        return criterion(logits, y, aux["phi"])
    raise ValueError(f"Unsupported loss_name: {config.loss_name}")


def loss_component_names(config: ExperimentConfig):
    if config.loss_name == "dice":
        return ["dice_loss"]
    if config.loss_name == "bce_dice":
        return ["bce_loss", "dice_loss"]
    if config.loss_name == "bce_dice_boundary":
        return ["bce_loss", "dice_loss", "boundary_loss", "boundary_scaled_loss"]
    raise ValueError(f"Unsupported loss_name: {config.loss_name}")


def compute_loss_components(config: ExperimentConfig, criterion, logits: torch.Tensor, y: torch.Tensor, aux: dict):
    zero = logits.new_tensor(0.0)

    if config.loss_name == "dice":
        dice_loss = criterion(logits, y)
        return {
            "dice_loss": dice_loss,
            "loss": dice_loss,
        }

    if config.loss_name == "bce_dice":
        bce_loss = criterion.bce(logits, y)
        dice_loss = criterion.dice(logits, y)
        total_loss = criterion.bce_w * bce_loss + criterion.dice_w * dice_loss
        return {
            "bce_loss": bce_loss,
            "dice_loss": dice_loss,
            "loss": total_loss,
        }

    if config.loss_name == "bce_dice_boundary":
        bce_loss = criterion.bce(logits, y)
        has_tumor = y.sum(dim=(1, 2, 3)) > 0
        if has_tumor.any():
            logits_pos = logits[has_tumor]
            y_pos = y[has_tumor]
            phi_pos = aux["phi"][has_tumor]
            dice_loss = criterion.dice(logits_pos, y_pos)
            boundary_loss = criterion.sbd(logits_pos, phi_pos)
        else:
            dice_loss = zero
            boundary_loss = zero
        boundary_scaled_loss = criterion.w_sbd * boundary_loss
        total_loss = criterion.w_bce * bce_loss + criterion.w_dice * dice_loss + boundary_scaled_loss
        return {
            "bce_loss": bce_loss,
            "dice_loss": dice_loss,
            "boundary_loss": boundary_loss,
            "boundary_scaled_loss": boundary_scaled_loss,
            "loss": total_loss,
        }

    raise ValueError(f"Unsupported loss_name: {config.loss_name}")


def format_component_log(stats: dict, config: ExperimentConfig, prefix: str):
    parts = []
    for name in loss_component_names(config):
        parts.append(f"{prefix}_{name}={stats.get(name, 0.0):.4f}")
    return " ".join(parts)


@torch.no_grad()
def summarize_loader(model, dl, criterion, config: ExperimentConfig, device):
    model.eval()
    total_loss_pos = 0.0
    total_dice_pos = 0.0
    n_pos = 0
    fp_slices = 0
    neg_slices = 0
    fn_slices = 0
    component_sums = {name: 0.0 for name in loss_component_names(config)}

    for x, y in dl:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(x)
        aux = _build_aux_tensors(config.loss_name, y, device)
        has_tumor = y.sum(dim=(1, 2, 3)) > 0
        if has_tumor.any():
            logits_pos = logits[has_tumor]
            y_pos = y[has_tumor]
            aux_pos = _select_aux(aux, has_tumor)
            loss_parts = compute_loss_components(config, criterion, logits_pos, y_pos, aux_pos)
            loss_pos = loss_parts["loss"]
            dsc_pos = dice_score(logits_pos, y_pos)
            bs_pos = logits_pos.size(0)
            total_loss_pos += loss_pos.item() * bs_pos
            total_dice_pos += dsc_pos.item() * bs_pos
            n_pos += bs_pos
            for name in component_sums:
                component_sums[name] += loss_parts[name].item() * bs_pos

        preds = (torch.sigmoid(logits).view(logits.size(0), -1) > 0.5).any(dim=1)
        neg_mask = ~has_tumor
        if neg_mask.any():
            neg_slices += int(neg_mask.sum().item())
            fp_slices += int(preds[neg_mask].sum().item())
        if has_tumor.any():
            fn_slices += int((~preds[has_tumor]).sum().item())

    stats = {
        "loss": total_loss_pos / max(n_pos, 1),
        "dice": total_dice_pos / max(n_pos, 1),
        "fpr": fp_slices / max(neg_slices, 1),
        "fnr": fn_slices / max(n_pos, 1),
    }
    for name in component_sums:
        stats[name] = component_sums[name] / max(n_pos, 1)
    return stats


@torch.no_grad()
def evaluate_test_metrics(model, dl, config: ExperimentConfig, device, threshold: float | None = None):
    model.eval()
    hd_vals = []
    dice_total = 0.0
    dice_count = 0
    fp_slices = 0
    neg_slices = 0
    fn_slices = 0
    pos_slices = 0

    for x, y in dl:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        has_tumor = y.sum(dim=(1, 2, 3)) > 0
        if has_tumor.any():
            logits_pos = logits[has_tumor]
            y_pos = y[has_tumor]
            dice_thr = config.metric_threshold if threshold is None else threshold
            dice_total += dice_score(logits_pos, y_pos, thresh=dice_thr).item() * logits_pos.size(0)
            dice_count += logits_pos.size(0)
            pos_slices += int(has_tumor.sum().item())

        thr = config.metric_threshold if threshold is None else threshold
        preds = (torch.sigmoid(logits).view(logits.size(0), -1) > thr).any(dim=1)
        neg_mask = ~has_tumor
        if neg_mask.any():
            neg_slices += int(neg_mask.sum().item())
            fp_slices += int(preds[neg_mask].sum().item())
        if has_tumor.any():
            fn_slices += int((~preds[has_tumor]).sum().item())

        pred = (torch.sigmoid(logits) > thr).float().cpu().numpy()
        gt = y.cpu().numpy()
        for i in range(pred.shape[0]):
            p = pred[i, 0] > 0.5
            g = gt[i, 0] > 0.5
            if g.sum() == 0:
                continue
            hd_vals.append(hd95_2d(p, g))

    hd_vals = np.asarray(hd_vals, dtype=np.float32)
    hd_vals = hd_vals[np.isfinite(hd_vals)]
    return {
        "dice": dice_total / max(dice_count, 1),
        "hd95_mean": float(hd_vals.mean()) if hd_vals.size else float("nan"),
        "hd95_median": float(np.median(hd_vals)) if hd_vals.size else float("nan"),
        "hd95_count": int(hd_vals.shape[0]),
        "fpr": fp_slices / max(neg_slices, 1),
        "fnr": fn_slices / max(pos_slices, 1),
    }


def save_curve(history: dict, output_path: Path):
    if plt is None:
        return

    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].plot(epochs, history["train_loss"], label="train_loss")
    axes[0].plot(epochs, history["val_loss"], label="val_loss")
    axes[0].set_xlabel("epoch")
    axes[0].set_title("Loss")
    axes[0].legend()

    axes[1].plot(epochs, history["train_dice"], label="train_dice")
    axes[1].plot(epochs, history["val_dice"], label="val_dice")
    axes[1].set_xlabel("epoch")
    axes[1].set_title("Dice")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def run_experiment(config: ExperimentConfig):
    set_seed(config.split_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    dataset = build_dataset(config)
    train_dl, val_dl, test_dl, split = build_loaders(dataset, config)
    model = build_model(config, device)
    pos_weight = 1.0
    if config.use_pos_weight:
        train_subset = train_dl.dataset.subset
        pos_weight = compute_train_pos_weight(train_subset)
    criterion = build_criterion(config, device, pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    artifact_prefix = config.artifact_prefix
    out_dir = Path(config.results_root) / artifact_prefix
    out_dir.mkdir(parents=True, exist_ok=True)

    config_payload = asdict(config)
    config_payload["effective_pos_weight"] = pos_weight
    config_payload["augmentation"] = {
        "train_only": True,
        "horizontal_flip": True,
        "vertical_flip": True,
        "rotation_degrees": [-10.0, 10.0],
    }
    config_payload["split"] = {
        "train_frac": config.train_frac,
        "val_frac": config.val_frac,
        "test_frac": 1.0 - config.train_frac - config.val_frac,
        "seed": config.split_seed,
        "train_patients": split["train_patients"],
        "val_patients": split["val_patients"],
        "test_patients": split["test_patients"],
    }
    (out_dir / f"{artifact_prefix}_config.json").write_text(json.dumps(config_payload, indent=2), encoding="utf-8")

    history = {
        "train_loss": [],
        "train_dice": [],
        "train_fpr": [],
        "train_fnr": [],
        "val_loss": [],
        "val_dice": [],
        "val_fpr": [],
        "val_fnr": [],
    }
    for name in loss_component_names(config):
        history[f"train_{name}"] = []
        history[f"val_{name}"] = []

    best_val_dice = float("-inf")
    best_ckpt = out_dir / f"{artifact_prefix}_best.pt"

    for epoch in range(1, config.epochs + 1):
        epoch_start = time.time()
        model.train()
        running_loss_pos = 0.0
        running_dice_pos = 0.0
        train_pos = 0
        fp_slices = 0
        neg_slices = 0
        fn_slices = 0
        component_sums = {name: 0.0 for name in loss_component_names(config)}

        for x, y in train_dl:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            aux = _build_aux_tensors(config.loss_name, y, device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = compute_loss(config, criterion, logits, y, aux)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            with torch.no_grad():
                has_tumor = y.sum(dim=(1, 2, 3)) > 0
                if has_tumor.any():
                    logits_pos = logits[has_tumor]
                    y_pos = y[has_tumor]
                    aux_pos = _select_aux(aux, has_tumor)
                    loss_parts = compute_loss_components(config, criterion, logits_pos, y_pos, aux_pos)
                    loss_pos = loss_parts["loss"]
                    dsc_pos = dice_score(logits_pos, y_pos)
                    bs_pos = logits_pos.size(0)
                    running_loss_pos += loss_pos.item() * bs_pos
                    running_dice_pos += dsc_pos.item() * bs_pos
                    train_pos += bs_pos
                    for name in component_sums:
                        component_sums[name] += loss_parts[name].item() * bs_pos

                preds = (torch.sigmoid(logits).view(logits.size(0), -1) > 0.5).any(dim=1)
                neg_mask = ~has_tumor
                if neg_mask.any():
                    neg_slices += int(neg_mask.sum().item())
                    fp_slices += int(preds[neg_mask].sum().item())
                if has_tumor.any():
                    fn_slices += int((~preds[has_tumor]).sum().item())

        train_loss = running_loss_pos / max(train_pos, 1)
        train_dice = running_dice_pos / max(train_pos, 1)
        train_fpr = fp_slices / max(neg_slices, 1)
        train_fnr = fn_slices / max(train_pos, 1)
        val_stats = summarize_loader(model, val_dl, criterion, config, device)
        train_component_stats = {name: component_sums[name] / max(train_pos, 1) for name in component_sums}

        history["train_loss"].append(train_loss)
        history["train_dice"].append(train_dice)
        history["train_fpr"].append(train_fpr)
        history["train_fnr"].append(train_fnr)
        history["val_loss"].append(val_stats["loss"])
        history["val_dice"].append(val_stats["dice"])
        history["val_fpr"].append(val_stats["fpr"])
        history["val_fnr"].append(val_stats["fnr"])
        for name in loss_component_names(config):
            history[f"train_{name}"].append(train_component_stats[name])
            history[f"val_{name}"].append(val_stats[name])
        epoch_seconds = time.time() - epoch_start

        if val_stats["dice"] > best_val_dice:
            best_val_dice = val_stats["dice"]
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "val_dice": best_val_dice,
                    "config": config_payload,
                },
                best_ckpt,
            )

        print(
            f"Epoch {epoch:02d} | "
            f"train_loss={train_loss:.4f} {format_component_log(train_component_stats, config, 'train')} "
            f"train_dice={train_dice:.4f} train_fpr={train_fpr:.4f} train_fnr={train_fnr:.4f} | "
            f"val_loss={val_stats['loss']:.4f} {format_component_log(val_stats, config, 'val')} "
            f"val_dice={val_stats['dice']:.4f} val_fpr={val_stats['fpr']:.4f} val_fnr={val_stats['fnr']:.4f} | "
            f"time={epoch_seconds:.1f}s"
        )

    save_curve(history, out_dir / f"{artifact_prefix}_train_curve.png")
    (out_dir / f"{artifact_prefix}_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

    best_state = torch.load(best_ckpt, map_location=device)
    model.load_state_dict(best_state["model"])
    test_metrics = evaluate_test_metrics(model, test_dl, config, device)
    (out_dir / f"{artifact_prefix}_metrics_test.json").write_text(json.dumps(test_metrics, indent=2), encoding="utf-8")

    print("best checkpoint:", best_ckpt)
    print("test metrics:", json.dumps(test_metrics, indent=2))
    return {
        "output_dir": str(out_dir),
        "best_ckpt": str(best_ckpt),
        "metrics_test": test_metrics,
    }


def load_saved_model(ckpt_path: str | Path, input_mode: str, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt.get("config", {})
    base_ch = cfg.get("base_ch", 32)
    in_ch = 1 if input_mode == "2d" else 3
    model = UNet(in_ch=in_ch, out_ch=1, base_ch=base_ch).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


def evaluate_and_save_threshold_metrics(
    config: ExperimentConfig,
    threshold: float,
    output_path: str | Path | None = None,
):
    artifact_prefix = config.artifact_prefix
    out_dir = Path(config.results_root) / artifact_prefix
    if output_path is None:
        output_path = out_dir / f"{artifact_prefix}_metrics_test_{threshold:g}.json"
    else:
        output_path = Path(output_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = build_dataset(config)
    _, _, test_dl, _ = build_loaders(dataset, config)
    model = load_saved_model(out_dir / f"{artifact_prefix}_best.pt", input_mode=config.input_mode, device=device)
    metrics = evaluate_test_metrics(model, test_dl, config, device, threshold=threshold)
    metrics["selected_threshold"] = threshold
    output_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics, output_path


def evaluate_saved_experiment(name: str, input_mode: str, dataset_seed: int = 3):
    artifact_prefix = f"{name}_seed{dataset_seed}"
    out_dir = Path("results") / artifact_prefix
    config = json.loads((out_dir / f"{artifact_prefix}_config.json").read_text(encoding="utf-8"))
    exp_cfg = ExperimentConfig(**{k: v for k, v in config.items() if k in ExperimentConfig.__dataclass_fields__})
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = build_dataset(exp_cfg)
    _, _, test_dl, _ = build_loaders(dataset, exp_cfg)
    model = load_saved_model(out_dir / f"{artifact_prefix}_best.pt", input_mode=input_mode, device=device)
    return evaluate_test_metrics(model, test_dl, exp_cfg, device)
