import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from skimage.measure import perimeter

from brats2p5d_dataset import _clamp_triplet, _load_nifti, _robust_zscore
from experiment_utils import ExperimentConfig, load_saved_model
from metrics_boundary import hd95_2d
from splits import build_patient_split
from brats2d_dataset import BraTS2D
from brats2p5d_dataset import BraTS2p5D


DEFAULT_RESULTS_ROOT = Path("result_new")
DEFAULT_OUTPUT_ROOT = Path("result_0.5")
DEFAULT_MODEL_2D = "2d_bce_dice_boundary_0.05"
DEFAULT_MODEL_25D = "25d_bce_dice_boundary_0.05"
DEFAULT_THRESHOLD = 0.5
DELTA_METRICS = ["delta_dice", "delta_hd95", "delta_fnr", "delta_fpr"]
SIZE_GROUPS = ["bottom_10pct", "top_10pct"]
COMPLEXITY_GROUPS = ["bottom_10pct", "top_10pct"]


def parse_args():
    parser = argparse.ArgumentParser(description="Grouped tau=0.5 analysis for 2D vs 2.5D boundary models.")
    parser.add_argument("--results-root", default=str(DEFAULT_RESULTS_ROOT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--model-2d", default=DEFAULT_MODEL_2D)
    parser.add_argument("--model-25d", default=DEFAULT_MODEL_25D)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--seed-end", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--extreme-frac", type=float, default=0.10)
    return parser.parse_args()


def load_config(results_root: Path, model_name: str, dataset_seed: int) -> tuple[ExperimentConfig, Path]:
    artifact_prefix = f"{model_name}_seed{dataset_seed}"
    out_dir = results_root / artifact_prefix
    cfg_path = out_dir / f"{artifact_prefix}_config.json"
    ckpt_path = out_dir / f"{artifact_prefix}_best.pt"
    if not cfg_path.exists() or not ckpt_path.exists():
        raise FileNotFoundError(f"Missing config or checkpoint for {artifact_prefix}")

    cfg_json = json.loads(cfg_path.read_text(encoding="utf-8"))
    cfg_data = {k: v for k, v in cfg_json.items() if k in ExperimentConfig.__dataclass_fields__}
    config = ExperimentConfig(**cfg_data)
    return config, ckpt_path


def build_dataset_for_config(config: ExperimentConfig):
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


def patient_dir_map(dataset) -> dict[str, Path]:
    patient_dirs = dataset.patient_dirs if hasattr(dataset, "patient_dirs") else dataset.ds.patient_dirs
    return {Path(p).name: Path(p) for p in patient_dirs}


def load_patient_volume(patient_dir: Path):
    flair_path = next(patient_dir.glob("*_flair.nii*"))
    seg_path = next(patient_dir.glob("*_seg.nii*"))
    flair = _load_nifti(flair_path)
    seg = _load_nifti(seg_path)
    gt = (seg > 0).astype(np.uint8)
    flair_n = _robust_zscore(flair)
    return flair_n, gt


def predict_volume(model, flair_n: np.ndarray, input_mode: str, threshold: float, device, batch_size: int):
    _, _, depth = flair_n.shape
    pred_volume = np.zeros_like(flair_n, dtype=np.uint8)

    with torch.no_grad():
        for start in range(0, depth, batch_size):
            stop = min(start + batch_size, depth)
            batch_ts = list(range(start, stop))
            if input_mode == "2d":
                batch = np.stack([flair_n[:, :, t][None, ...] for t in batch_ts], axis=0).astype(np.float32)
            elif input_mode == "2p5d":
                batch = []
                for t in batch_ts:
                    t0, t1, t2 = _clamp_triplet(t, depth)
                    batch.append(np.stack([flair_n[:, :, t0], flair_n[:, :, t1], flair_n[:, :, t2]], axis=0))
                batch = np.stack(batch, axis=0).astype(np.float32)
            else:
                raise ValueError(f"Unsupported input_mode: {input_mode}")

            x = torch.from_numpy(batch).to(device, non_blocking=True)
            logits = model(x)
            probs = torch.sigmoid(logits)
            pred = (probs > threshold).cpu().numpy().astype(np.uint8)[:, 0]
            for i, t in enumerate(batch_ts):
                pred_volume[:, :, t] = pred[i]

    return pred_volume


def dice_3d(pred: np.ndarray, gt: np.ndarray) -> float:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    pred_sum = pred.sum()
    gt_sum = gt.sum()
    if pred_sum == 0 and gt_sum == 0:
        return 1.0
    if pred_sum == 0 or gt_sum == 0:
        return 0.0
    inter = np.logical_and(pred, gt).sum()
    return float((2.0 * inter) / (pred_sum + gt_sum + 1e-6))


def patient_hd95(pred: np.ndarray, gt: np.ndarray) -> float:
    values = []
    depth = gt.shape[-1]
    for t in range(depth):
        gt_slice = gt[:, :, t]
        if gt_slice.sum() == 0:
            continue
        value = hd95_2d(pred[:, :, t], gt_slice)
        if np.isfinite(value):
            values.append(value)
    if values:
        return float(np.median(np.asarray(values, dtype=np.float32)))
    if gt.sum() == 0 and pred.sum() == 0:
        return 0.0
    return float("inf")


def patient_fnr_fpr(pred: np.ndarray, gt: np.ndarray) -> tuple[float, float]:
    depth = gt.shape[-1]
    pos_count = 0
    neg_count = 0
    fn_count = 0
    fp_count = 0
    for t in range(depth):
        gt_slice = gt[:, :, t]
        pred_slice = pred[:, :, t]
        gt_has = gt_slice.sum() > 0
        pred_has = pred_slice.sum() > 0
        if gt_has:
            pos_count += 1
            if not pred_has:
                fn_count += 1
        else:
            neg_count += 1
            if pred_has:
                fp_count += 1
    fnr = fn_count / max(pos_count, 1)
    fpr = fp_count / max(neg_count, 1)
    return float(fnr), float(fpr)


def compute_size_features(gt: np.ndarray):
    tumor_vox = int(gt.sum())
    total_vox = int(gt.size)
    ratio = float(tumor_vox / max(total_vox, 1))
    return tumor_vox, total_vox, ratio


def compute_complexity(gt: np.ndarray, eps: float = 1e-6):
    depth = gt.shape[-1]
    ratios = []
    for t in range(depth):
        mask = gt[:, :, t].astype(np.uint8)
        area = float(mask.sum())
        if area <= 0:
            continue
        peri = float(perimeter(mask, neighborhood=8))
        ratios.append(peri / (area + eps))
    if not ratios:
        return 0.0
    return float(np.mean(ratios))


def assign_extremes(values: pd.Series, frac: float, low_label: str = "bottom_10pct", high_label: str = "top_10pct") -> pd.Series:
    order = np.argsort(values.to_numpy(dtype=float), kind="stable")
    groups = pd.Series(index=values.index, dtype="object")
    n = len(values)
    k = max(1, int(np.ceil(n * frac)))
    low_idx = order[:k]
    high_idx = order[-k:]
    for idx in low_idx:
        groups.iloc[int(idx)] = low_label
    for idx in high_idx:
        groups.iloc[int(idx)] = high_label
    return groups


def summarize_groups(df: pd.DataFrame, group_col: str, grouping_type: str):
    rows = []
    for group_name, group_df in df.groupby(group_col, sort=False):
        for metric in DELTA_METRICS:
            values = group_df[metric].to_numpy(dtype=float)
            rows.append(
                {
                    "grouping_type": grouping_type,
                    "group_name": group_name,
                    "metric": metric,
                    "count": int(len(values)),
                    "mean": float(np.mean(values)) if len(values) else np.nan,
                    "std": float(np.std(values, ddof=1)) if len(values) > 1 else np.nan,
                }
            )
    return pd.DataFrame(rows)


def plot_group_summary(summary_df: pd.DataFrame, grouping_type: str, group_order: list[str], metric: str, output_path: Path):
    plot_df = summary_df[
        (summary_df["grouping_type"] == grouping_type) & (summary_df["metric"] == metric)
    ].copy()
    plot_df["group_name"] = pd.Categorical(plot_df["group_name"], categories=group_order, ordered=True)
    plot_df = plot_df.sort_values("group_name")

    x = np.arange(len(plot_df))
    means = plot_df["mean_across_seeds"].to_numpy(dtype=float)
    stds = plot_df["std_across_seeds"].fillna(0.0).to_numpy(dtype=float)

    plt.figure(figsize=(7, 4.5))
    plt.bar(x, means, yerr=stds, capsize=4)
    plt.xticks(x, plot_df["group_name"])
    plt.ylabel(metric)
    plt.title(f"{metric} by {grouping_type} (25d - 2d)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def main():
    args = parse_args()
    results_root = Path(args.results_root)
    output_root = Path(args.output_root)
    by_seed_dir = output_root / "by_seed"
    tables_dir = output_root / "tables"
    plots_dir = output_root / "plots"
    by_seed_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    all_group_seed_rows = []

    for dataset_seed in range(args.seed_start, args.seed_end + 1):
        print(f"Processing seed={dataset_seed}")
        config_2d, ckpt_2d = load_config(results_root, args.model_2d, dataset_seed)
        config_25d, ckpt_25d = load_config(results_root, args.model_25d, dataset_seed)

        dataset_2d = build_dataset_for_config(config_2d)
        dataset_25d = build_dataset_for_config(config_25d)
        split_2d = build_patient_split(dataset_2d, seed=config_2d.split_seed, train_frac=config_2d.train_frac, val_frac=config_2d.val_frac)
        split_25d = build_patient_split(dataset_25d, seed=config_25d.split_seed, train_frac=config_25d.train_frac, val_frac=config_25d.val_frac)

        if split_2d["test_patients"] != split_25d["test_patients"]:
            raise RuntimeError(f"Test patient mismatch for seed={dataset_seed}")

        test_patients = split_2d["test_patients"]
        patient_dirs = patient_dir_map(dataset_25d)
        model_2d = load_saved_model(ckpt_2d, input_mode=config_2d.input_mode, device=device)
        model_25d = load_saved_model(ckpt_25d, input_mode=config_25d.input_mode, device=device)

        patient_rows = []
        for patient_id in test_patients:
            patient_dir = patient_dirs[patient_id]
            flair_n, gt = load_patient_volume(patient_dir)
            pred_2d = predict_volume(model_2d, flair_n, config_2d.input_mode, args.threshold, device, args.batch_size)
            pred_25d = predict_volume(model_25d, flair_n, config_25d.input_mode, args.threshold, device, args.batch_size)

            tumor_vox, total_vox, ratio = compute_size_features(gt)
            complexity = compute_complexity(gt)

            dice_2d = dice_3d(pred_2d, gt)
            dice_25d = dice_3d(pred_25d, gt)
            hd95_2d_val = patient_hd95(pred_2d, gt)
            hd95_25d_val = patient_hd95(pred_25d, gt)
            fnr_2d, fpr_2d = patient_fnr_fpr(pred_2d, gt)
            fnr_25d, fpr_25d = patient_fnr_fpr(pred_25d, gt)

            patient_rows.append(
                {
                    "seed": dataset_seed,
                    "patient_id": patient_id,
                    "tumor_vox": tumor_vox,
                    "total_vox": total_vox,
                    "ratio": ratio,
                    "complexity": complexity,
                    "dice_2d": dice_2d,
                    "dice_25d": dice_25d,
                    "hd95_2d": hd95_2d_val,
                    "hd95_25d": hd95_25d_val,
                    "fnr_2d": fnr_2d,
                    "fnr_25d": fnr_25d,
                    "fpr_2d": fpr_2d,
                    "fpr_25d": fpr_25d,
                    "delta_dice": dice_25d - dice_2d,
                    "delta_hd95": hd95_25d_val - hd95_2d_val,
                    "delta_fnr": fnr_25d - fnr_2d,
                    "delta_fpr": fpr_25d - fpr_2d,
                }
            )

        patient_df = pd.DataFrame(patient_rows)
        patient_df["size_group"] = assign_extremes(patient_df["ratio"], frac=args.extreme_frac)
        patient_df["complexity_group"] = assign_extremes(patient_df["complexity"], frac=args.extreme_frac)

        patients_size_df = patient_df[
            [
                "seed",
                "patient_id",
                "tumor_vox",
                "total_vox",
                "ratio",
                "size_group",
                "dice_2d",
                "dice_25d",
                "hd95_2d",
                "hd95_25d",
                "fnr_2d",
                "fnr_25d",
                "fpr_2d",
                "fpr_25d",
                "delta_dice",
                "delta_hd95",
                "delta_fnr",
                "delta_fpr",
            ]
        ].copy()
        patients_complexity_df = patient_df[
            [
                "seed",
                "patient_id",
                "complexity",
                "complexity_group",
                "dice_2d",
                "dice_25d",
                "hd95_2d",
                "hd95_25d",
                "fnr_2d",
                "fnr_25d",
                "fpr_2d",
                "fpr_25d",
                "delta_dice",
                "delta_hd95",
                "delta_fnr",
                "delta_fpr",
            ]
        ].copy()

        patients_size_df = patients_size_df[patients_size_df["size_group"].notna()].copy()
        patients_complexity_df = patients_complexity_df[patients_complexity_df["complexity_group"].notna()].copy()

        size_summary = summarize_groups(patients_size_df, "size_group", "size_extreme_10pct")
        complexity_summary = summarize_groups(patients_complexity_df, "complexity_group", "complexity_extreme_10pct")
        size_summary["seed"] = dataset_seed
        complexity_summary["seed"] = dataset_seed
        all_group_seed_rows.extend(size_summary.to_dict("records"))
        all_group_seed_rows.extend(complexity_summary.to_dict("records"))

        patients_size_df.to_csv(by_seed_dir / f"seed{dataset_seed}_patients_size_extreme_10pct.csv", index=False, encoding="utf-8")
        patients_complexity_df.to_csv(by_seed_dir / f"seed{dataset_seed}_patients_complexity_extreme_10pct.csv", index=False, encoding="utf-8")
        size_summary.to_csv(by_seed_dir / f"seed{dataset_seed}_summary_size_extreme_10pct.csv", index=False, encoding="utf-8")
        complexity_summary.to_csv(by_seed_dir / f"seed{dataset_seed}_summary_complexity_extreme_10pct.csv", index=False, encoding="utf-8")

    group_seed_df = pd.DataFrame(all_group_seed_rows)
    group_seed_df.to_csv(tables_dir / "grouped_summary_by_seed_extreme_10pct.csv", index=False, encoding="utf-8")

    across_rows = []
    for (grouping_type, group_name, metric), group_df in group_seed_df.groupby(["grouping_type", "group_name", "metric"]):
        seed_means = group_df["mean"].to_numpy(dtype=float)
        across_rows.append(
            {
                "grouping_type": grouping_type,
                "group_name": group_name,
                "metric": metric,
                "mean_across_seeds": float(np.mean(seed_means)) if len(seed_means) else np.nan,
                "std_across_seeds": float(np.std(seed_means, ddof=1)) if len(seed_means) > 1 else np.nan,
                "n_seeds_used": int(len(seed_means)),
            }
        )

    across_df = pd.DataFrame(across_rows)
    across_df.to_csv(tables_dir / "grouped_summary_across_seeds_extreme_10pct.csv", index=False, encoding="utf-8")
    (tables_dir / "grouped_summary_across_seeds_extreme_10pct.json").write_text(
        across_df.to_json(orient="records", force_ascii=False, indent=2),
        encoding="utf-8",
    )

    metric_to_filename = {
        "delta_dice": "delta_dice",
        "delta_hd95": "delta_hd95",
        "delta_fnr": "delta_fnr",
        "delta_fpr": "delta_fpr",
    }
    for metric, stub in metric_to_filename.items():
        plot_group_summary(across_df, "size_extreme_10pct", SIZE_GROUPS, metric, plots_dir / f"{stub}_by_size_extreme_10pct.png")
        plot_group_summary(across_df, "complexity_extreme_10pct", COMPLEXITY_GROUPS, metric, plots_dir / f"{stub}_by_complexity_extreme_10pct.png")

    print("Saved:")
    print(by_seed_dir)
    print(tables_dir / "grouped_summary_by_seed_extreme_10pct.csv")
    print(tables_dir / "grouped_summary_across_seeds_extreme_10pct.csv")
    print(tables_dir / "grouped_summary_across_seeds_extreme_10pct.json")
    print(plots_dir)


if __name__ == "__main__":
    main()
