import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from skimage.measure import perimeter

from brats2d_dataset import BraTS2D
from brats2p5d_dataset import BraTS2p5D, _clamp_triplet, _load_nifti, _robust_zscore
from experiment_utils import ExperimentConfig, load_saved_model
from metrics_boundary import hd95_2d
from splits import build_patient_split


DEFAULT_RESULTS_ROOT = Path("result_500")
DEFAULT_THRESHOLD = 0.5
MODEL_NAMES = {
    "2d_bce_dice": "2d_bce_dice",
    "2d_boundary": "2d_bce_dice_boundary_0.05",
    "25d_bce_dice": "25d_bce_dice",
    "25d_boundary": "25d_bce_dice_boundary_0.05",
}
COMPARISONS = {
    "2d_boundary_vs_nonboundary": ("2d_boundary", "2d_bce_dice"),
    "25d_vs_2d_bce_dice": ("25d_bce_dice", "2d_bce_dice"),
    "25d_boundary_vs_2d_boundary": ("25d_boundary", "2d_boundary"),
    "25d_boundary_vs_nonboundary": ("25d_boundary", "25d_bce_dice"),
}
TARGET_METRICS = ["dice", "hd95", "fnr", "fpr"]
GROUP_LABELS = ["low_30pct", "mid_40pct", "high_30pct"]


def parse_args():
    parser = argparse.ArgumentParser(description="Grouped delta analysis for result_500 at threshold 0.5.")
    parser.add_argument("--results-root", default=str(DEFAULT_RESULTS_ROOT))
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--seed-end", type=int, default=2)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--batch-size", type=int, default=16)
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
    return ExperimentConfig(**cfg_data), ckpt_path


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
                triplets = []
                for t in batch_ts:
                    t0, t1, t2 = _clamp_triplet(t, depth)
                    triplets.append(np.stack([flair_n[:, :, t0], flair_n[:, :, t1], flair_n[:, :, t2]], axis=0))
                batch = np.stack(triplets, axis=0).astype(np.float32)
            else:
                raise ValueError(f"Unsupported input_mode: {input_mode}")

            x = torch.from_numpy(batch).to(device, non_blocking=True)
            pred = (torch.sigmoid(model(x)) > threshold).cpu().numpy().astype(np.uint8)[:, 0]
            for i, t in enumerate(batch_ts):
                pred_volume[:, :, t] = pred[i]

    return pred_volume


def dice_3d(pred: np.ndarray, gt: np.ndarray) -> float:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    pred_sum = int(pred.sum())
    gt_sum = int(gt.sum())
    if pred_sum == 0 and gt_sum == 0:
        return 1.0
    if pred_sum == 0 or gt_sum == 0:
        return 0.0
    inter = int(np.logical_and(pred, gt).sum())
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
    pos_count = 0
    neg_count = 0
    fn_count = 0
    fp_count = 0
    for t in range(gt.shape[-1]):
        gt_has = gt[:, :, t].sum() > 0
        pred_has = pred[:, :, t].sum() > 0
        if gt_has:
            pos_count += 1
            if not pred_has:
                fn_count += 1
        else:
            neg_count += 1
            if pred_has:
                fp_count += 1
    return float(fn_count / max(pos_count, 1)), float(fp_count / max(neg_count, 1))


def compute_size_ratio(gt: np.ndarray) -> tuple[int, int, float]:
    tumor_vox = int(gt.sum())
    total_vox = int(gt.size)
    return tumor_vox, total_vox, float(tumor_vox / max(total_vox, 1))


def compute_complexity(gt: np.ndarray, eps: float = 1e-6) -> float:
    values = []
    for t in range(gt.shape[-1]):
        mask = gt[:, :, t].astype(np.uint8)
        area = float(mask.sum())
        if area <= 0:
            continue
        values.append(float(perimeter(mask, neighborhood=8)) / (area + eps))
    return float(np.mean(values)) if values else 0.0


def assign_30_40_30(values: pd.Series) -> pd.Series:
    order = np.argsort(values.to_numpy(dtype=float), kind="stable")
    groups = pd.Series(index=values.index, dtype="object")
    n = len(values)
    low_n = int(np.floor(n * 0.30))
    high_n = int(np.floor(n * 0.30))
    if n >= 3:
        low_n = max(1, low_n)
        high_n = max(1, high_n)
    if low_n + high_n >= n:
        low_n = max(1, n // 3)
        high_n = max(1, n // 3)
    mid_start = low_n
    mid_end = n - high_n

    for idx in order[:low_n]:
        groups.iloc[int(idx)] = GROUP_LABELS[0]
    for idx in order[mid_start:mid_end]:
        groups.iloc[int(idx)] = GROUP_LABELS[1]
    for idx in order[mid_end:]:
        groups.iloc[int(idx)] = GROUP_LABELS[2]
    return groups


def metric_bundle(pred: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    fnr, fpr = patient_fnr_fpr(pred, gt)
    return {
        "dice": dice_3d(pred, gt),
        "hd95": patient_hd95(pred, gt),
        "fnr": fnr,
        "fpr": fpr,
    }


def add_delta_columns(df: pd.DataFrame):
    for comp_name, (lhs_model, rhs_model) in COMPARISONS.items():
        for metric in TARGET_METRICS:
            lhs = f"{lhs_model}_{metric}"
            rhs = f"{rhs_model}_{metric}"
            df[f"{comp_name}_{metric}"] = df[lhs] - df[rhs]
    return df


def summarize_seed_groups(df: pd.DataFrame, group_col: str, grouping_name: str) -> pd.DataFrame:
    rows = []
    for group_name, group_df in df.groupby(group_col, sort=False):
        for comp_name in COMPARISONS:
            for metric in TARGET_METRICS:
                col = f"{comp_name}_{metric}"
                values = group_df[col].to_numpy(dtype=float)
                rows.append(
                    {
                        "grouping": grouping_name,
                        "group_name": group_name,
                        "comparison": comp_name,
                        "metric": metric,
                        "count": int(len(values)),
                        "mean": float(np.mean(values)) if len(values) else np.nan,
                        "std": float(np.std(values, ddof=1)) if len(values) > 1 else np.nan,
                    }
                )
    return pd.DataFrame(rows)


def summarize_across_seeds(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (group_name, comparison, metric), group_df in df.groupby(["group_name", "comparison", "metric"]):
        means = group_df["mean"].to_numpy(dtype=float)
        rows.append(
            {
                "group_name": group_name,
                "comparison": comparison,
                "metric": metric,
                "mean_across_seeds": float(np.mean(means)) if len(means) else np.nan,
                "std_across_seeds": float(np.std(means, ddof=1)) if len(means) > 1 else np.nan,
                "n_seeds_used": int(len(means)),
            }
        )
    return pd.DataFrame(rows)


def plot_grouped_metric(summary_df: pd.DataFrame, metric: str, title: str, out_path: Path):
    plot_df = summary_df[summary_df["metric"] == metric].copy()
    comparisons = list(COMPARISONS.keys())
    x = np.arange(len(GROUP_LABELS))
    width = 0.18

    plt.figure(figsize=(10, 5))
    for idx, comp_name in enumerate(comparisons):
        comp_df = plot_df[plot_df["comparison"] == comp_name].copy()
        comp_df["group_name"] = pd.Categorical(comp_df["group_name"], categories=GROUP_LABELS, ordered=True)
        comp_df = comp_df.sort_values("group_name")
        means = comp_df["mean_across_seeds"].to_numpy(dtype=float)
        stds = comp_df["std_across_seeds"].fillna(0.0).to_numpy(dtype=float)
        offset = (idx - (len(comparisons) - 1) / 2) * width
        plt.bar(x + offset, means, width=width, yerr=stds, capsize=4, label=comp_name)

    plt.xticks(x, GROUP_LABELS)
    plt.ylabel(metric)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def save_outputs(base_dir: Path, patient_df: pd.DataFrame, by_seed_df: pd.DataFrame, across_df: pd.DataFrame, stub: str):
    base_dir.mkdir(parents=True, exist_ok=True)
    patient_csv = base_dir / f"patients_{stub}_by_seed.csv"
    by_seed_csv = base_dir / "grouped_delta_by_seed.csv"
    across_csv = base_dir / "grouped_delta_across_seeds.csv"
    across_json = base_dir / "grouped_delta_across_seeds.json"

    patient_df.to_csv(patient_csv, index=False, encoding="utf-8")
    by_seed_df.to_csv(by_seed_csv, index=False, encoding="utf-8")
    across_df.to_csv(across_csv, index=False, encoding="utf-8")
    across_json.write_text(across_df.to_json(orient="records", force_ascii=False, indent=2), encoding="utf-8")

    plot_grouped_metric(across_df, "dice", f"delta_dice by {stub}", base_dir / f"delta_dice_by_{stub}.png")
    plot_grouped_metric(across_df, "hd95", f"delta_hd95 by {stub}", base_dir / f"delta_hd95_by_{stub}.png")
    plot_grouped_metric(across_df, "fnr", f"delta_fnr by {stub}", base_dir / f"delta_fnr_by_{stub}.png")
    plot_grouped_metric(across_df, "fpr", f"delta_fpr by {stub}", base_dir / f"delta_fpr_by_{stub}.png")


def main():
    args = parse_args()
    results_root = Path(args.results_root)
    summary_size_dir = results_root / "summary_size"
    summary_complexity_dir = results_root / "summary_complexity"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    patient_rows = []
    size_seed_rows = []
    complexity_seed_rows = []

    for dataset_seed in range(args.seed_start, args.seed_end + 1):
        print(f"processing seed={dataset_seed}")
        configs = {}
        checkpoints = {}
        datasets = {}
        splits = {}
        models = {}

        for key, model_name in MODEL_NAMES.items():
            config, ckpt_path = load_config(results_root, model_name, dataset_seed)
            dataset = build_dataset_for_config(config)
            split = build_patient_split(dataset, seed=config.split_seed, train_frac=config.train_frac, val_frac=config.val_frac)
            configs[key] = config
            checkpoints[key] = ckpt_path
            datasets[key] = dataset
            splits[key] = split["test_patients"]
            models[key] = load_saved_model(ckpt_path, input_mode=config.input_mode, device=device)

        reference_patients = splits["2d_bce_dice"]
        for key, test_patients in splits.items():
            if test_patients != reference_patients:
                raise RuntimeError(f"Test patient mismatch for seed={dataset_seed}: {key}")

        patient_dirs = patient_dir_map(datasets["25d_boundary"])
        for patient_id in reference_patients:
            flair_n, gt = load_patient_volume(patient_dirs[patient_id])
            tumor_vox, total_vox, ratio = compute_size_ratio(gt)
            complexity = compute_complexity(gt)
            row = {
                "seed": dataset_seed,
                "patient_id": patient_id,
                "tumor_vox": tumor_vox,
                "total_vox": total_vox,
                "ratio": ratio,
                "complexity": complexity,
            }

            for key, config in configs.items():
                pred = predict_volume(models[key], flair_n, config.input_mode, args.threshold, device, args.batch_size)
                metrics = metric_bundle(pred, gt)
                for metric, value in metrics.items():
                    row[f"{key}_{metric}"] = value

            patient_rows.append(row)

        seed_df = pd.DataFrame([r for r in patient_rows if r["seed"] == dataset_seed]).copy()
        seed_df = add_delta_columns(seed_df)
        seed_df["size_group"] = assign_30_40_30(seed_df["ratio"])
        seed_df["complexity_group"] = assign_30_40_30(seed_df["complexity"])

        size_seed_summary = summarize_seed_groups(seed_df, "size_group", "size")
        size_seed_summary["seed"] = dataset_seed
        complexity_seed_summary = summarize_seed_groups(seed_df, "complexity_group", "complexity")
        complexity_seed_summary["seed"] = dataset_seed

        size_seed_rows.extend(size_seed_summary.to_dict("records"))
        complexity_seed_rows.extend(complexity_seed_summary.to_dict("records"))

        patient_rows = [r for r in patient_rows if r["seed"] != dataset_seed] + seed_df.to_dict("records")

    patient_df = pd.DataFrame(patient_rows).sort_values(["seed", "patient_id"]).reset_index(drop=True)
    size_seed_df = pd.DataFrame(size_seed_rows).sort_values(["seed", "group_name", "comparison", "metric"]).reset_index(drop=True)
    complexity_seed_df = pd.DataFrame(complexity_seed_rows).sort_values(["seed", "group_name", "comparison", "metric"]).reset_index(drop=True)

    size_across_df = summarize_across_seeds(size_seed_df)
    complexity_across_df = summarize_across_seeds(complexity_seed_df)

    size_patient_df = patient_df.drop(columns=["complexity_group"], errors="ignore")
    complexity_patient_df = patient_df.drop(columns=["size_group"], errors="ignore")

    save_outputs(summary_size_dir, size_patient_df, size_seed_df, size_across_df, "size")
    save_outputs(summary_complexity_dir, complexity_patient_df, complexity_seed_df, complexity_across_df, "complexity")

    print("saved:")
    print(summary_size_dir)
    print(summary_complexity_dir)


if __name__ == "__main__":
    main()
