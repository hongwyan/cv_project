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
DEFAULT_BATCH_SIZE = 32
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
TARGET_METRICS = ["dice", "hd95"]


def parse_args():
    parser = argparse.ArgumentParser(description="Slice-level Dice/HD95 delta analysis for result_500 at threshold 0.5.")
    parser.add_argument("--results-root", default=str(DEFAULT_RESULTS_ROOT))
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--seed-end", type=int, default=2)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
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


def dice_2d_binary(pred: np.ndarray, gt: np.ndarray) -> float:
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


def complexity_2d(mask: np.ndarray, eps: float = 1e-6) -> float:
    area = float(mask.sum())
    if area <= 0:
        return 0.0
    return float(perimeter(mask.astype(np.uint8), neighborhood=8)) / (area + eps)


def assign_bottom20(values: pd.Series) -> pd.Series:
    n = len(values)
    k = max(1, int(np.ceil(n * 0.20)))
    order = np.argsort(values.to_numpy(dtype=float), kind="stable")
    labels = pd.Series(index=values.index, data=False)
    labels.iloc[order[:k]] = True
    return labels


def assign_top20(values: pd.Series) -> pd.Series:
    n = len(values)
    k = max(1, int(np.ceil(n * 0.20)))
    order = np.argsort(values.to_numpy(dtype=float), kind="stable")
    labels = pd.Series(index=values.index, data=False)
    labels.iloc[order[-k:]] = True
    return labels


def summarize_by_seed(df: pd.DataFrame, group_name: str) -> pd.DataFrame:
    rows = []
    for (seed, comparison), comp_df in df.groupby(["seed", "comparison"], sort=True):
        for metric in TARGET_METRICS:
            values = comp_df[f"delta_{metric}"].to_numpy(dtype=float)
            rows.append(
                {
                    "seed": int(seed),
                    "group_name": group_name,
                    "comparison": comparison,
                    "metric": metric,
                    "count": int(len(values)),
                    "mean": float(np.mean(values)) if len(values) else np.nan,
                    "std": float(np.std(values, ddof=1)) if len(values) > 1 else np.nan,
                }
            )
    return pd.DataFrame(rows)


def summarize_across_seeds(df: pd.DataFrame, group_name: str) -> pd.DataFrame:
    rows = []
    for (comparison, metric), comp_df in df.groupby(["comparison", "metric"], sort=True):
        seed_means = comp_df["mean"].to_numpy(dtype=float)
        rows.append(
            {
                "group_name": group_name,
                "comparison": comparison,
                "metric": metric,
                "mean_across_seeds": float(np.mean(seed_means)) if len(seed_means) else np.nan,
                "std_across_seeds": float(np.std(seed_means, ddof=1)) if len(seed_means) > 1 else np.nan,
                "n_seeds_used": int(len(seed_means)),
            }
        )
    return pd.DataFrame(rows)


def plot_metric(summary_df: pd.DataFrame, metric: str, title: str, out_path: Path):
    plot_df = summary_df[summary_df["metric"] == metric].copy()
    comparisons = list(COMPARISONS.keys())
    plot_df["comparison"] = pd.Categorical(plot_df["comparison"], categories=comparisons, ordered=True)
    plot_df = plot_df.sort_values("comparison")

    x = np.arange(len(plot_df))
    means = plot_df["mean_across_seeds"].to_numpy(dtype=float)
    stds = plot_df["std_across_seeds"].fillna(0.0).to_numpy(dtype=float)

    plt.figure(figsize=(10, 5))
    plt.bar(x, means, yerr=stds, capsize=4)
    plt.xticks(x, plot_df["comparison"], rotation=20, ha="right")
    plt.ylabel(metric)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def save_outputs(
    output_dir: Path,
    slices_df: pd.DataFrame,
    by_seed_df: pd.DataFrame,
    across_df: pd.DataFrame,
    slices_name: str,
    by_seed_name: str,
    across_name: str,
    json_name: str,
    dice_plot_name: str,
    hd95_plot_name: str,
    title_suffix: str,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    slices_path = output_dir / slices_name
    by_seed_path = output_dir / by_seed_name
    across_path = output_dir / across_name
    json_path = output_dir / json_name

    slices_df.to_csv(slices_path, index=False, encoding="utf-8")
    by_seed_df.to_csv(by_seed_path, index=False, encoding="utf-8")
    across_df.to_csv(across_path, index=False, encoding="utf-8")
    json_path.write_text(across_df.to_json(orient="records", force_ascii=False, indent=2), encoding="utf-8")

    plot_metric(across_df, "dice", f"delta_dice {title_suffix}", output_dir / dice_plot_name)
    plot_metric(across_df, "hd95", f"delta_hd95 {title_suffix}", output_dir / hd95_plot_name)


def main():
    args = parse_args()
    results_root = Path(args.results_root)
    summary_size_dir = results_root / "summary_size_slicelevel_bottom20"
    summary_complexity_dir = results_root / "summary_complexity_slicelevel_top20"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    size_slice_rows = []
    complexity_slice_rows = []

    for dataset_seed in range(args.seed_start, args.seed_end + 1):
        print(f"processing seed={dataset_seed}")
        configs = {}
        datasets = {}
        test_patients_map = {}
        models = {}

        for key, model_name in MODEL_NAMES.items():
            config, ckpt_path = load_config(results_root, model_name, dataset_seed)
            dataset = build_dataset_for_config(config)
            split = build_patient_split(dataset, seed=config.split_seed, train_frac=config.train_frac, val_frac=config.val_frac)
            configs[key] = config
            datasets[key] = dataset
            test_patients_map[key] = split["test_patients"]
            models[key] = load_saved_model(ckpt_path, input_mode=config.input_mode, device=device)

        reference_patients = test_patients_map["2d_bce_dice"]
        for key, test_patients in test_patients_map.items():
            if test_patients != reference_patients:
                raise RuntimeError(f"Test patient mismatch for seed={dataset_seed}: {key}")

        patient_dirs = patient_dir_map(datasets["25d_boundary"])
        seed_slice_rows = []

        for patient_id in reference_patients:
            flair_n, gt = load_patient_volume(patient_dirs[patient_id])
            pred_volumes = {}
            for key, config in configs.items():
                pred_volumes[key] = predict_volume(models[key], flair_n, config.input_mode, args.threshold, device, args.batch_size)

            for t in range(gt.shape[-1]):
                gt_slice = gt[:, :, t]
                gt_sum = int(gt_slice.sum())
                if gt_sum == 0:
                    continue

                row = {
                    "seed": dataset_seed,
                    "patient_id": patient_id,
                    "slice_index": int(t),
                    "tumor_vox": gt_sum,
                    "total_vox": int(gt_slice.size),
                    "size_ratio": float(gt_sum / max(int(gt_slice.size), 1)),
                    "complexity": complexity_2d(gt_slice),
                }

                for key, pred_volume in pred_volumes.items():
                    pred_slice = pred_volume[:, :, t]
                    row[f"{key}_pred_nonempty"] = bool(pred_slice.sum() > 0)
                    row[f"{key}_dice"] = dice_2d_binary(pred_slice, gt_slice)
                    row[f"{key}_hd95"] = hd95_2d(pred_slice, gt_slice)

                seed_slice_rows.append(row)

        seed_df = pd.DataFrame(seed_slice_rows).sort_values(["patient_id", "slice_index"]).reset_index(drop=True)
        seed_df["is_size_bottom20"] = assign_bottom20(seed_df["size_ratio"])
        seed_df["is_complexity_top20"] = assign_top20(seed_df["complexity"])

        for comparison, (lhs_key, rhs_key) in COMPARISONS.items():
            joint_nonempty = seed_df[f"{lhs_key}_pred_nonempty"] & seed_df[f"{rhs_key}_pred_nonempty"]

            size_df = seed_df[seed_df["is_size_bottom20"] & joint_nonempty].copy()
            if not size_df.empty:
                size_df["comparison"] = comparison
                size_df["lhs_model"] = lhs_key
                size_df["rhs_model"] = rhs_key
                size_df["lhs_dice"] = size_df[f"{lhs_key}_dice"]
                size_df["rhs_dice"] = size_df[f"{rhs_key}_dice"]
                size_df["lhs_hd95"] = size_df[f"{lhs_key}_hd95"]
                size_df["rhs_hd95"] = size_df[f"{rhs_key}_hd95"]
                size_df["delta_dice"] = size_df["lhs_dice"] - size_df["rhs_dice"]
                size_df["delta_hd95"] = size_df["lhs_hd95"] - size_df["rhs_hd95"]
                size_slice_rows.extend(
                    size_df[
                        [
                            "seed",
                            "patient_id",
                            "slice_index",
                            "tumor_vox",
                            "total_vox",
                            "size_ratio",
                            "comparison",
                            "lhs_model",
                            "rhs_model",
                            "lhs_dice",
                            "rhs_dice",
                            "lhs_hd95",
                            "rhs_hd95",
                            "delta_dice",
                            "delta_hd95",
                        ]
                    ].to_dict("records")
                )

            complexity_df = seed_df[seed_df["is_complexity_top20"] & joint_nonempty].copy()
            if not complexity_df.empty:
                complexity_df["comparison"] = comparison
                complexity_df["lhs_model"] = lhs_key
                complexity_df["rhs_model"] = rhs_key
                complexity_df["lhs_dice"] = complexity_df[f"{lhs_key}_dice"]
                complexity_df["rhs_dice"] = complexity_df[f"{rhs_key}_dice"]
                complexity_df["lhs_hd95"] = complexity_df[f"{lhs_key}_hd95"]
                complexity_df["rhs_hd95"] = complexity_df[f"{rhs_key}_hd95"]
                complexity_df["delta_dice"] = complexity_df["lhs_dice"] - complexity_df["rhs_dice"]
                complexity_df["delta_hd95"] = complexity_df["lhs_hd95"] - complexity_df["rhs_hd95"]
                complexity_slice_rows.extend(
                    complexity_df[
                        [
                            "seed",
                            "patient_id",
                            "slice_index",
                            "tumor_vox",
                            "total_vox",
                            "complexity",
                            "comparison",
                            "lhs_model",
                            "rhs_model",
                            "lhs_dice",
                            "rhs_dice",
                            "lhs_hd95",
                            "rhs_hd95",
                            "delta_dice",
                            "delta_hd95",
                        ]
                    ].to_dict("records")
                )

    size_slice_df = pd.DataFrame(size_slice_rows).sort_values(["seed", "comparison", "patient_id", "slice_index"]).reset_index(drop=True)
    complexity_slice_df = pd.DataFrame(complexity_slice_rows).sort_values(["seed", "comparison", "patient_id", "slice_index"]).reset_index(drop=True)

    size_by_seed_df = summarize_by_seed(size_slice_df, "size_bottom20pct")
    complexity_by_seed_df = summarize_by_seed(complexity_slice_df, "complexity_top20pct")
    size_across_df = summarize_across_seeds(size_by_seed_df, "size_bottom20pct")
    complexity_across_df = summarize_across_seeds(complexity_by_seed_df, "complexity_top20pct")

    save_outputs(
        summary_size_dir,
        size_slice_df,
        size_by_seed_df,
        size_across_df,
        "slices_size_bottom20_by_seed.csv",
        "grouped_delta_by_seed_size_bottom20.csv",
        "grouped_delta_across_seeds_size_bottom20.csv",
        "grouped_delta_across_seeds_size_bottom20.json",
        "delta_dice_by_size_bottom20.png",
        "delta_hd95_by_size_bottom20.png",
        "(size bottom20, slice-level)",
    )
    save_outputs(
        summary_complexity_dir,
        complexity_slice_df,
        complexity_by_seed_df,
        complexity_across_df,
        "slices_complexity_top20_by_seed.csv",
        "grouped_delta_by_seed_complexity_top20.csv",
        "grouped_delta_across_seeds_complexity_top20.csv",
        "grouped_delta_across_seeds_complexity_top20.json",
        "delta_dice_by_complexity_top20.png",
        "delta_hd95_by_complexity_top20.png",
        "(complexity top20, slice-level)",
    )

    print("saved:")
    print(summary_size_dir)
    print(summary_complexity_dir)


if __name__ == "__main__":
    main()
