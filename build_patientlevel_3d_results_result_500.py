import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.ndimage import binary_erosion, distance_transform_edt

from brats2d_dataset import BraTS2D
from brats2p5d_dataset import BraTS2p5D, _clamp_triplet, _load_nifti, _robust_zscore
from experiment_utils import ExperimentConfig, load_saved_model
from splits import build_patient_split


DEFAULT_RESULTS_ROOT = Path("result_500")
DEFAULT_SUMMARY_ROOT = DEFAULT_RESULTS_ROOT / "summary_0.5" / "results_3D"
DEFAULT_THRESHOLD = 0.5
DEFAULT_BATCH_SIZE = 32
MODEL_NAMES = {
    "2d_bce_dice": "2d_bce_dice",
    "2d_boundary": "2d_bce_dice_boundary_0.05",
    "25d_bce_dice": "25d_bce_dice",
    "25d_boundary": "25d_bce_dice_boundary_0.05",
}
MODEL_ORDER = ["2d_bce_dice", "2d_boundary", "25d_bce_dice", "25d_boundary"]


def parse_args():
    parser = argparse.ArgumentParser(description="Build patient-level 3D metrics for result_500 at threshold 0.5.")
    parser.add_argument("--results-root", default=str(DEFAULT_RESULTS_ROOT))
    parser.add_argument("--output-dir", default=str(DEFAULT_SUMMARY_ROOT))
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


def dice_3d(pred: np.ndarray, gt: np.ndarray) -> float:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    inter = np.logical_and(pred, gt).sum()
    return float((2.0 * inter) / (pred.sum() + gt.sum() + 1e-6))


def surface_voxels_3d(mask: np.ndarray) -> np.ndarray:
    mask = (mask > 0).astype(bool)
    if mask.sum() == 0:
        return mask
    eroded = binary_erosion(mask, iterations=1)
    return np.logical_xor(mask, eroded)


def hd95_3d(pred: np.ndarray, gt: np.ndarray) -> float:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    pred_surface = surface_voxels_3d(pred)
    gt_surface = surface_voxels_3d(gt)
    if pred_surface.sum() == 0 and gt_surface.sum() == 0:
        return 0.0
    if pred_surface.sum() == 0 or gt_surface.sum() == 0:
        return float("nan")

    dt_gt = distance_transform_edt(~gt_surface)
    dt_pred = distance_transform_edt(~pred_surface)
    d_pred_to_gt = dt_gt[pred_surface]
    d_gt_to_pred = dt_pred[gt_surface]
    distances = np.concatenate([d_pred_to_gt, d_gt_to_pred], axis=0)
    return float(np.percentile(distances, 95))


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


def metric_bundle(pred: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    gt_nonempty = bool(gt.sum() > 0)
    pred_nonempty = bool(pred.sum() > 0)
    fnr, fpr = patient_fnr_fpr(pred, gt)
    if gt_nonempty and pred_nonempty:
        dice = dice_3d(pred, gt)
        hd95 = hd95_3d(pred, gt)
    else:
        dice = np.nan
        hd95 = np.nan
    return {
        "gt_nonempty": gt_nonempty,
        "pred_nonempty": pred_nonempty,
        "dice": dice,
        "hd95": hd95,
        "fnr": fnr,
        "fpr": fpr,
    }


def fmt_3sf(value):
    if pd.isna(value):
        return ""
    return f"{float(value):.3g}"


def build_summary(patient_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (seed, model_name), group_df in patient_df.groupby(["seed", "model"], sort=True):
        dice_vals = group_df["dice"].to_numpy(dtype=float)
        hd95_vals = group_df["hd95"].to_numpy(dtype=float)
        fnr_vals = group_df["fnr"].to_numpy(dtype=float)
        fpr_vals = group_df["fpr"].to_numpy(dtype=float)

        dice_mask = np.isfinite(dice_vals)
        hd95_mask = np.isfinite(hd95_vals)

        rows.append(
            {
                "seed": int(seed),
                "model": model_name,
                "n_patients_total": int(len(group_df)),
                "n_patients_used_for_dice_hd95": int(np.isfinite(dice_vals).sum()),
                "dice_mean": float(np.mean(dice_vals[dice_mask])) if dice_mask.any() else np.nan,
                "dice_std": float(np.std(dice_vals[dice_mask], ddof=1)) if dice_mask.sum() > 1 else np.nan,
                "hd95_mean": float(np.mean(hd95_vals[hd95_mask])) if hd95_mask.any() else np.nan,
                "hd95_median": float(np.median(hd95_vals[hd95_mask])) if hd95_mask.any() else np.nan,
                "hd95_std": float(np.std(hd95_vals[hd95_mask], ddof=1)) if hd95_mask.sum() > 1 else np.nan,
                "fnr_mean": float(np.mean(fnr_vals)) if len(fnr_vals) else np.nan,
                "fnr_std": float(np.std(fnr_vals, ddof=1)) if len(fnr_vals) > 1 else np.nan,
                "fpr_mean": float(np.mean(fpr_vals)) if len(fpr_vals) else np.nan,
                "fpr_std": float(np.std(fpr_vals, ddof=1)) if len(fpr_vals) > 1 else np.nan,
            }
        )
    return pd.DataFrame(rows)


def main():
    args = parse_args()
    results_root = Path(args.results_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    patient_rows = []
    observed_shapes = set()

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
        for patient_id in reference_patients:
            flair_n, gt = load_patient_volume(patient_dirs[patient_id])
            observed_shapes.add(tuple(int(x) for x in gt.shape))
            for key, config in configs.items():
                pred = predict_volume(models[key], flair_n, config.input_mode, args.threshold, device, args.batch_size)
                metrics = metric_bundle(pred, gt)
                patient_rows.append(
                    {
                        "seed": dataset_seed,
                        "patient_id": patient_id,
                        "model": key,
                        "gt_nonempty": metrics["gt_nonempty"],
                        "pred_nonempty": metrics["pred_nonempty"],
                        "dice": metrics["dice"],
                        "hd95": metrics["hd95"],
                        "fnr": metrics["fnr"],
                        "fpr": metrics["fpr"],
                    }
                )

    patient_df = pd.DataFrame(patient_rows)
    patient_df["model"] = pd.Categorical(patient_df["model"], categories=MODEL_ORDER, ordered=True)
    patient_df = patient_df.sort_values(["seed", "patient_id", "model"]).reset_index(drop=True)
    summary_df = build_summary(patient_df)
    summary_df["model"] = pd.Categorical(summary_df["model"], categories=MODEL_ORDER, ordered=True)
    summary_df = summary_df.sort_values(["seed", "model"]).reset_index(drop=True)

    patient_csv = output_dir / "patient_metrics_3d_by_seed.csv"
    summary_csv = output_dir / "patient_metrics_3d_mean_std_by_seed.csv"
    summary_json = output_dir / "patient_metrics_3d_mean_std_by_seed.json"
    summary_md = output_dir / "patient_metrics_3d_mean_std_by_seed_3sf.md"
    shape_json = output_dir / "patient_volume_shapes.json"

    patient_df.to_csv(patient_csv, index=False, encoding="utf-8")
    summary_df.to_csv(summary_csv, index=False, encoding="utf-8")
    summary_json.write_text(summary_df.to_json(orient="records", force_ascii=False, indent=2), encoding="utf-8")
    shape_json.write_text(json.dumps({"observed_shapes": [list(s) for s in sorted(observed_shapes)]}, indent=2), encoding="utf-8")

    summary_3sf = summary_df.copy()
    for col in [c for c in summary_3sf.columns if c not in {"seed", "model", "n_patients_total", "n_patients_used_for_dice_hd95"}]:
        summary_3sf[col] = summary_3sf[col].map(fmt_3sf)
    lines = [
        "| seed | model | n_patients_total | n_patients_used_for_dice_hd95 | dice_mean | dice_std | hd95_mean | hd95_median | hd95_std | fnr_mean | fnr_std | fpr_mean | fpr_std |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for _, row in summary_3sf.iterrows():
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["seed"]),
                    str(row["model"]),
                    str(row["n_patients_total"]),
                    str(row["n_patients_used_for_dice_hd95"]),
                    str(row["dice_mean"]),
                    str(row["dice_std"]),
                    str(row["hd95_mean"]),
                    str(row["hd95_median"]),
                    str(row["hd95_std"]),
                    str(row["fnr_mean"]),
                    str(row["fnr_std"]),
                    str(row["fpr_mean"]),
                    str(row["fpr_std"]),
                ]
            )
            + " |"
        )
    summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("Observed patient volume shapes:", sorted(observed_shapes))
    print("Saved:")
    print(patient_csv)
    print(summary_csv)
    print(summary_json)
    print(summary_md)
    print(shape_json)


if __name__ == "__main__":
    main()
