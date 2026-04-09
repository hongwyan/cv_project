import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.ndimage import binary_dilation, binary_erosion, generate_binary_structure, label

from build_patientlevel_3d_results_cc50_result_500 import (
    MODEL_NAMES,
    MODEL_ORDER,
    build_dataset_for_config,
    dice_3d,
    fmt_3sf,
    hd95_3d,
    load_config,
    load_patient_volume,
    patient_dir_map,
    predict_volume,
)
from experiment_utils import load_saved_model
from splits import build_patient_split


DEFAULT_RESULTS_ROOT = Path("result_500")
DEFAULT_OUTPUT_DIR = DEFAULT_RESULTS_ROOT / "summary_0.5" / "results_final"
DEFAULT_THRESHOLD = 0.5
DEFAULT_BATCH_SIZE = 32
MIN_COMPONENT_SIZE = 300


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build pooled 3D Dice/HD95 results after cc300 removal, two rounds of dilation/erosion, and largest-component filtering."
    )
    parser.add_argument("--results-root", default=str(DEFAULT_RESULTS_ROOT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--seed-end", type=int, default=2)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--min-component-size", type=int, default=MIN_COMPONENT_SIZE)
    return parser.parse_args()


def remove_small_components_3d(pred: np.ndarray, min_size: int, structure: np.ndarray) -> np.ndarray:
    pred_bool = pred.astype(bool)
    if pred_bool.sum() == 0:
        return pred.astype(np.uint8)

    labeled, num_features = label(pred_bool, structure=structure)
    if num_features == 0:
        return np.zeros_like(pred, dtype=np.uint8)

    component_sizes = np.bincount(labeled.ravel())
    keep_labels = np.flatnonzero(component_sizes >= min_size)
    keep_labels = keep_labels[keep_labels != 0]
    if keep_labels.size == 0:
        return np.zeros_like(pred, dtype=np.uint8)

    return np.isin(labeled, keep_labels).astype(np.uint8)


def apply_morphology_twice(pred: np.ndarray, structure: np.ndarray) -> np.ndarray:
    pred_bool = pred.astype(bool)
    for _ in range(2):
        pred_bool = binary_dilation(pred_bool, structure=structure, iterations=1)
        pred_bool = binary_erosion(pred_bool, structure=structure, iterations=1)
    return pred_bool.astype(np.uint8)


def keep_largest_component_3d(pred: np.ndarray, structure: np.ndarray) -> np.ndarray:
    pred_bool = pred.astype(bool)
    if pred_bool.sum() == 0:
        return pred.astype(np.uint8)

    labeled, num_features = label(pred_bool, structure=structure)
    if num_features == 0:
        return np.zeros_like(pred, dtype=np.uint8)

    component_sizes = np.bincount(labeled.ravel())
    component_sizes[0] = 0
    largest_label = int(np.argmax(component_sizes))
    if largest_label == 0 or component_sizes[largest_label] == 0:
        return np.zeros_like(pred, dtype=np.uint8)
    return (labeled == largest_label).astype(np.uint8)


def postprocess_prediction(pred: np.ndarray, min_component_size: int, structure: np.ndarray) -> np.ndarray:
    pred = remove_small_components_3d(pred, min_component_size=min_component_size, structure=structure)
    pred = apply_morphology_twice(pred, structure=structure)
    pred = keep_largest_component_3d(pred, structure=structure)
    return pred.astype(np.uint8)


def metric_bundle(pred: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    gt_nonempty = bool(gt.sum() > 0)
    pred_nonempty = bool(pred.sum() > 0)
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
    }


def iqr(values: np.ndarray) -> float:
    q1 = np.quantile(values, 0.25)
    q3 = np.quantile(values, 0.75)
    return float(q3 - q1)


def build_pooled_summary(patient_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model_name, group_df in patient_df.groupby("model", sort=False, observed=False):
        dice_vals = group_df["dice"].to_numpy(dtype=float)
        hd95_vals = group_df["hd95"].to_numpy(dtype=float)
        dice_mask = np.isfinite(dice_vals)
        hd95_mask = np.isfinite(hd95_vals)
        dice_used = dice_vals[dice_mask]
        hd95_used = hd95_vals[hd95_mask]
        rows.append(
            {
                "model": model_name,
                "n_patients_total": int(len(group_df)),
                "n_patients_used_for_dice": int(dice_used.size),
                "dice_mean": float(np.mean(dice_used)) if dice_used.size else np.nan,
                "dice_std": float(np.std(dice_used, ddof=1)) if dice_used.size > 1 else np.nan,
                "n_patients_used_for_hd95": int(hd95_used.size),
                "hd95_median": float(np.median(hd95_used)) if hd95_used.size else np.nan,
                "hd95_iqr": iqr(hd95_used) if hd95_used.size else np.nan,
            }
        )

    out_df = pd.DataFrame(rows)
    out_df["model"] = pd.Categorical(out_df["model"], categories=MODEL_ORDER, ordered=True)
    return out_df.sort_values("model").reset_index(drop=True)


def format_display(df: pd.DataFrame) -> pd.DataFrame:
    out_df = df.copy()
    for col in out_df.columns:
        if col in {"model", "n_patients_total", "n_patients_used_for_dice", "n_patients_used_for_hd95"}:
            continue
        out_df[col] = out_df[col].map(fmt_3sf)
    return out_df


def main():
    args = parse_args()
    results_root = Path(args.results_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    structure = generate_binary_structure(rank=3, connectivity=3)

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
                pred_raw = predict_volume(models[key], flair_n, config.input_mode, args.threshold, device, args.batch_size)
                pred_pp = postprocess_prediction(pred_raw, min_component_size=args.min_component_size, structure=structure)
                metrics = metric_bundle(pred_pp, gt)
                patient_rows.append(
                    {
                        "seed": dataset_seed,
                        "patient_id": patient_id,
                        "model": key,
                        "gt_nonempty": metrics["gt_nonempty"],
                        "pred_nonempty": metrics["pred_nonempty"],
                        "dice": metrics["dice"],
                        "hd95": metrics["hd95"],
                    }
                )

    patient_df = pd.DataFrame(patient_rows)
    patient_df["model"] = pd.Categorical(patient_df["model"], categories=MODEL_ORDER, ordered=True)
    patient_df = patient_df.sort_values(["seed", "patient_id", "model"]).reset_index(drop=True)
    pooled_df = build_pooled_summary(patient_df)
    display_df = format_display(pooled_df)

    patient_csv = output_dir / "patient_metrics_cc300_morph2_lcc_pooled.csv"
    pooled_csv = output_dir / "pooled_results_cc300_morph2_lcc.csv"
    shape_json = output_dir / "patient_volume_shapes_cc300_morph2_lcc.json"

    patient_df.to_csv(patient_csv, index=False, encoding="utf-8")
    display_df.to_csv(pooled_csv, index=False, encoding="utf-8")
    shape_json.write_text(
        pd.Series({"observed_shapes": [list(s) for s in sorted(observed_shapes)]}).to_json(force_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("Saved:")
    print(patient_csv)
    print(pooled_csv)
    print(shape_json)


if __name__ == "__main__":
    main()
