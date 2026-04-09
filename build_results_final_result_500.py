import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_SUMMARY_ROOT = Path("result_500") / "summary_0.5"
DEFAULT_RESULTS_FINAL_DIR = DEFAULT_SUMMARY_ROOT / "results_final"
DEFAULT_ORIGINAL_CSV = DEFAULT_SUMMARY_ROOT / "results_3D" / "patient_metrics_3d_by_seed.csv"
DEFAULT_CC200_CSV = DEFAULT_SUMMARY_ROOT / "results_3D_cc200" / "patient_metrics_3d_cc200_by_seed.csv"
MODEL_ORDER = ["2d_bce_dice", "2d_boundary", "25d_bce_dice", "25d_boundary"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build pooled final tables across 3 seeds for original 3D results and cc200 postprocessed 3D results."
    )
    parser.add_argument("--original-csv", default=str(DEFAULT_ORIGINAL_CSV))
    parser.add_argument("--cc200-csv", default=str(DEFAULT_CC200_CSV))
    parser.add_argument("--output-dir", default=str(DEFAULT_RESULTS_FINAL_DIR))
    return parser.parse_args()


def fmt_3sf(value):
    if pd.isna(value):
        return ""
    return f"{float(value):.3g}"


def iqr(values: np.ndarray) -> float:
    q1 = np.quantile(values, 0.25)
    q3 = np.quantile(values, 0.75)
    return float(q3 - q1)


def summarize_original(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model, group_df in df.groupby("model", sort=False, observed=False):
        dice_vals = group_df["dice"].to_numpy(dtype=float)
        hd95_vals = group_df["hd95"].to_numpy(dtype=float)
        fnr_vals = group_df["fnr"].to_numpy(dtype=float)
        fpr_vals = group_df["fpr"].to_numpy(dtype=float)

        dice_mask = np.isfinite(dice_vals)
        hd95_mask = np.isfinite(hd95_vals)
        fnr_mask = np.isfinite(fnr_vals)
        fpr_mask = np.isfinite(fpr_vals)

        dice_used = dice_vals[dice_mask]
        hd95_used = hd95_vals[hd95_mask]
        fnr_used = fnr_vals[fnr_mask]
        fpr_used = fpr_vals[fpr_mask]

        rows.append(
            {
                "model": model,
                "n_patients_total": int(len(group_df)),
                "n_patients_used_for_dice": int(dice_used.size),
                "dice_mean": float(np.mean(dice_used)) if dice_used.size else np.nan,
                "dice_std": float(np.std(dice_used, ddof=1)) if dice_used.size > 1 else np.nan,
                "n_patients_used_for_hd95": int(hd95_used.size),
                "hd95_median": float(np.median(hd95_used)) if hd95_used.size else np.nan,
                "hd95_iqr": iqr(hd95_used) if hd95_used.size else np.nan,
                "n_patients_used_for_fnr": int(fnr_used.size),
                "fnr_mean": float(np.mean(fnr_used)) if fnr_used.size else np.nan,
                "fnr_std": float(np.std(fnr_used, ddof=1)) if fnr_used.size > 1 else np.nan,
                "n_patients_used_for_fpr": int(fpr_used.size),
                "fpr_mean": float(np.mean(fpr_used)) if fpr_used.size else np.nan,
                "fpr_std": float(np.std(fpr_used, ddof=1)) if fpr_used.size > 1 else np.nan,
            }
        )
    out_df = pd.DataFrame(rows)
    out_df["model"] = pd.Categorical(out_df["model"], categories=MODEL_ORDER, ordered=True)
    return out_df.sort_values("model").reset_index(drop=True)


def summarize_cc200(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model, group_df in df.groupby("model", sort=False, observed=False):
        dice_vals = group_df["dice"].to_numpy(dtype=float)
        hd95_vals = group_df["hd95"].to_numpy(dtype=float)

        dice_mask = np.isfinite(dice_vals)
        hd95_mask = np.isfinite(hd95_vals)

        dice_used = dice_vals[dice_mask]
        hd95_used = hd95_vals[hd95_mask]

        rows.append(
            {
                "model": model,
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


def format_display(df: pd.DataFrame, int_cols: set[str]) -> pd.DataFrame:
    out_df = df.copy()
    for col in out_df.columns:
        if col in int_cols or col == "model":
            continue
        out_df[col] = out_df[col].map(fmt_3sf)
    return out_df


def main():
    args = parse_args()
    original_csv = Path(args.original_csv)
    cc200_csv = Path(args.cc200_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    original_df = pd.read_csv(original_csv)
    cc200_df = pd.read_csv(cc200_csv)

    original_summary = summarize_original(original_df)
    cc200_summary = summarize_cc200(cc200_df)

    original_display = format_display(
        original_summary,
        {
            "n_patients_total",
            "n_patients_used_for_dice",
            "n_patients_used_for_hd95",
            "n_patients_used_for_fnr",
            "n_patients_used_for_fpr",
        },
    )
    cc200_display = format_display(
        cc200_summary,
        {"n_patients_total", "n_patients_used_for_dice", "n_patients_used_for_hd95"},
    )

    original_out = output_dir / "pooled_results_original.csv"
    cc200_out = output_dir / "pooled_results_cc200.csv"

    original_display.to_csv(original_out, index=False, encoding="utf-8")
    cc200_display.to_csv(cc200_out, index=False, encoding="utf-8")

    print("Saved:")
    print(original_out)
    print(cc200_out)


if __name__ == "__main__":
    main()
