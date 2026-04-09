import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_SUMMARY_ROOT = Path("result_500") / "summary_0.5"
DEFAULT_CC500_CSV = DEFAULT_SUMMARY_ROOT / "results_3D_cc500" / "patient_metrics_3d_cc500_by_seed.csv"
DEFAULT_SIZE_CSV = Path("result_500") / "summary_size" / "patients_size_by_seed.csv"
DEFAULT_OUTPUT_DIR = DEFAULT_SUMMARY_ROOT / "results_final"
MODEL_ORDER = ["2d_bce_dice", "2d_boundary", "25d_bce_dice", "25d_boundary"]
GROUP_ORDER = ["low_30pct", "mid_40pct", "high_30pct"]


def parse_args():
    parser = argparse.ArgumentParser(description="Build pooled cc500 results grouped by tumor size 3/4/3.")
    parser.add_argument("--cc500-csv", default=str(DEFAULT_CC500_CSV))
    parser.add_argument("--size-csv", default=str(DEFAULT_SIZE_CSV))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args()


def fmt_3sf(value):
    if pd.isna(value):
        return ""
    return f"{float(value):.3g}"


def iqr(values: np.ndarray) -> float:
    q1 = np.quantile(values, 0.25)
    q3 = np.quantile(values, 0.75)
    return float(q3 - q1)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (size_group, model), group_df in df.groupby(["size_group", "model"], sort=False, observed=False):
        dice_vals = group_df["dice"].to_numpy(dtype=float)
        hd95_vals = group_df["hd95"].to_numpy(dtype=float)

        dice_mask = np.isfinite(dice_vals)
        hd95_mask = np.isfinite(hd95_vals)
        dice_used = dice_vals[dice_mask]
        hd95_used = hd95_vals[hd95_mask]

        rows.append(
            {
                "size_group": size_group,
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
    out_df["size_group"] = pd.Categorical(out_df["size_group"], categories=GROUP_ORDER, ordered=True)
    out_df["model"] = pd.Categorical(out_df["model"], categories=MODEL_ORDER, ordered=True)
    return out_df.sort_values(["size_group", "model"]).reset_index(drop=True)


def format_display(df: pd.DataFrame) -> pd.DataFrame:
    out_df = df.copy()
    for col in out_df.columns:
        if col in {"size_group", "model", "n_patients_total", "n_patients_used_for_dice", "n_patients_used_for_hd95"}:
            continue
        out_df[col] = out_df[col].map(fmt_3sf)
    return out_df


def main():
    args = parse_args()
    cc500_csv = Path(args.cc500_csv)
    size_csv = Path(args.size_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cc500_df = pd.read_csv(cc500_csv)
    size_df = pd.read_csv(size_csv, usecols=["seed", "patient_id", "size_group"])
    size_df = size_df.drop_duplicates(subset=["seed", "patient_id"])

    merged_df = cc500_df.merge(size_df, on=["seed", "patient_id"], how="inner")
    summary_df = summarize(merged_df)
    display_df = format_display(summary_df)

    out_path = output_dir / "pooled_results_cc500_by_size_30_40_30.csv"
    display_df.to_csv(out_path, index=False, encoding="utf-8")

    print("Saved:")
    print(out_path)


if __name__ == "__main__":
    main()
