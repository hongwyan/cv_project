import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_SUMMARY_ROOT = Path("result_500") / "summary_0.5"
DEFAULT_INPUT_CSV = DEFAULT_SUMMARY_ROOT / "results_3D_cc500" / "patient_metrics_3d_cc500_by_seed.csv"
DEFAULT_OUTPUT_DIR = DEFAULT_SUMMARY_ROOT / "results_final"
MODEL_ORDER = ["2d_bce_dice", "2d_boundary", "25d_bce_dice", "25d_boundary"]


def parse_args():
    parser = argparse.ArgumentParser(description="Build pooled final table for cc500 postprocessed 3D results.")
    parser.add_argument("--input-csv", default=str(DEFAULT_INPUT_CSV))
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


def summarize_cc500(df: pd.DataFrame) -> pd.DataFrame:
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


def format_display(df: pd.DataFrame) -> pd.DataFrame:
    out_df = df.copy()
    for col in out_df.columns:
        if col in {"model", "n_patients_total", "n_patients_used_for_dice", "n_patients_used_for_hd95"}:
            continue
        out_df[col] = out_df[col].map(fmt_3sf)
    return out_df


def main():
    args = parse_args()
    input_csv = Path(args.input_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_csv)
    summary_df = summarize_cc500(df)
    display_df = format_display(summary_df)

    out_path = output_dir / "pooled_results_cc500.csv"
    display_df.to_csv(out_path, index=False, encoding="utf-8")
    print("Saved:")
    print(out_path)


if __name__ == "__main__":
    main()
