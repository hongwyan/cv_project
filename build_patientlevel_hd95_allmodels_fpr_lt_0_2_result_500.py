import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_RESULTS_3D_DIR = Path("result_500") / "summary_0.5" / "results_3D"
DEFAULT_INPUT_CSV = DEFAULT_RESULTS_3D_DIR / "patient_metrics_3d_by_seed.csv"
DEFAULT_FILTERED_CSV = DEFAULT_RESULTS_3D_DIR / "patient_hd95_allmodels_fpr_lt_0_2_by_seed.csv"
DEFAULT_SUMMARY_CSV = DEFAULT_RESULTS_3D_DIR / "patient_hd95_allmodels_fpr_lt_0_2_mean_std_by_seed.csv"
DEFAULT_SUMMARY_JSON = DEFAULT_RESULTS_3D_DIR / "patient_hd95_allmodels_fpr_lt_0_2_mean_std_by_seed.json"
DEFAULT_SUMMARY_MD = DEFAULT_RESULTS_3D_DIR / "patient_hd95_allmodels_fpr_lt_0_2_mean_std_by_seed_3sf.md"
MODEL_ORDER = ["2d_bce_dice", "2d_boundary", "25d_bce_dice", "25d_boundary"]
FPR_THRESHOLD = 0.20


def parse_args():
    parser = argparse.ArgumentParser(
        description="Summarize HD95 using patients whose FPR < 0.15 under all four models within each seed."
    )
    parser.add_argument("--input-csv", default=str(DEFAULT_INPUT_CSV))
    parser.add_argument("--filtered-csv", default=str(DEFAULT_FILTERED_CSV))
    parser.add_argument("--summary-csv", default=str(DEFAULT_SUMMARY_CSV))
    parser.add_argument("--summary-json", default=str(DEFAULT_SUMMARY_JSON))
    parser.add_argument("--summary-md", default=str(DEFAULT_SUMMARY_MD))
    return parser.parse_args()


def fmt_3dec(value):
    if pd.isna(value):
        return ""
    return f"{float(value):.3f}"


def round_float_columns(df: pd.DataFrame, skip_cols: set[str]) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if col in skip_cols:
            continue
        if pd.api.types.is_numeric_dtype(out[col]):
            out[col] = out[col].round(3)
    return out


def main():
    args = parse_args()
    input_csv = Path(args.input_csv)
    filtered_csv = Path(args.filtered_csv)
    summary_csv = Path(args.summary_csv)
    summary_json = Path(args.summary_json)
    summary_md = Path(args.summary_md)

    df = pd.read_csv(input_csv)
    required_cols = {"seed", "patient_id", "model", "fpr", "hd95"}
    missing = required_cols.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = df[df["model"].isin(MODEL_ORDER)].copy()
    pivot_fpr = df.pivot_table(index=["seed", "patient_id"], columns="model", values="fpr", aggfunc="first")
    valid_mask = pivot_fpr[MODEL_ORDER].lt(FPR_THRESHOLD).all(axis=1)
    valid_pairs = pivot_fpr.index[valid_mask]

    filtered_df = df.set_index(["seed", "patient_id"]).loc[valid_pairs].reset_index()
    filtered_df["model"] = pd.Categorical(filtered_df["model"], categories=MODEL_ORDER, ordered=True)
    filtered_df = filtered_df.sort_values(["seed", "patient_id", "model"]).reset_index(drop=True)

    summary_rows = []
    for (seed, model), group_df in filtered_df.groupby(["seed", "model"], sort=True):
        dice_vals = group_df["dice"].to_numpy(dtype=float)
        hd95_vals = group_df["hd95"].to_numpy(dtype=float)
        dice_mask = np.isfinite(dice_vals)
        finite_mask = np.isfinite(hd95_vals)
        finite_dice_vals = dice_vals[dice_mask]
        finite_vals = hd95_vals[finite_mask]
        summary_rows.append(
            {
                "seed": int(seed),
                "model": model,
                "n_patients_allmodels_fpr_lt_0_2": int(len(group_df)),
                "n_patients_used_for_dice": int(finite_dice_vals.size),
                "n_patients_used_for_hd95": int(finite_vals.size),
                "dice_mean": float(np.mean(finite_dice_vals)) if finite_dice_vals.size else np.nan,
                "hd95_mean": float(np.mean(finite_vals)) if finite_vals.size else np.nan,
                "hd95_std": float(np.std(finite_vals, ddof=1)) if finite_vals.size > 1 else np.nan,
                "hd95_median": float(np.median(finite_vals)) if finite_vals.size else np.nan,
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_df["model"] = pd.Categorical(summary_df["model"], categories=MODEL_ORDER, ordered=True)
    summary_df = summary_df.sort_values(["seed", "model"]).reset_index(drop=True)

    filtered_df_rounded = round_float_columns(filtered_df, {"seed", "patient_id"})
    summary_df_rounded = round_float_columns(
        summary_df,
        {"seed", "model", "n_patients_allmodels_fpr_lt_0_2", "n_patients_used_for_dice", "n_patients_used_for_hd95"},
    )

    filtered_csv.parent.mkdir(parents=True, exist_ok=True)
    filtered_df_rounded.to_csv(filtered_csv, index=False, encoding="utf-8", float_format="%.3f")
    summary_df_rounded.to_csv(summary_csv, index=False, encoding="utf-8", float_format="%.3f")
    summary_json.write_text(summary_df_rounded.to_json(orient="records", force_ascii=False, indent=2), encoding="utf-8")

    summary_3sf = summary_df_rounded.copy()
    for col in [c for c in summary_3sf.columns if c not in {"seed", "model", "n_patients_allmodels_fpr_lt_0_2", "n_patients_used_for_dice", "n_patients_used_for_hd95"}]:
        summary_3sf[col] = summary_3sf[col].map(fmt_3dec)

    lines = [
        "| seed | model | n_patients_allmodels_fpr_lt_0_2 | n_patients_used_for_dice | n_patients_used_for_hd95 | dice_mean | hd95_mean | hd95_std | hd95_median |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for _, row in summary_3sf.iterrows():
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["seed"]),
                    str(row["model"]),
                    str(row["n_patients_allmodels_fpr_lt_0_2"]),
                    str(row["n_patients_used_for_dice"]),
                    str(row["n_patients_used_for_hd95"]),
                    str(row["dice_mean"]),
                    str(row["hd95_mean"]),
                    str(row["hd95_std"]),
                    str(row["hd95_median"]),
                ]
            )
            + " |"
        )
    summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("Saved:")
    print(filtered_csv)
    print(summary_csv)
    print(summary_json)
    print(summary_md)


if __name__ == "__main__":
    main()
