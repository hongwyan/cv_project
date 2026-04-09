import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_RESULTS_ROOT = Path("result_500")
DEFAULT_RESULTS_3D_DIR = DEFAULT_RESULTS_ROOT / "summary_0.5" / "results_3D"
DEFAULT_INPUT_CSV = DEFAULT_RESULTS_3D_DIR / "patient_metrics_3d_by_seed.csv"


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize patient-level HD95 for patients with FPR < 0.1.")
    parser.add_argument("--input-csv", default=str(DEFAULT_INPUT_CSV))
    parser.add_argument("--output-dir", default=str(DEFAULT_RESULTS_3D_DIR))
    return parser.parse_args()


def fmt_3sf(value):
    if pd.isna(value):
        return ""
    return f"{float(value):.3g}"


def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (seed, model), group_df in df.groupby(["seed", "model"], sort=True):
        hd95_vals = group_df["hd95"].to_numpy(dtype=float)
        mask = np.isfinite(hd95_vals)
        valid = hd95_vals[mask]
        rows.append(
            {
                "seed": int(seed),
                "model": model,
                "n_patients_fpr_lt_0_1": int(len(group_df)),
                "hd95_mean": float(np.mean(valid)) if len(valid) else np.nan,
                "hd95_std": float(np.std(valid, ddof=1)) if len(valid) > 1 else np.nan,
                "hd95_median": float(np.median(valid)) if len(valid) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def main():
    args = parse_args()
    input_csv = Path(args.input_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    df = pd.read_csv(input_csv)
    required_cols = {"seed", "patient_id", "model", "fpr", "hd95"}
    missing = required_cols - set(df.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")

    filtered_df = df[(df["fpr"] < 0.1) & np.isfinite(df["hd95"])].copy()
    filtered_df = filtered_df.sort_values(["seed", "model", "patient_id"]).reset_index(drop=True)
    summary_df = build_summary(filtered_df).sort_values(["seed", "model"]).reset_index(drop=True)

    detail_csv = output_dir / "patient_hd95_fpr_lt_0_1_by_seed.csv"
    summary_csv = output_dir / "patient_hd95_fpr_lt_0_1_mean_std_by_seed.csv"
    summary_json = output_dir / "patient_hd95_fpr_lt_0_1_mean_std_by_seed.json"
    summary_md = output_dir / "patient_hd95_fpr_lt_0_1_mean_std_by_seed_3sf.md"

    filtered_df[["seed", "patient_id", "model", "fpr", "hd95"]].to_csv(detail_csv, index=False, encoding="utf-8")
    summary_df.to_csv(summary_csv, index=False, encoding="utf-8")
    summary_json.write_text(summary_df.to_json(orient="records", force_ascii=False, indent=2), encoding="utf-8")

    summary_3sf = summary_df.copy()
    for col in ["hd95_mean", "hd95_std", "hd95_median"]:
        summary_3sf[col] = summary_3sf[col].map(fmt_3sf)

    lines = [
        "| seed | model | n_patients_fpr_lt_0_1 | hd95_mean | hd95_std | hd95_median |",
        "|---|---|---|---|---|---|",
    ]
    for _, row in summary_3sf.iterrows():
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["seed"]),
                    str(row["model"]),
                    str(row["n_patients_fpr_lt_0_1"]),
                    str(row["hd95_mean"]),
                    str(row["hd95_std"]),
                    str(row["hd95_median"]),
                ]
            )
            + " |"
        )
    summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("Input:", input_csv)
    print("Saved:")
    print(detail_csv)
    print(summary_csv)
    print(summary_json)
    print(summary_md)


if __name__ == "__main__":
    main()
