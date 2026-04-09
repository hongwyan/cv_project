import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_RESULTS_3D_DIR = Path("result_500") / "summary_0.5" / "results_3D"
DEFAULT_INPUT_CSV = DEFAULT_RESULTS_3D_DIR / "patient_metrics_3d_by_seed.csv"
DEFAULT_OUTPUT_CSV = DEFAULT_RESULTS_3D_DIR / "patient_hd95_q20_by_seed.csv"
MODEL_ORDER = ["2d_bce_dice", "2d_boundary", "25d_bce_dice", "25d_boundary"]


def parse_args():
    parser = argparse.ArgumentParser(description="Build seed/model HD95 20th percentile summary from patient_metrics_3d_by_seed.csv.")
    parser.add_argument("--input-csv", default=str(DEFAULT_INPUT_CSV))
    parser.add_argument("--output-csv", default=str(DEFAULT_OUTPUT_CSV))
    return parser.parse_args()


def main():
    args = parse_args()
    input_csv = Path(args.input_csv)
    output_csv = Path(args.output_csv)

    df = pd.read_csv(input_csv)
    required_cols = {"seed", "model", "hd95"}
    missing = required_cols.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    rows = []
    for (seed, model), group_df in df.groupby(["seed", "model"], sort=True):
        hd95_vals = group_df["hd95"].to_numpy(dtype=float)
        finite_mask = np.isfinite(hd95_vals)
        finite_vals = hd95_vals[finite_mask]
        rows.append(
            {
                "seed": int(seed),
                "model": model,
                "n_patients_total": int(len(group_df)),
                "n_patients_used_for_hd95_q20": int(finite_vals.size),
                "hd95_q20": float(np.quantile(finite_vals, 0.2)) if finite_vals.size else np.nan,
            }
        )

    out_df = pd.DataFrame(rows)
    out_df["model"] = pd.Categorical(out_df["model"], categories=MODEL_ORDER, ordered=True)
    out_df = out_df.sort_values(["seed", "model"]).reset_index(drop=True)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_csv, index=False, encoding="utf-8")
    print(f"Saved: {output_csv}")


if __name__ == "__main__":
    main()
