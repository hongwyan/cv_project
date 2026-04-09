import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


DEFAULT_SUMMARY_ROOT = Path("result_500") / "summary_0.5"
DEFAULT_INPUT_CSV = DEFAULT_SUMMARY_ROOT / "results_3D_cc200" / "patient_metrics_3d_cc200_by_seed.csv"
DEFAULT_OUTPUT_DIR = DEFAULT_SUMMARY_ROOT / "results_final"
COMPARISONS = [
    ("25d_boundary", "2d_boundary"),
    ("25d_boundary", "25d_bce_dice"),
]
METRICS = ["dice", "hd95"]


def parse_args():
    parser = argparse.ArgumentParser(description="Build pooled Wilcoxon test table for cc200 25d_boundary vs 2d_boundary.")
    parser.add_argument("--input-csv", default=str(DEFAULT_INPUT_CSV))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args()


def fmt_3sf(value):
    if pd.isna(value):
        return ""
    return f"{float(value):.3g}"


def safe_wilcoxon(lhs: np.ndarray, rhs: np.ndarray):
    try:
        stat, pvalue = wilcoxon(lhs, rhs)
        return float(stat), float(pvalue)
    except Exception:
        return np.nan, np.nan


def main():
    args = parse_args()
    input_csv = Path(args.input_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_csv)
    compared_models = sorted({model for pair in COMPARISONS for model in pair})
    df = df[df["model"].isin(compared_models)].copy()

    wide_df = df.pivot_table(
        index=["seed", "patient_id"],
        columns="model",
        values=METRICS,
        aggfunc="first",
    )
    wide_df.columns = [f"{model}_{metric}" for metric, model in wide_df.columns]
    wide_df = wide_df.reset_index()

    rows = []
    for lhs_model, rhs_model in COMPARISONS:
        for metric in METRICS:
            lhs_col = f"{lhs_model}_{metric}"
            rhs_col = f"{rhs_model}_{metric}"
            lhs_vals = wide_df[lhs_col].to_numpy(dtype=float)
            rhs_vals = wide_df[rhs_col].to_numpy(dtype=float)
            mask = np.isfinite(lhs_vals) & np.isfinite(rhs_vals)
            lhs_used = lhs_vals[mask]
            rhs_used = rhs_vals[mask]
            stat, pvalue = safe_wilcoxon(lhs_used, rhs_used)
            rows.append(
                {
                    "lhs_model": lhs_model,
                    "rhs_model": rhs_model,
                    "metric": metric,
                    "n_pairs_used": int(mask.sum()),
                    "wilcoxon_pvalue": pvalue,
                }
            )

    out_df = pd.DataFrame(rows)
    out_df["comparison_key"] = list(zip(out_df["lhs_model"], out_df["rhs_model"]))
    out_df["comparison_key"] = pd.Categorical(out_df["comparison_key"], categories=COMPARISONS, ordered=True)
    out_df["metric"] = pd.Categorical(out_df["metric"], categories=METRICS, ordered=True)
    out_df = out_df.sort_values(["comparison_key", "metric"]).reset_index(drop=True)
    out_df = out_df.drop(columns=["comparison_key"])
    display_df = out_df.copy()
    display_df["wilcoxon_pvalue"] = display_df["wilcoxon_pvalue"].map(fmt_3sf)

    csv_path = output_dir / "pooled_results_cc200_wilcoxon.csv"
    md_path = output_dir / "pooled_results_cc200_wilcoxon_3sf.md"

    display_df.to_csv(csv_path, index=False, encoding="utf-8")

    lines = [
        "| lhs_model | rhs_model | metric | n_pairs_used | wilcoxon_pvalue |",
        "|---|---|---|---|---|",
    ]
    for _, row in display_df.iterrows():
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["lhs_model"]),
                    str(row["rhs_model"]),
                    str(row["metric"]),
                    str(row["n_pairs_used"]),
                    str(row["wilcoxon_pvalue"]),
                ]
            )
            + " |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("Saved:")
    print(csv_path)
    print(md_path)


if __name__ == "__main__":
    main()
