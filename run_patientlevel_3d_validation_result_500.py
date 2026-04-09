import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


DEFAULT_RESULTS_ROOT = Path("result_500")
DEFAULT_RESULTS_3D_DIR = DEFAULT_RESULTS_ROOT / "summary_0.5" / "results_3D"
DEFAULT_VALIDATION_DIR = DEFAULT_RESULTS_3D_DIR / "validation"
DEFAULT_INPUT_CSV = DEFAULT_RESULTS_3D_DIR / "patient_metrics_3d_by_seed.csv"
COMPARISONS = {
    "2d_boundary_effect": ("2d_boundary", "2d_bce_dice"),
    "25d_boundary_effect": ("25d_boundary", "25d_bce_dice"),
    "25d_vs_2d_with_boundary": ("25d_boundary", "2d_boundary"),
}
COMPARISON_ORDER = [
    ("2d_boundary", "2d_bce_dice"),
    ("25d_boundary", "25d_bce_dice"),
    ("25d_boundary", "2d_boundary"),
]
METRIC_ORDER = ["dice", "hd95", "fpr", "fnr"]


def parse_args():
    parser = argparse.ArgumentParser(description="Wilcoxon validation for patient-level 3D result_500 metrics.")
    parser.add_argument("--input-csv", default=str(DEFAULT_INPUT_CSV))
    parser.add_argument("--output-dir", default=str(DEFAULT_VALIDATION_DIR))
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


def prepare_pairs(wide_df: pd.DataFrame, lhs_model: str, rhs_model: str, metric: str):
    lhs_col = f"{lhs_model}_{metric}"
    rhs_col = f"{rhs_model}_{metric}"
    lhs = wide_df[lhs_col].to_numpy(dtype=float)
    rhs = wide_df[rhs_col].to_numpy(dtype=float)
    mask = np.isfinite(lhs) & np.isfinite(rhs)
    return lhs[mask], rhs[mask]


def main():
    args = parse_args()
    input_csv = Path(args.input_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    long_df = pd.read_csv(input_csv)
    wide_df = long_df.pivot_table(
        index=["seed", "patient_id"],
        columns="model",
        values=["dice", "hd95", "fnr", "fpr"],
        aggfunc="first",
    )
    wide_df.columns = [f"{model}_{metric}" for metric, model in wide_df.columns]
    wide_df = wide_df.reset_index()

    rows = []
    for seed, seed_df in wide_df.groupby("seed", sort=True):
        seed_df = seed_df.sort_values("patient_id").reset_index(drop=True)
        for _, (lhs_model, rhs_model) in COMPARISONS.items():
            for metric in METRIC_ORDER:
                lhs, rhs = prepare_pairs(seed_df, lhs_model, rhs_model, metric)
                stat, pvalue = safe_wilcoxon(lhs, rhs)
                rows.append(
                    {
                        "seed": int(seed),
                        "lhs_model": lhs_model,
                        "rhs_model": rhs_model,
                        "metric": metric,
                        "wilcoxon_pvalue": pvalue,
                    }
                )

    out_df = pd.DataFrame(rows)
    out_df["comparison_key"] = list(zip(out_df["lhs_model"], out_df["rhs_model"]))
    out_df["comparison_key"] = pd.Categorical(out_df["comparison_key"], categories=COMPARISON_ORDER, ordered=True)
    out_df["metric"] = pd.Categorical(out_df["metric"], categories=METRIC_ORDER, ordered=True)
    out_df = out_df.sort_values(["comparison_key", "metric", "seed"]).reset_index(drop=True)
    out_df = out_df.drop(columns=["comparison_key"])

    display_df = out_df.copy()
    display_df["wilcoxon_pvalue"] = display_df["wilcoxon_pvalue"].map(fmt_3sf)

    csv_path = output_dir / "wilcoxon_by_seed.csv"
    json_path = output_dir / "wilcoxon_by_seed.json"
    md_path = output_dir / "wilcoxon_by_seed_3sf.md"

    display_df.to_csv(csv_path, index=False, encoding="utf-8")
    json_path.write_text(display_df.to_json(orient="records", force_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "| seed | lhs_model | rhs_model | metric | wilcoxon_pvalue |",
        "|---|---|---|---|---|",
    ]
    for _, row in display_df.iterrows():
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["seed"]),
                    str(row["lhs_model"]),
                    str(row["rhs_model"]),
                    str(row["metric"]),
                    str(row["wilcoxon_pvalue"]),
                ]
            )
            + " |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("Input:", input_csv)
    print("Saved:")
    print(csv_path)
    print(json_path)
    print(md_path)


if __name__ == "__main__":
    main()
