import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon


DEFAULT_RESULTS_ROOT = Path("result_500")
DEFAULT_SUMMARY_ROOT = DEFAULT_RESULTS_ROOT / "summary_0.5"
DEFAULT_VALIDATION_DIR = DEFAULT_SUMMARY_ROOT / "validation"
PRIMARY_INPUT = DEFAULT_RESULTS_ROOT / "summary_size" / "patients_size_by_seed.csv"
FALLBACK_INPUT = DEFAULT_RESULTS_ROOT / "summary_complexity" / "patients_complexity_by_seed.csv"

COMPARISONS = {
    "2d_boundary_effect": ("2d_boundary", "2d_bce_dice"),
    "25d_boundary_effect": ("25d_boundary", "25d_bce_dice"),
    "25d_vs_2d_with_boundary": ("25d_boundary", "2d_boundary"),
}
METRICS = ["dice", "hd95", "fpr", "fnr"]
COMPARISON_ORDER = [
    ("2d_boundary", "2d_bce_dice"),
    ("25d_boundary", "25d_bce_dice"),
    ("25d_boundary", "2d_boundary"),
]
METRIC_ORDER = ["dice", "hd95", "fpr", "fnr"]


def parse_args():
    parser = argparse.ArgumentParser(description="Paired statistical validation for result_500.")
    parser.add_argument("--results-root", default=str(DEFAULT_RESULTS_ROOT))
    parser.add_argument("--input-csv", default="", help="Optional explicit patient-level metrics CSV.")
    parser.add_argument("--output-dir", default=str(DEFAULT_VALIDATION_DIR))
    return parser.parse_args()


def resolve_input_csv(results_root: Path, explicit_path: str) -> Path:
    if explicit_path:
        path = Path(explicit_path)
        if not path.exists():
            raise FileNotFoundError(f"Input CSV not found: {path}")
        return path

    primary = results_root / "summary_size" / "patients_size_by_seed.csv"
    fallback = results_root / "summary_complexity" / "patients_complexity_by_seed.csv"
    if primary.exists():
        return primary
    if fallback.exists():
        return fallback
    raise FileNotFoundError(
        "No patient-level CSV found. Expected one of:\n"
        f"- {primary}\n"
        f"- {fallback}"
    )


def safe_wilcoxon(lhs: np.ndarray, rhs: np.ndarray):
    try:
        stat, pvalue = wilcoxon(lhs, rhs)
        return float(stat), float(pvalue)
    except Exception:
        return np.nan, np.nan


def paired_tests(lhs: np.ndarray, rhs: np.ndarray):
    mask = np.isfinite(lhs) & np.isfinite(rhs)
    lhs = lhs[mask]
    rhs = rhs[mask]
    n_pairs = int(lhs.shape[0])

    if n_pairs < 2:
        return {
            "n_pairs_used": n_pairs,
            "mean_lhs": float(np.mean(lhs)) if n_pairs else np.nan,
            "mean_rhs": float(np.mean(rhs)) if n_pairs else np.nan,
            "mean_delta": float(np.mean(lhs - rhs)) if n_pairs else np.nan,
            "ttest_stat": np.nan,
            "ttest_pvalue": np.nan,
            "wilcoxon_stat": np.nan,
            "wilcoxon_pvalue": np.nan,
        }

    t_stat, t_pvalue = ttest_rel(lhs, rhs, nan_policy="omit")
    w_stat, w_pvalue = safe_wilcoxon(lhs, rhs)
    return {
        "n_pairs_used": n_pairs,
        "mean_lhs": float(np.mean(lhs)),
        "mean_rhs": float(np.mean(rhs)),
        "mean_delta": float(np.mean(lhs - rhs)),
        "ttest_stat": float(t_stat) if np.isfinite(t_stat) else np.nan,
        "ttest_pvalue": float(t_pvalue) if np.isfinite(t_pvalue) else np.nan,
        "wilcoxon_stat": w_stat,
        "wilcoxon_pvalue": w_pvalue,
    }


def fmt_3sf(value):
    if pd.isna(value):
        return ""
    return f"{float(value):.3g}"


def main():
    args = parse_args()
    results_root = Path(args.results_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    input_csv = resolve_input_csv(results_root, args.input_csv)
    df = pd.read_csv(input_csv)

    rows = []
    for seed, seed_df in df.groupby("seed", sort=True):
        seed_df = seed_df.sort_values("patient_id").reset_index(drop=True)
        for comparison, (lhs_model, rhs_model) in COMPARISONS.items():
            for metric in METRICS:
                lhs_col = f"{lhs_model}_{metric}"
                rhs_col = f"{rhs_model}_{metric}"
                if lhs_col not in seed_df.columns or rhs_col not in seed_df.columns:
                    raise KeyError(f"Missing columns for comparison {comparison}: {lhs_col}, {rhs_col}")

                lhs = seed_df[lhs_col].to_numpy(dtype=float)
                rhs = seed_df[rhs_col].to_numpy(dtype=float)
                stats = paired_tests(lhs, rhs)
                rows.append(
                    {
                        "seed": int(seed),
                        "comparison": comparison,
                        "metric": metric,
                        "lhs_model": lhs_model,
                        "rhs_model": rhs_model,
                        **stats,
                    }
                )

    out_df = pd.DataFrame(rows).sort_values(["seed", "comparison", "metric"]).reset_index(drop=True)
    pooled_rows = []
    for comparison, (lhs_model, rhs_model) in COMPARISONS.items():
        for metric in METRICS:
            lhs_col = f"{lhs_model}_{metric}"
            rhs_col = f"{rhs_model}_{metric}"
            if lhs_col not in df.columns or rhs_col not in df.columns:
                raise KeyError(f"Missing columns for pooled comparison {comparison}: {lhs_col}, {rhs_col}")

            lhs = df[lhs_col].to_numpy(dtype=float)
            rhs = df[rhs_col].to_numpy(dtype=float)
            stats = paired_tests(lhs, rhs)
            pooled_rows.append(
                {
                    "lhs_model": lhs_model,
                    "rhs_model": rhs_model,
                    "metric": metric,
                    **stats,
                }
            )

    pooled_df = pd.DataFrame(pooled_rows)

    display_df = out_df[
        [
            "seed",
            "lhs_model",
            "rhs_model",
            "metric",
            "ttest_pvalue",
            "wilcoxon_pvalue",
        ]
    ].copy()
    display_df["comparison_key"] = list(zip(display_df["lhs_model"], display_df["rhs_model"]))
    display_df["comparison_key"] = pd.Categorical(
        display_df["comparison_key"],
        categories=COMPARISON_ORDER,
        ordered=True,
    )
    display_df["metric"] = pd.Categorical(display_df["metric"], categories=METRIC_ORDER, ordered=True)
    display_df = display_df.sort_values(["comparison_key", "metric", "seed"]).reset_index(drop=True)
    display_df = display_df.drop(columns=["comparison_key"])
    for col in ["ttest_pvalue", "wilcoxon_pvalue"]:
        display_df[col] = display_df[col].map(fmt_3sf)

    pooled_display_df = pooled_df[
        [
            "lhs_model",
            "rhs_model",
            "metric",
            "ttest_pvalue",
            "wilcoxon_pvalue",
        ]
    ].copy()
    pooled_display_df["comparison_key"] = list(zip(pooled_display_df["lhs_model"], pooled_display_df["rhs_model"]))
    pooled_display_df["comparison_key"] = pd.Categorical(
        pooled_display_df["comparison_key"],
        categories=COMPARISON_ORDER,
        ordered=True,
    )
    pooled_display_df["metric"] = pd.Categorical(pooled_display_df["metric"], categories=METRIC_ORDER, ordered=True)
    pooled_display_df = pooled_display_df.sort_values(["comparison_key", "metric"]).reset_index(drop=True)
    pooled_display_df = pooled_display_df.drop(columns=["comparison_key"])
    for col in ["ttest_pvalue", "wilcoxon_pvalue"]:
        pooled_display_df[col] = pooled_display_df[col].map(fmt_3sf)

    csv_path = output_dir / "paired_tests_by_seed.csv"
    json_path = output_dir / "paired_tests_by_seed.json"
    md_path = output_dir / "paired_tests_by_seed_3sf.md"
    pooled_csv_path = output_dir / "paired_tests_pooled.csv"
    pooled_json_path = output_dir / "paired_tests_pooled.json"
    pooled_md_path = output_dir / "paired_tests_pooled_3sf.md"

    display_df.to_csv(csv_path, index=False, encoding="utf-8")
    json_path.write_text(display_df.to_json(orient="records", force_ascii=False, indent=2), encoding="utf-8")
    pooled_display_df.to_csv(pooled_csv_path, index=False, encoding="utf-8")
    pooled_json_path.write_text(pooled_display_df.to_json(orient="records", force_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "| seed | lhs_model | rhs_model | metric | ttest_pvalue | wilcoxon_pvalue |",
        "|---|---|---|---|---|---|",
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
                    str(row["ttest_pvalue"]),
                    str(row["wilcoxon_pvalue"]),
                ]
            )
            + " |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    pooled_lines = [
        "| lhs_model | rhs_model | metric | ttest_pvalue | wilcoxon_pvalue |",
        "|---|---|---|---|---|",
    ]
    for _, row in pooled_display_df.iterrows():
        pooled_lines.append(
            "| "
            + " | ".join(
                [
                    str(row["lhs_model"]),
                    str(row["rhs_model"]),
                    str(row["metric"]),
                    str(row["ttest_pvalue"]),
                    str(row["wilcoxon_pvalue"]),
                ]
            )
            + " |"
        )
    pooled_md_path.write_text("\n".join(pooled_lines) + "\n", encoding="utf-8")

    print("Input:", input_csv)
    print("Saved:")
    print(csv_path)
    print(json_path)
    print(md_path)
    print(pooled_csv_path)
    print(pooled_json_path)
    print(pooled_md_path)


if __name__ == "__main__":
    main()
