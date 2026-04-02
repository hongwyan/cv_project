import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASE_DIR = Path("result_0.5")
BY_SEED_DIR = BASE_DIR / "by_seed"
TABLES_DIR = BASE_DIR / "tables"
PLOTS_DIR = BASE_DIR / "plots"
TOP_FRAC = 0.20
METRICS = ["delta_dice", "delta_hd95", "delta_fnr", "delta_fpr"]


def plot_summary(summary_df: pd.DataFrame, out_path: Path):
    labels = summary_df["metric"].tolist()
    means = summary_df["mean_across_seeds"].to_numpy(dtype=float)
    stds = summary_df["std_across_seeds"].fillna(0.0).to_numpy(dtype=float)

    x = np.arange(len(labels))
    plt.figure(figsize=(8, 4.5))
    plt.bar(x, means, yerr=stds, capsize=4)
    plt.xticks(x, labels, rotation=20, ha="right")
    plt.ylabel("25d - 2d")
    plt.title("Top 20% Complexity Delta Metrics")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    per_seed_rows = []
    for seed in range(5):
        path = BY_SEED_DIR / f"seed{seed}_patients_complexity.csv"
        df = pd.read_csv(path)
        df = df.sort_values("complexity", ascending=False, kind="stable").reset_index(drop=True)
        k = max(1, math.ceil(len(df) * TOP_FRAC))
        top = df.iloc[:k].copy()
        top["grouping_type"] = "complexity_top_20pct"
        top["group_name"] = "top_20pct"
        top.to_csv(TABLES_DIR / f"seed{seed}_patients_complexity_top20pct.csv", index=False, encoding="utf-8")

        for metric in METRICS:
            vals = top[metric].to_numpy(dtype=float)
            per_seed_rows.append(
                {
                    "seed": seed,
                    "grouping_type": "complexity_top_20pct",
                    "group_name": "top_20pct",
                    "metric": metric,
                    "count": int(len(vals)),
                    "mean": float(np.mean(vals)) if len(vals) else np.nan,
                    "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else np.nan,
                }
            )

    per_seed_df = pd.DataFrame(per_seed_rows)
    per_seed_path = TABLES_DIR / "grouped_summary_by_seed_top20pct_complexity.csv"
    per_seed_df.to_csv(per_seed_path, index=False, encoding="utf-8")

    summary_rows = []
    for metric, group in per_seed_df.groupby("metric"):
        means = group["mean"].to_numpy(dtype=float)
        summary_rows.append(
            {
                "grouping_type": "complexity_top_20pct",
                "group_name": "top_20pct",
                "metric": metric,
                "mean_across_seeds": float(np.mean(means)) if len(means) else np.nan,
                "std_across_seeds": float(np.std(means, ddof=1)) if len(means) > 1 else np.nan,
                "n_seeds_used": int(len(means)),
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_path = TABLES_DIR / "grouped_summary_across_seeds_top20pct_complexity.csv"
    summary_json_path = TABLES_DIR / "grouped_summary_across_seeds_top20pct_complexity.json"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8")
    summary_json_path.write_text(summary_df.to_json(orient="records", force_ascii=False, indent=2), encoding="utf-8")

    plot_path = PLOTS_DIR / "delta_metrics_top20pct_complexity.png"
    plot_summary(summary_df, plot_path)

    print("Saved:")
    print(per_seed_path)
    print(summary_path)
    print(summary_json_path)
    print(plot_path)


if __name__ == "__main__":
    main()
