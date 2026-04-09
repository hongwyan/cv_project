import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_RESULTS_ROOT = Path("result_500")
DEFAULT_RESULTS_3D_DIR = DEFAULT_RESULTS_ROOT / "summary_0.5" / "results_3D"
DEFAULT_INPUT_CSV = DEFAULT_RESULTS_3D_DIR / "patient_metrics_3d_by_seed.csv"
DEFAULT_OUTPUT_DIR = DEFAULT_RESULTS_3D_DIR / "validation"
MODEL_ORDER = ["2d_bce_dice", "2d_boundary", "25d_bce_dice", "25d_boundary"]
BINS = np.linspace(0.0, 1.0, 21)


def parse_args():
    parser = argparse.ArgumentParser(description="Plot patient-level FPR distributions for each seed.")
    parser.add_argument("--input-csv", default=str(DEFAULT_INPUT_CSV))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args()


def build_bin_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for seed in sorted(df["seed"].unique()):
        seed_df = df[df["seed"] == seed]
        for model in MODEL_ORDER:
            model_df = seed_df[seed_df["model"] == model]
            values = model_df["fpr"].to_numpy(dtype=float)
            counts, _ = np.histogram(values, bins=BINS)
            total = len(values)
            for idx, count in enumerate(counts):
                rows.append(
                    {
                        "seed": int(seed),
                        "model": model,
                        "bin_left": float(BINS[idx]),
                        "bin_right": float(BINS[idx + 1]),
                        "count": int(count),
                        "proportion": float(count / total) if total else np.nan,
                    }
                )
    return pd.DataFrame(rows)


def plot_seed(seed: int, df: pd.DataFrame, raw_df: pd.DataFrame, output_path: Path):
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharey=True)
    axes = axes.flatten()
    x = np.arange(len(BINS) - 1)
    labels = [f"{int(BINS[i]*100)}-{int(BINS[i+1]*100)}%" for i in range(len(BINS) - 1)]

    for ax, model in zip(axes, MODEL_ORDER):
        model_df = df[(df["seed"] == seed) & (df["model"] == model)].copy()
        model_df = model_df.sort_values("bin_left")
        proportions = model_df["proportion"].to_numpy(dtype=float)
        raw_model_df = raw_df[(raw_df["seed"] == seed) & (raw_df["model"] == model)].copy()
        nonzero_fpr_ratio = float((raw_model_df["fpr"].to_numpy(dtype=float) > 0).mean()) if len(raw_model_df) else np.nan
        ax.bar(x, proportions, width=0.85)
        ax.set_title(f"{model} | seed {seed}")
        ax.set_xlabel("FPR")
        ax.set_ylabel("Proportion of patients")
        tick_idx = x[::2]
        ax.set_xticks(tick_idx)
        ax.set_xticklabels([labels[i] for i in tick_idx], rotation=45, ha="right")
        ax.set_ylim(0.0, max(0.35, float(np.nanmax(proportions)) * 1.15 if len(proportions) else 0.35))
        ax.text(
            0.98,
            0.98,
            f"FPR>0: {nonzero_fpr_ratio:.1%}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.85, edgecolor="gray"),
        )

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main():
    args = parse_args()
    input_csv = Path(args.input_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    df = pd.read_csv(input_csv)
    required_cols = {"seed", "patient_id", "model", "fpr"}
    missing = required_cols - set(df.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")

    bin_df = build_bin_table(df)
    bin_csv = output_dir / "patient_fpr_distribution_bins.csv"
    bin_df.to_csv(bin_csv, index=False, encoding="utf-8")

    for seed in sorted(df["seed"].unique()):
        plot_seed(int(seed), bin_df, df, output_dir / f"patient_fpr_distribution_seed{int(seed)}.png")

    print("Saved:")
    print(bin_csv)
    for seed in sorted(df["seed"].unique()):
        print(output_dir / f"patient_fpr_distribution_seed{int(seed)}.png")


if __name__ == "__main__":
    main()
