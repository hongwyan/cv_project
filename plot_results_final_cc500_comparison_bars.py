import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_SUMMARY_ROOT = Path("result_500") / "summary_0.5"
DEFAULT_RESULTS_FINAL_DIR = DEFAULT_SUMMARY_ROOT / "results_final"
DEFAULT_VIS_DIR = DEFAULT_RESULTS_FINAL_DIR / "visualize"
DEFAULT_ORIGINAL_CSV = DEFAULT_RESULTS_FINAL_DIR / "pooled_results_original.csv"
DEFAULT_CC500_CSV = DEFAULT_RESULTS_FINAL_DIR / "pooled_results_cc500.csv"
DEFAULT_CC500_SIZE_CSV = DEFAULT_RESULTS_FINAL_DIR / "pooled_results_cc500_by_size_30_40_30.csv"
MODEL_ORDER = ["2d_bce_dice", "2d_boundary", "25d_bce_dice", "25d_boundary"]
MODEL_LABELS = {
    "2d_bce_dice": "2D BCE+Dice",
    "2d_boundary": "2D Boundary",
    "25d_bce_dice": "2.5D BCE+Dice",
    "25d_boundary": "2.5D Boundary",
}
SERIES = [
    ("Original", "#4e79a7"),
    ("Postprocessed", "#f28e2b"),
    ("Postprocessed Top30%", "#59a14f"),
]


def parse_args():
    parser = argparse.ArgumentParser(description="Plot Dice mean and HD95 median comparisons for original vs cc500.")
    parser.add_argument("--original-csv", default=str(DEFAULT_ORIGINAL_CSV))
    parser.add_argument("--cc500-csv", default=str(DEFAULT_CC500_CSV))
    parser.add_argument("--cc500-size-csv", default=str(DEFAULT_CC500_SIZE_CSV))
    parser.add_argument("--output-dir", default=str(DEFAULT_VIS_DIR))
    return parser.parse_args()


def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].astype(str).str.strip()
    return df


def load_value_map(original_csv: Path, cc500_csv: Path, cc500_size_csv: Path):
    original_df = clean_df(pd.read_csv(original_csv))
    cc500_df = clean_df(pd.read_csv(cc500_csv))
    cc500_size_df = clean_df(pd.read_csv(cc500_size_csv))

    top30_df = cc500_size_df[cc500_size_df["size_group"] == "high_30pct"].copy()

    value_map = {
        "Original": original_df.set_index("model")[["dice_mean", "hd95_median"]],
        "Postprocessed": cc500_df.set_index("model")[["dice_mean", "hd95_median"]],
        "Postprocessed Top30%": top30_df.set_index("model")[["dice_mean", "hd95_median"]],
    }
    return value_map


def plot_grouped_bars(metric: str, ylabel: str, title: str, out_path: Path, value_map: dict[str, pd.DataFrame], ylim=None):
    x = np.arange(len(MODEL_ORDER))
    width = 0.24

    plt.figure(figsize=(12, 7))
    for idx, (series_name, color) in enumerate(SERIES):
        series_df = value_map[series_name]
        values = [float(series_df.loc[model, metric]) for model in MODEL_ORDER]
        offset = (idx - 1) * width
        bars = plt.bar(x + offset, values, width=width, color=color, label=series_name)
        for bar, value in zip(bars, values):
            plt.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height(),
                f"{value:.3g}",
                ha="center",
                va="bottom",
                fontsize=11,
            )

    plt.xticks(x, [MODEL_LABELS[m] for m in MODEL_ORDER], rotation=18, ha="right", fontsize=12)
    plt.ylabel(ylabel, fontsize=13)
    plt.title(title, fontsize=16)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    value_map = load_value_map(Path(args.original_csv), Path(args.cc500_csv), Path(args.cc500_size_csv))

    dice_out = output_dir / "dice_mean_original_vs_cc500_vs_top30.png"
    hd95_out = output_dir / "hd95_median_original_vs_cc500_vs_top30.png"

    plot_grouped_bars(
        metric="dice_mean",
        ylabel="Dice Mean",
        title="Dice Mean: Original vs Postprocessed vs Top 30% Size",
        out_path=dice_out,
        value_map=value_map,
        ylim=(0.0, 1.0),
    )
    plot_grouped_bars(
        metric="hd95_median",
        ylabel="HD95 Median",
        title="HD95 Median: Original vs Postprocessed vs Top 30% Size",
        out_path=hd95_out,
        value_map=value_map,
    )

    print("Saved:")
    print(dice_out)
    print(hd95_out)


if __name__ == "__main__":
    main()
