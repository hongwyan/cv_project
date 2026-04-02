import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_BASE_DIR = Path("result_new")
DEFAULT_OUTPUT_DIR = DEFAULT_BASE_DIR / "summary"
MODEL_ORDER = [
    "2d_dice",
    "2d_bce_dice",
    "2d_bce_dice_posweighted",
    "2d_bce_dice_boundary_0.05",
    "2d_bce_dice_boundary_0.1",
    "2d_bce_dice_boundary_0.2",
    "2d_bce_dice_posweighted_boundary_0.05",
    "2d_bce_dice_posweighted_boundary_0.1",
    "2d_bce_dice_posweighted_boundary_0.2",
    "25d_dice",
    "25d_bce_dice",
    "25d_bce_dice_posweighted",
    "25d_bce_dice_boundary_0.01",
    "25d_bce_dice_boundary_0.05",
    "25d_bce_dice_boundary_0.1",
    "25d_bce_dice_boundary_0.2",
    "25d_bce_dice_posweighted_boundary_0.05",
    "25d_bce_dice_posweighted_boundary_0.1",
    "25d_bce_dice_posweighted_boundary_0.2",
]
TARGET_METRICS = ["dice", "hd95_median", "fnr", "fpr"]


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize test metrics across seeds.")
    parser.add_argument("--base-dir", default=str(DEFAULT_BASE_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--artifact-suffix", default="", help="Only include experiment dirs ending with this suffix.")
    parser.add_argument(
        "--metrics-file-suffix",
        default="",
        help="Optional suffix appended to metrics_test filename, e.g. '_0.5' -> *_metrics_test_0.5.json",
    )
    return parser.parse_args()


def parse_name_seed(exp_dir_name: str):
    match = re.match(r"^(?P<model>.+)_seed(?P<seed>\d+)(?:_(?P<suffix>.+))?$", exp_dir_name)
    if not match:
        return None, None, None
    return match.group("model"), int(match.group("seed")), match.group("suffix") or ""


def load_records(results_dir: Path, artifact_suffix: str, metrics_file_suffix: str):
    records = []
    for exp_dir in sorted(results_dir.iterdir()):
        if not exp_dir.is_dir():
            continue
        model_name, seed, suffix = parse_name_seed(exp_dir.name)
        if model_name is None:
            continue
        if artifact_suffix and suffix != artifact_suffix:
            continue
        if not artifact_suffix and suffix:
            continue

        metrics_path = exp_dir / f"{exp_dir.name}_metrics_test{metrics_file_suffix}.json"
        if not metrics_path.exists():
            continue

        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        row = {"model": model_name, "seed": seed}
        for metric in TARGET_METRICS:
            row[metric] = metrics.get(metric, np.nan)
        records.append(row)

    if not records:
        raise RuntimeError(f"No metrics files found under: {results_dir} with suffix '{artifact_suffix}'")
    return pd.DataFrame(records)


def build_summary(df: pd.DataFrame):
    agg = df.groupby("model")[TARGET_METRICS].agg(["mean", "std"])
    agg.columns = [f"{metric}_{stat}" for metric, stat in agg.columns]
    agg = agg.reset_index()
    present_models = [m for m in MODEL_ORDER if m in agg["model"].tolist()]
    agg["model"] = pd.Categorical(agg["model"], categories=present_models, ordered=True)
    agg = agg.sort_values("model").reset_index(drop=True)
    return agg


def fmt_3sf(x):
    if pd.isna(x):
        return ""
    return f"{float(x):.3g}"


def plot_metric_bar(summary_df: pd.DataFrame, metric: str, y_label: str, title: str, out_path: Path, ylim=None):
    x = np.arange(len(summary_df))
    means = summary_df[f"{metric}_mean"].to_numpy(dtype=float)
    stds = summary_df[f"{metric}_std"].fillna(0.0).to_numpy(dtype=float)

    plt.figure(figsize=(10, 5))
    plt.bar(x, means, yerr=stds, capsize=4)
    plt.xticks(x, summary_df["model"], rotation=25, ha="right")
    plt.ylabel(y_label)
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    args = parse_args()
    base_dir = Path(args.base_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_records(base_dir, args.artifact_suffix, args.metrics_file_suffix)
    df = df.sort_values(["model", "seed"]).reset_index(drop=True)
    summary_df = build_summary(df)

    suffix_part = f"_{args.artifact_suffix}" if args.artifact_suffix else ""
    per_seed_csv = output_dir / "test_metrics_per_seed.csv"
    summary_csv = output_dir / "test_metrics_mean_std_by_model.csv"
    summary_json = output_dir / "test_metrics_mean_std_by_model.json"
    summary_3sf_csv = output_dir / "test_metrics_mean_std_by_model_3sf.csv"
    summary_3sf_md = output_dir / "test_metrics_mean_std_by_model_3sf.md"
    dice_plot_path = output_dir / "test_dice_bar_mean_std.png"
    hd95_plot_path = output_dir / "test_hd95_median_bar_mean_std.png"
    fpr_plot_path = output_dir / "test_fpr_bar_mean_std.png"
    fnr_plot_path = output_dir / "test_fnr_bar_mean_std.png"

    df.to_csv(per_seed_csv, index=False, encoding="utf-8")
    summary_df.to_csv(summary_csv, index=False, encoding="utf-8")
    summary_df.to_json(summary_json, orient="records", force_ascii=False, indent=2)
    summary_3sf = summary_df.copy()
    for col in [c for c in summary_3sf.columns if c != "model"]:
        summary_3sf[col] = summary_3sf[col].map(fmt_3sf)
    summary_3sf.to_csv(summary_3sf_csv, index=False, encoding="utf-8")
    md_lines = [
        "| Model | Dice mean | Dice std | HD95 median mean | HD95 median std | FNR mean | FNR std | FPR mean | FPR std |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for _, row in summary_3sf.iterrows():
        md_lines.append(
            f"| {row['model']} | {row['dice_mean']} | {row['dice_std']} | "
            f"{row['hd95_median_mean']} | {row['hd95_median_std']} | "
            f"{row['fnr_mean']} | {row['fnr_std']} | {row['fpr_mean']} | {row['fpr_std']} |"
        )
    summary_3sf_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    plot_metric_bar(
        summary_df,
        metric="dice",
        y_label="Test Dice (mean across seeds)",
        title=f"Test Dice Mean ± Std{suffix_part}",
        out_path=dice_plot_path,
        ylim=(0.0, 1.0),
    )
    plot_metric_bar(
        summary_df,
        metric="hd95_median",
        y_label="Test HD95 Median (mean across seeds)",
        title=f"Test HD95 Median Mean ± Std{suffix_part}",
        out_path=hd95_plot_path,
    )
    plot_metric_bar(
        summary_df,
        metric="fpr",
        y_label="Test FPR (mean across seeds)",
        title=f"Test FPR Mean ± Std{suffix_part}",
        out_path=fpr_plot_path,
        ylim=(0.0, 1.0),
    )
    plot_metric_bar(
        summary_df,
        metric="fnr",
        y_label="Test FNR (mean across seeds)",
        title=f"Test FNR Mean ± Std{suffix_part}",
        out_path=fnr_plot_path,
        ylim=(0.0, 1.0),
    )

    print("Saved:")
    print(per_seed_csv)
    print(summary_csv)
    print(summary_json)
    print(summary_3sf_csv)
    print(summary_3sf_md)
    print(dice_plot_path)
    print(hd95_plot_path)
    print(fpr_plot_path)
    print(fnr_plot_path)


if __name__ == "__main__":
    main()
