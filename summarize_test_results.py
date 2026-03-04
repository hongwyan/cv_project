import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RESULTS_DIR = Path("results")
OUT_DIR = RESULTS_DIR / "summary"
MODEL_ORDER = [
    "2d_baseline",
    "2d_weightedBCE",
    "2d_boundary_combo",
    "25d_baseline",
    "25d_weightedBCE",
    "25d_boundary_combo",
]


def parse_name_seed(exp_dir_name: str):
    if "_seed" not in exp_dir_name:
        return None, None
    model_name, seed_str = exp_dir_name.rsplit("_seed", 1)
    if not seed_str.isdigit():
        return None, None
    return model_name, int(seed_str)


def load_records(results_dir: Path):
    records = []
    for exp_dir in sorted(results_dir.iterdir()):
        if not exp_dir.is_dir():
            continue
        model_name, seed = parse_name_seed(exp_dir.name)
        if model_name is None:
            continue

        metrics_path = exp_dir / f"{exp_dir.name}_metrics_test.json"
        if not metrics_path.exists():
            continue

        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        row = {"model": model_name, "seed": seed}
        row.update(metrics)
        records.append(row)

    if not records:
        raise RuntimeError(f"No metrics files found under: {results_dir}")
    return pd.DataFrame(records)


def build_summary(df: pd.DataFrame):
    metric_cols = [c for c in df.columns if c not in ("model", "seed")]
    agg = df.groupby("model")[metric_cols].agg(["mean", "var"])
    agg.columns = [f"{m}_{stat}" for m, stat in agg.columns]
    agg = agg.reset_index()
    return agg


def plot_metric_bar(summary_df: pd.DataFrame, metric: str, y_label: str, title: str, out_path: Path, ylim=None):
    x = np.arange(len(summary_df))
    means = summary_df[f"{metric}_mean"].to_numpy(dtype=float)
    vars_ = summary_df[f"{metric}_var"].fillna(0.0).to_numpy(dtype=float)
    stds = np.sqrt(np.clip(vars_, 0.0, None))

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
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_records(RESULTS_DIR)
    df = df.sort_values(["model", "seed"]).reset_index(drop=True)

    summary_df = build_summary(df)
    summary_df["model"] = pd.Categorical(summary_df["model"], categories=MODEL_ORDER, ordered=True)
    summary_df = summary_df.sort_values("model").reset_index(drop=True)

    per_seed_csv = OUT_DIR / "test_metrics_per_seed.csv"
    summary_csv = OUT_DIR / "test_metrics_mean_var_by_model.csv"
    summary_json = OUT_DIR / "test_metrics_mean_var_by_model.json"
    dice_plot_path = OUT_DIR / "test_dice_bar_mean_std.png"
    hd95_plot_path = OUT_DIR / "test_hd95_median_bar_mean_std.png"
    fpr_plot_path = OUT_DIR / "test_fpr_bar_mean_std.png"
    fnr_plot_path = OUT_DIR / "test_fnr_bar_mean_std.png"

    df.to_csv(per_seed_csv, index=False, encoding="utf-8")
    summary_df.to_csv(summary_csv, index=False, encoding="utf-8")
    summary_df.to_json(summary_json, orient="records", force_ascii=False, indent=2)
    plot_metric_bar(
        summary_df,
        metric="dice",
        y_label="Test Dice (mean across seeds)",
        title="6 Models: Test Dice Mean ± Std (seed=0..4)",
        out_path=dice_plot_path,
        ylim=(0.0, 1.0),
    )
    plot_metric_bar(
        summary_df,
        metric="hd95_median",
        y_label="Test HD95 Median (mean across seeds)",
        title="6 Models: Test HD95 Median Mean ± Std (seed=0..4)",
        out_path=hd95_plot_path,
    )
    plot_metric_bar(
        summary_df,
        metric="fpr",
        y_label="Test FPR (mean across seeds)",
        title="6 Models: Test FPR Mean ± Std (seed=0..4)",
        out_path=fpr_plot_path,
        ylim=(0.0, 1.0),
    )
    plot_metric_bar(
        summary_df,
        metric="fnr",
        y_label="Test FNR (mean across seeds)",
        title="6 Models: Test FNR Mean ± Std (seed=0..4)",
        out_path=fnr_plot_path,
        ylim=(0.0, 1.0),
    )

    print("Saved:")
    print(per_seed_csv)
    print(summary_csv)
    print(summary_json)
    print(dice_plot_path)
    print(hd95_plot_path)
    print(fpr_plot_path)
    print(fnr_plot_path)
    print("\nDice mean/var by model:")
    for _, row in summary_df.iterrows():
        print(f"  {row['model']}: mean={row['dice_mean']:.4f}, var={row['dice_var']:.6f}")


if __name__ == "__main__":
    main()
