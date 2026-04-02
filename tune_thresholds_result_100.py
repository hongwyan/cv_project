import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from experiment_utils import (
    ExperimentConfig,
    build_dataset,
    build_loaders,
    load_saved_model,
)
from metrics_boundary import hd95_2d


DEFAULT_BASE_DIR = Path("result_new")
DEFAULT_SUMMARY_DIR = DEFAULT_BASE_DIR / "summary_dynthres"
THRESHOLDS = [0.5, 0.55, 0.6, 0.65, 0.7]
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
    "25d_bce_dice_boundary_0.05",
    "25d_bce_dice_boundary_0.1",
    "25d_bce_dice_boundary_0.2",
    "25d_bce_dice_posweighted_boundary_0.05",
    "25d_bce_dice_posweighted_boundary_0.1",
    "25d_bce_dice_posweighted_boundary_0.2",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Tune threshold on val and re-evaluate test.")
    parser.add_argument("--base-dir", default=str(DEFAULT_BASE_DIR))
    parser.add_argument("--summary-dir", default=str(DEFAULT_SUMMARY_DIR))
    parser.add_argument("--artifact-suffix", default="", help="Only process experiment dirs with this suffix.")
    return parser.parse_args()


def parse_exp_dir_name(name: str):
    match = re.match(r"^(?P<model>.+)_seed(?P<seed>\d+)(?:_(?P<suffix>.+))?$", name)
    if not match:
        return None, None, None
    return match.group("model"), int(match.group("seed")), match.group("suffix") or ""


def hard_dice_from_probs(probs: torch.Tensor, target: torch.Tensor, thr: float) -> torch.Tensor:
    preds = (probs > thr).float()
    preds = preds.view(preds.size(0), -1)
    target = target.view(target.size(0), -1)
    inter = (preds * target).sum(dim=1)
    union = preds.sum(dim=1) + target.sum(dim=1)
    dice = (2.0 * inter) / (union + 1e-6)
    return dice.mean()


@torch.no_grad()
def evaluate_val_dice_at_threshold(model, val_dl, device, thr: float):
    model.eval()
    dice_total = 0.0
    dice_count = 0
    for x, y in val_dl:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        probs = torch.sigmoid(logits)

        has_tumor = y.sum(dim=(1, 2, 3)) > 0
        if has_tumor.any():
            p = probs[has_tumor]
            g = y[has_tumor]
            d = hard_dice_from_probs(p, g, thr=thr)
            bs = p.size(0)
            dice_total += d.item() * bs
            dice_count += bs

    return dice_total / max(dice_count, 1)


@torch.no_grad()
def evaluate_test_metrics_at_threshold(model, test_dl, device, thr: float):
    model.eval()
    hd_vals = []
    dice_total = 0.0
    dice_count = 0
    fp_slices = 0
    neg_slices = 0
    fn_slices = 0
    pos_slices = 0

    for x, y in test_dl:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        probs = torch.sigmoid(logits)

        has_tumor = y.sum(dim=(1, 2, 3)) > 0
        if has_tumor.any():
            p_pos = probs[has_tumor]
            g_pos = y[has_tumor]
            d = hard_dice_from_probs(p_pos, g_pos, thr=thr)
            bs = p_pos.size(0)
            dice_total += d.item() * bs
            dice_count += bs
            pos_slices += int(has_tumor.sum().item())

        preds_exists = (probs.view(probs.size(0), -1) > thr).any(dim=1)
        neg_mask = ~has_tumor
        if neg_mask.any():
            neg_slices += int(neg_mask.sum().item())
            fp_slices += int(preds_exists[neg_mask].sum().item())
        if has_tumor.any():
            fn_slices += int((~preds_exists[has_tumor]).sum().item())

        pred_mask = (probs > thr).float().cpu().numpy()
        gt = y.cpu().numpy()
        for i in range(pred_mask.shape[0]):
            p = pred_mask[i, 0] > 0.5
            g = gt[i, 0] > 0.5
            if g.sum() == 0:
                continue
            hd_vals.append(hd95_2d(p, g))

    hd_vals = np.asarray(hd_vals, dtype=np.float32)
    hd_vals = hd_vals[np.isfinite(hd_vals)]
    return {
        "dice": dice_total / max(dice_count, 1),
        "hd95_median": float(np.median(hd_vals)) if hd_vals.size else float("nan"),
        "fnr": fn_slices / max(pos_slices, 1),
        "fpr": fp_slices / max(neg_slices, 1),
    }


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
    summary_dir = Path(args.summary_dir)
    summary_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    exp_dirs = []
    for d in sorted(base_dir.iterdir()):
        if not d.is_dir():
            continue
        if d.name.startswith("summary"):
            continue
        model_name, seed, suffix = parse_exp_dir_name(d.name)
        if model_name is None:
            continue
        if args.artifact_suffix and suffix != args.artifact_suffix:
            continue
        if not args.artifact_suffix and suffix:
            continue
        exp_dirs.append(d)

    checkpoint_rows = []
    for exp_dir in exp_dirs:
        model_name, seed, _ = parse_exp_dir_name(exp_dir.name)
        prefix = exp_dir.name
        ckpt_path = exp_dir / f"{prefix}_best.pt"
        cfg_path = exp_dir / f"{prefix}_config.json"
        scan_path = exp_dir / f"{prefix}_val_threshold_scan.json"
        tuned_path = exp_dir / f"{prefix}_metrics_test_tuned.json"

        row = {
            "model": model_name,
            "seed": seed,
            "artifact_prefix": prefix,
            "status": "ok",
            "best_threshold": np.nan,
            "best_val_dice": np.nan,
            "test_dice": np.nan,
            "test_hd95_median": np.nan,
            "test_fnr": np.nan,
            "test_fpr": np.nan,
            "error": "",
        }

        if scan_path.exists() and tuned_path.exists():
            try:
                scan_payload = json.loads(scan_path.read_text(encoding="utf-8"))
                tuned_payload = json.loads(tuned_path.read_text(encoding="utf-8"))
                row["best_threshold"] = scan_payload.get("best_threshold", np.nan)
                row["best_val_dice"] = scan_payload.get("best_val_dice", np.nan)
                row["test_dice"] = tuned_payload.get("dice", np.nan)
                row["test_hd95_median"] = tuned_payload.get("hd95_median", np.nan)
                row["test_fnr"] = tuned_payload.get("fnr", np.nan)
                row["test_fpr"] = tuned_payload.get("fpr", np.nan)
                checkpoint_rows.append(row)
                print(f"{prefix}: skip (already tuned)")
                continue
            except Exception:
                pass

        if not ckpt_path.exists() or not cfg_path.exists():
            row["status"] = "missing_files"
            row["error"] = "missing best.pt or config.json"
            checkpoint_rows.append(row)
            continue

        try:
            cfg_json = json.loads(cfg_path.read_text(encoding="utf-8"))
            cfg_data = {k: v for k, v in cfg_json.items() if k in ExperimentConfig.__dataclass_fields__}
            config = ExperimentConfig(**cfg_data)
            dataset = build_dataset(config)
            _, val_dl, test_dl, _ = build_loaders(dataset, config)
            model = load_saved_model(ckpt_path, input_mode=config.input_mode, device=device)

            val_dice_map = {}
            for thr in THRESHOLDS:
                val_dice_map[f"{thr:.2f}"] = evaluate_val_dice_at_threshold(model, val_dl, device, thr=thr)

            best_threshold = min(THRESHOLDS, key=lambda t: (-val_dice_map[f"{t:.2f}"], t))
            best_val_dice = val_dice_map[f"{best_threshold:.2f}"]
            test_metrics = evaluate_test_metrics_at_threshold(model, test_dl, device, thr=best_threshold)

            scan_payload = {
                "thresholds": THRESHOLDS,
                "val_dice_by_threshold": val_dice_map,
                "best_threshold": best_threshold,
                "best_val_dice": best_val_dice,
            }
            tuned_payload = {
                "selected_threshold": best_threshold,
                "dice": test_metrics["dice"],
                "hd95_median": test_metrics["hd95_median"],
                "fnr": test_metrics["fnr"],
                "fpr": test_metrics["fpr"],
            }
            scan_path.write_text(json.dumps(scan_payload, indent=2), encoding="utf-8")
            tuned_path.write_text(json.dumps(tuned_payload, indent=2), encoding="utf-8")

            row["best_threshold"] = best_threshold
            row["best_val_dice"] = best_val_dice
            row["test_dice"] = test_metrics["dice"]
            row["test_hd95_median"] = test_metrics["hd95_median"]
            row["test_fnr"] = test_metrics["fnr"]
            row["test_fpr"] = test_metrics["fpr"]
        except Exception as exc:
            row["status"] = "failed"
            row["error"] = repr(exc)

        checkpoint_rows.append(row)
        print(f"{prefix}: {row['status']}")

    checkpoints_df = pd.DataFrame(checkpoint_rows)
    checkpoints_df["model"] = pd.Categorical(checkpoints_df["model"], categories=MODEL_ORDER, ordered=True)
    checkpoints_df = checkpoints_df.sort_values(["model", "seed"]).reset_index(drop=True)

    ok_df = checkpoints_df[checkpoints_df["status"] == "ok"].copy()
    grouped = ok_df.groupby("model").agg(
        n_seeds_used=("seed", "count"),
        dice_mean=("test_dice", "mean"),
        dice_std=("test_dice", "std"),
        hd95_median_mean=("test_hd95_median", "mean"),
        hd95_median_std=("test_hd95_median", "std"),
        fnr_mean=("test_fnr", "mean"),
        fnr_std=("test_fnr", "std"),
        fpr_mean=("test_fpr", "mean"),
        fpr_std=("test_fpr", "std"),
    )

    present_models = [m for m in MODEL_ORDER if m in grouped.index]
    summary_df = grouped.reindex(present_models).reset_index()
    summary_df["n_seeds_used"] = summary_df["n_seeds_used"].fillna(0).astype(int)

    checkpoints_csv = summary_dir / "threshold_tuning_all_checkpoints.csv"
    checkpoints_json = summary_dir / "threshold_tuning_all_checkpoints.json"
    summary_csv = summary_dir / "threshold_tuning_mean_std_by_model.csv"
    summary_json = summary_dir / "threshold_tuning_mean_std_by_model.json"
    summary_3sf_csv = summary_dir / "threshold_tuning_mean_std_by_model_3sf.csv"
    summary_3sf_md = summary_dir / "threshold_tuning_mean_std_by_model_3sf.md"
    dice_plot = summary_dir / "threshold_tuned_dice_bar_mean_std.png"
    hd95_plot = summary_dir / "threshold_tuned_hd95_median_bar_mean_std.png"
    fnr_plot = summary_dir / "threshold_tuned_fnr_bar_mean_std.png"
    fpr_plot = summary_dir / "threshold_tuned_fpr_bar_mean_std.png"

    checkpoints_df.to_csv(checkpoints_csv, index=False, encoding="utf-8")
    checkpoints_json.write_text(checkpoints_df.to_json(orient="records", force_ascii=False, indent=2), encoding="utf-8")
    summary_df.to_csv(summary_csv, index=False, encoding="utf-8")
    summary_json.write_text(summary_df.to_json(orient="records", force_ascii=False, indent=2), encoding="utf-8")

    summary_3sf = summary_df.copy()
    numeric_cols = [c for c in summary_3sf.columns if c not in {"model", "n_seeds_used"}]
    for col in numeric_cols:
        summary_3sf[col] = summary_3sf[col].map(fmt_3sf)
    summary_3sf.to_csv(summary_3sf_csv, index=False, encoding="utf-8")

    headers = [
        "Model",
        "n_seeds_used",
        "Dice mean",
        "Dice std",
        "HD95 median mean",
        "HD95 median std",
        "FNR mean",
        "FNR std",
        "FPR mean",
        "FPR std",
    ]
    md_lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    for _, row in summary_3sf.iterrows():
        md_lines.append(
            "| "
            + " | ".join(
                [
                    str(row["model"]),
                    str(row["n_seeds_used"]),
                    str(row["dice_mean"]),
                    str(row["dice_std"]),
                    str(row["hd95_median_mean"]),
                    str(row["hd95_median_std"]),
                    str(row["fnr_mean"]),
                    str(row["fnr_std"]),
                    str(row["fpr_mean"]),
                    str(row["fpr_std"]),
                ]
            )
            + " |"
        )
    summary_3sf_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    plot_metric_bar(summary_df, "dice", "Test Dice (mean across seeds)", "Test Dice Mean ± Std (dynamic threshold)", dice_plot, ylim=(0.0, 1.0))
    plot_metric_bar(summary_df, "hd95_median", "Test HD95 Median (mean across seeds)", "Test HD95 Median Mean ± Std (dynamic threshold)", hd95_plot)
    plot_metric_bar(summary_df, "fnr", "Test FNR (mean across seeds)", "Test FNR Mean ± Std (dynamic threshold)", fnr_plot, ylim=(0.0, 1.0))
    plot_metric_bar(summary_df, "fpr", "Test FPR (mean across seeds)", "Test FPR Mean ± Std (dynamic threshold)", fpr_plot, ylim=(0.0, 1.0))

    print("Saved:")
    print(checkpoints_csv)
    print(checkpoints_json)
    print(summary_csv)
    print(summary_json)
    print(summary_3sf_csv)
    print(summary_3sf_md)
    print(dice_plot)
    print(hd95_plot)
    print(fnr_plot)
    print(fpr_plot)


if __name__ == "__main__":
    main()
