import argparse
import json
import os
from pathlib import Path

# Windows + conda 下 torch / skimage 组合有时会重复加载 OpenMP runtime。
# 这里只对这个可视化脚本启用兼容开关，避免影响其他训练脚本。
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.gridspec import GridSpec
from skimage.measure import marching_cubes

from brats2d_dataset import BraTS2D
from brats2p5d_dataset import BraTS2p5D, _clamp_triplet, _load_nifti, _robust_zscore
from experiment_utils import ExperimentConfig, load_saved_model
from splits import build_patient_split


DEFAULT_RESULTS_ROOT = Path("result_500")
DEFAULT_SIZE_CSV = DEFAULT_RESULTS_ROOT / "summary_size" / "patients_size_by_seed.csv"
DEFAULT_OUTPUT_DIR = DEFAULT_RESULTS_ROOT / "summary_0.5" / "results_final" / "visualize"
DEFAULT_THRESHOLD = 0.5
DEFAULT_BATCH_SIZE = 32
DEFAULT_CASES = "0:BraTS2021_00147,0:BraTS2021_00071"
MODEL_NAMES = {
    "2d_bce_dice": "2d_bce_dice",
    "2d_boundary": "2d_bce_dice_boundary_0.05",
    "25d_bce_dice": "25d_bce_dice",
    "25d_boundary": "25d_bce_dice_boundary_0.05",
}
MODEL_ORDER = ["2d_bce_dice", "2d_boundary", "25d_bce_dice", "25d_boundary"]
GROUP_ORDER = ["high_30pct", "mid_40pct", "low_30pct"]
GROUP_LABELS = {
    "high_30pct": "Large",
    "mid_40pct": "Medium",
    "low_30pct": "Small",
}
PANEL_ORDER = ["gt", "2d_bce_dice", "2d_boundary", "25d_bce_dice", "25d_boundary"]
PANEL_TITLES = {
    "gt": "GT",
    "2d_bce_dice": "2D BCE+Dice",
    "2d_boundary": "2D Boundary",
    "25d_bce_dice": "2.5D BCE+Dice",
    "25d_boundary": "2.5D Boundary",
}
PANEL_COLORS = {
    "gt": "#2ca02c",
    "2d_bce_dice": "#ffd54f",
    "2d_boundary": "#ffd54f",
    "25d_bce_dice": "#ffd54f",
    "25d_boundary": "#ffd54f",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize selected 3D cases with GT on the left and 2D/2.5D predictions on the right."
    )
    parser.add_argument("--results-root", default=str(DEFAULT_RESULTS_ROOT))
    parser.add_argument("--size-csv", default=str(DEFAULT_SIZE_CSV))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--examples-per-group", type=int, default=3)
    parser.add_argument(
        "--cases",
        default=DEFAULT_CASES,
        help="Comma-separated list like '0:BraTS2021_00147,0:BraTS2021_00071'. If empty, fall back to size-group sampling.",
    )
    return parser.parse_args()


def load_config(results_root: Path, model_name: str, dataset_seed: int) -> tuple[ExperimentConfig, Path]:
    artifact_prefix = f"{model_name}_seed{dataset_seed}"
    out_dir = results_root / artifact_prefix
    cfg_path = out_dir / f"{artifact_prefix}_config.json"
    ckpt_path = out_dir / f"{artifact_prefix}_best.pt"
    if not cfg_path.exists() or not ckpt_path.exists():
        raise FileNotFoundError(f"Missing config or checkpoint for {artifact_prefix}")
    cfg_json = json.loads(cfg_path.read_text(encoding="utf-8"))
    cfg_data = {k: v for k, v in cfg_json.items() if k in ExperimentConfig.__dataclass_fields__}
    return ExperimentConfig(**cfg_data), ckpt_path


def build_dataset_for_config(config: ExperimentConfig):
    common = dict(
        root=config.root,
        max_patients=config.max_patients,
        only_tumor_slices=config.only_tumor_slices,
        neg_to_pos_ratio=config.neg_to_pos_ratio,
        cache_volumes=config.cache_volumes,
        seed=config.dataset_seed,
    )
    if config.input_mode == "2d":
        return BraTS2D(**common)
    if config.input_mode == "2p5d":
        return BraTS2p5D(**common)
    raise ValueError(f"Unsupported input_mode: {config.input_mode}")


def patient_dir_map(dataset) -> dict[str, Path]:
    patient_dirs = dataset.patient_dirs if hasattr(dataset, "patient_dirs") else dataset.ds.patient_dirs
    return {Path(p).name: Path(p) for p in patient_dirs}


def load_patient_volume(patient_dir: Path):
    flair_path = next(patient_dir.glob("*_flair.nii*"))
    seg_path = next(patient_dir.glob("*_seg.nii*"))
    flair = _load_nifti(flair_path)
    seg = _load_nifti(seg_path)
    gt = (seg > 0).astype(np.uint8)
    flair_n = _robust_zscore(flair)
    return flair_n, gt


def predict_volume(model, flair_n: np.ndarray, input_mode: str, threshold: float, device, batch_size: int):
    _, _, depth = flair_n.shape
    pred_volume = np.zeros_like(flair_n, dtype=np.uint8)
    with torch.no_grad():
        for start in range(0, depth, batch_size):
            stop = min(start + batch_size, depth)
            batch_ts = list(range(start, stop))
            if input_mode == "2d":
                batch = np.stack([flair_n[:, :, t][None, ...] for t in batch_ts], axis=0).astype(np.float32)
            elif input_mode == "2p5d":
                triplets = []
                for t in batch_ts:
                    t0, t1, t2 = _clamp_triplet(t, depth)
                    triplets.append(np.stack([flair_n[:, :, t0], flair_n[:, :, t1], flair_n[:, :, t2]], axis=0))
                batch = np.stack(triplets, axis=0).astype(np.float32)
            else:
                raise ValueError(f"Unsupported input_mode: {input_mode}")

            x = torch.from_numpy(batch).to(device, non_blocking=True)
            pred = (torch.sigmoid(model(x)) > threshold).cpu().numpy().astype(np.uint8)[:, 0]
            for i, t in enumerate(batch_ts):
                pred_volume[:, :, t] = pred[i]
    return pred_volume


def select_examples(size_df: pd.DataFrame, examples_per_group: int) -> pd.DataFrame:
    rows = []
    for group_name in GROUP_ORDER:
        group_df = size_df[size_df["size_group"] == group_name].copy()
        group_df = group_df.sort_values(["seed", "patient_id"]).head(examples_per_group)
        rows.append(group_df)
    out_df = pd.concat(rows, ignore_index=True)
    out_df["display_group"] = out_df["size_group"].map(GROUP_LABELS)
    return out_df


def parse_cases(case_arg: str) -> list[tuple[int, str]]:
    cases = []
    if not case_arg.strip():
        return cases
    for item in case_arg.split(","):
        seed_str, patient_id = item.strip().split(":", 1)
        cases.append((int(seed_str), patient_id.strip()))
    return cases


def add_surface(ax, volume: np.ndarray, color: str):
    mask = volume.astype(bool)
    if mask.sum() == 0:
        ax.text2D(0.5, 0.5, "Empty", transform=ax.transAxes, ha="center", va="center")
        return
    verts, faces, _, _ = marching_cubes(mask.astype(np.float32), level=0.5)
    mesh = Poly3DCollection(verts[faces], alpha=0.75)
    mesh.set_facecolor(color)
    mesh.set_edgecolor("none")
    ax.add_collection3d(mesh)
    ax.set_xlim(0, mask.shape[0])
    ax.set_ylim(0, mask.shape[1])
    ax.set_zlim(0, mask.shape[2])
    ax.set_box_aspect(mask.shape)


def style_ax(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.grid(False)
    ax.view_init(elev=20, azim=45)


def main():
    args = parse_args()
    results_root = Path(args.results_root)
    size_csv = Path(args.size_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    size_df = pd.read_csv(size_csv, usecols=["seed", "patient_id", "size_group", "ratio", "tumor_vox"])
    requested_cases = parse_cases(args.cases)
    if requested_cases:
        mask = False
        for dataset_seed, patient_id in requested_cases:
            mask = mask | ((size_df["seed"] == dataset_seed) & (size_df["patient_id"] == patient_id))
        examples_df = size_df[mask].copy()
        case_order = {(seed, pid): idx for idx, (seed, pid) in enumerate(requested_cases)}
        examples_df["case_order"] = examples_df.apply(lambda r: case_order[(int(r["seed"]), r["patient_id"])], axis=1)
        examples_df = examples_df.sort_values("case_order").drop(columns=["case_order"]).reset_index(drop=True)
        examples_df["display_group"] = examples_df["size_group"].map(GROUP_LABELS)
    else:
        examples_df = select_examples(size_df, args.examples_per_group)
    examples_df.to_csv(output_dir / "selected_examples_by_size.csv", index=False, encoding="utf-8")

    loaded_seed_context = {}
    for dataset_seed in sorted(examples_df["seed"].unique()):
        configs = {}
        datasets = {}
        test_patients_map = {}
        models = {}
        for key, model_name in MODEL_NAMES.items():
            config, ckpt_path = load_config(results_root, model_name, int(dataset_seed))
            dataset = build_dataset_for_config(config)
            split = build_patient_split(dataset, seed=config.split_seed, train_frac=config.train_frac, val_frac=config.val_frac)
            configs[key] = config
            datasets[key] = dataset
            test_patients_map[key] = split["test_patients"]
            models[key] = load_saved_model(ckpt_path, input_mode=config.input_mode, device=device)
        reference_patients = test_patients_map["2d_bce_dice"]
        for key, test_patients in test_patients_map.items():
            if test_patients != reference_patients:
                raise RuntimeError(f"Test patient mismatch for seed={dataset_seed}: {key}")
        loaded_seed_context[int(dataset_seed)] = {
            "configs": configs,
            "datasets": datasets,
            "models": models,
            "patient_dirs": patient_dir_map(datasets["25d_boundary"]),
        }

    for row_idx, row in examples_df.reset_index(drop=True).iterrows():
        dataset_seed = int(row["seed"])
        patient_id = row["patient_id"]
        group_name = row["display_group"]
        context = loaded_seed_context[dataset_seed]
        flair_n, gt = load_patient_volume(context["patient_dirs"][patient_id])

        volumes = {"gt": gt}
        for key in MODEL_ORDER:
            config = context["configs"][key]
            model = context["models"][key]
            volumes[key] = predict_volume(model, flair_n, config.input_mode, args.threshold, device, args.batch_size)

        fig = plt.figure(figsize=(16, 8))
        gs = GridSpec(2, 3, figure=fig, width_ratios=[1.25, 1, 1])

        gt_ax = fig.add_subplot(gs[:, 0], projection="3d")
        add_surface(gt_ax, volumes["gt"], PANEL_COLORS["gt"])
        style_ax(gt_ax)
        gt_ax.set_title(PANEL_TITLES["gt"], fontsize=12)

        panel_positions = {
            "2d_bce_dice": gs[0, 1],
            "2d_boundary": gs[0, 2],
            "25d_bce_dice": gs[1, 1],
            "25d_boundary": gs[1, 2],
        }
        for panel_key in MODEL_ORDER:
            ax = fig.add_subplot(panel_positions[panel_key], projection="3d")
            add_surface(ax, volumes[panel_key], PANEL_COLORS[panel_key])
            style_ax(ax)
            ax.set_title(PANEL_TITLES[panel_key], fontsize=11)

        fig.suptitle(
            f"{group_name} | seed {dataset_seed} | {patient_id} | "
            f"tumor_vox={int(row['tumor_vox'])} | ratio={row['ratio']:.4f}",
            fontsize=13,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        out_path = output_dir / f"seed{dataset_seed}_{patient_id}.png"
        fig.savefig(out_path, dpi=180)
        plt.close(fig)
        print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
