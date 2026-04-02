import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path

from experiment_utils import make_experiment_config, run_experiment


CLOUD_EXPERIMENTS = [
    "2d_bce_dice",
    "2d_bce_dice_boundary_0.05",
    "25d_bce_dice",
    "25d_bce_dice_boundary_0.05",
]
DEFAULT_RESULTS_ROOT = "result_500"
DEFAULT_CHECKPOINTS_ROOT = "checkpoints_500"
DEFAULT_DATA_ROOT = "../data/BraTS2021_Training_Data"


def utc_now():
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def append_jsonl(path: Path, row: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def is_completed(config) -> bool:
    out_dir = Path(config.results_root) / config.artifact_prefix
    return (out_dir / f"{config.artifact_prefix}_metrics_test.json").exists()


def configure_cloud_run(name: str, dataset_seed: int, results_root: str, data_root: str):
    config = make_experiment_config(name, dataset_seed=dataset_seed, results_root=results_root)
    config.root = data_root
    config.max_patients = 500
    config.train_frac = 0.7
    config.val_frac = 0.1
    config.batch_size = 4
    return config


def parse_args():
    parser = argparse.ArgumentParser(description="Cloud batch training for 4 selected experiments.")
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--seed-end", type=int, default=2)
    parser.add_argument("--results-root", default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--checkpoints-root", default=DEFAULT_CHECKPOINTS_ROOT)
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--force-retrain", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.seed_start > args.seed_end:
        raise ValueError("--seed-start must be <= --seed-end")

    total_jobs = len(CLOUD_EXPERIMENTS) * (args.seed_end - args.seed_start + 1)
    job_index = 0
    done = 0
    skipped = 0
    failed = 0
    progress_log = Path(args.results_root) / "batch_progress.jsonl"
    checkpoints_root = Path(args.checkpoints_root)
    checkpoints_root.mkdir(parents=True, exist_ok=True)

    for dataset_seed in range(args.seed_start, args.seed_end + 1):
        for name in CLOUD_EXPERIMENTS:
            job_index += 1
            config = configure_cloud_run(name, dataset_seed, args.results_root, args.data_root)
            out_dir = Path(config.results_root) / config.artifact_prefix

            if not args.force_retrain and is_completed(config):
                skipped += 1
                print(f"[{job_index}/{total_jobs}] SKIP dataset_seed={dataset_seed} model={name}")
                append_jsonl(
                    progress_log,
                    {
                        "time": utc_now(),
                        "status": "skipped",
                        "dataset_seed": dataset_seed,
                        "model": name,
                        "output_dir": str(out_dir),
                    },
                )
                continue

            print(f"[{job_index}/{total_jobs}] START dataset_seed={dataset_seed} model={name}")
            append_jsonl(
                progress_log,
                {
                    "time": utc_now(),
                    "status": "start",
                    "dataset_seed": dataset_seed,
                    "model": name,
                    "output_dir": str(out_dir),
                },
            )
            try:
                result = run_experiment(config)
                best_ckpt = Path(result["best_ckpt"])
                copied_ckpt = checkpoints_root / best_ckpt.name
                shutil.copy2(best_ckpt, copied_ckpt)
            except Exception as exc:
                failed += 1
                print(f"[{job_index}/{total_jobs}] FAILED dataset_seed={dataset_seed} model={name}")
                append_jsonl(
                    progress_log,
                    {
                        "time": utc_now(),
                        "status": "failed",
                        "dataset_seed": dataset_seed,
                        "model": name,
                        "output_dir": str(out_dir),
                        "error": repr(exc),
                    },
                )
                if not args.continue_on_error:
                    raise
                continue

            done += 1
            print(f"[{job_index}/{total_jobs}] DONE dataset_seed={dataset_seed} model={name}")
            append_jsonl(
                progress_log,
                {
                    "time": utc_now(),
                    "status": "done",
                    "dataset_seed": dataset_seed,
                    "model": name,
                    "output_dir": result["output_dir"],
                    "copied_checkpoint": str(copied_ckpt),
                },
            )

    print(f"Batch finished: total={total_jobs}, done={done}, skipped={skipped}, failed={failed}")
    print("Progress log:", progress_log)
    print("Copied checkpoints:", checkpoints_root)


if __name__ == "__main__":
    main()
