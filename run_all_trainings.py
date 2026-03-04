import argparse
import json
from datetime import datetime
from pathlib import Path

from experiment_utils import EXPERIMENT_ORDER, make_experiment_config, run_experiment


def utc_now():
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def append_jsonl(path: Path, row: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def is_completed(config) -> bool:
    out_dir = Path(config.results_root) / config.artifact_prefix
    metrics_path = out_dir / f"{config.artifact_prefix}_metrics_test.json"
    return metrics_path.exists()


def parse_args():
    parser = argparse.ArgumentParser(description="Batch training with resume support.")
    parser.add_argument("--seed-start", type=int, default=0, help="Start dataset_seed (inclusive).")
    parser.add_argument("--seed-end", type=int, default=4, help="End dataset_seed (inclusive).")
    parser.add_argument(
        "--force-retrain",
        action="store_true",
        help="Retrain all tasks even if metrics_test.json already exists.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Keep running remaining tasks if one task fails.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.seed_start > args.seed_end:
        raise ValueError("--seed-start must be <= --seed-end")

    total_jobs = len(EXPERIMENT_ORDER) * (args.seed_end - args.seed_start + 1)
    job_index = 0
    done = 0
    skipped = 0
    failed = 0
    progress_log = None

    for dataset_seed in range(args.seed_start, args.seed_end + 1):
        for name in EXPERIMENT_ORDER:
            job_index += 1
            config = make_experiment_config(name, dataset_seed=dataset_seed)
            out_dir = Path(config.results_root) / config.artifact_prefix
            progress_log = Path(config.results_root) / "batch_progress.jsonl"

            if not args.force_retrain and is_completed(config):
                skipped += 1
                print(
                    f"[{job_index}/{total_jobs}] SKIP dataset_seed={dataset_seed} "
                    f"model={name} (found metrics)"
                )
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
                },
            )

    print(
        f"Batch finished: total={total_jobs}, done={done}, "
        f"skipped={skipped}, failed={failed}"
    )
    if progress_log is not None:
        print("Progress log:", progress_log)


if __name__ == "__main__":
    main()
