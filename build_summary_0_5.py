import json
import argparse
from pathlib import Path

import torch

from experiment_utils import ExperimentConfig, build_dataset, build_loaders, evaluate_test_metrics, load_saved_model


METRICS_SUFFIX = "_0.5"


def parse_args():
    parser = argparse.ArgumentParser(description="Build fixed-threshold 0.5 metrics from saved checkpoints.")
    parser.add_argument("--base-dir", default="result_new")
    parser.add_argument("--summary-dir", default="")
    return parser.parse_args()


def parse_exp_dir_name(name: str):
    if "_seed" not in name:
        return None
    return name


def main():
    args = parse_args()
    base_dir = Path(args.base_dir)
    summary_dir = Path(args.summary_dir) if args.summary_dir else base_dir / "summary_0.5"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    processed = 0
    skipped = 0
    for exp_dir in sorted(base_dir.iterdir()):
        if not exp_dir.is_dir():
            continue
        if exp_dir.name.startswith("summary"):
            continue
        prefix = parse_exp_dir_name(exp_dir.name)
        if prefix is None:
            continue

        ckpt_path = exp_dir / f"{prefix}_best.pt"
        cfg_path = exp_dir / f"{prefix}_config.json"
        metrics_path = exp_dir / f"{prefix}_metrics_test{METRICS_SUFFIX}.json"

        if metrics_path.exists():
            print(f"{prefix}: skip (already has metrics_test{METRICS_SUFFIX}.json)")
            skipped += 1
            continue
        if not ckpt_path.exists() or not cfg_path.exists():
            print(f"{prefix}: skip (missing best.pt or config.json)")
            skipped += 1
            continue

        cfg_json = json.loads(cfg_path.read_text(encoding="utf-8"))
        cfg_data = {k: v for k, v in cfg_json.items() if k in ExperimentConfig.__dataclass_fields__}
        config = ExperimentConfig(**cfg_data)
        dataset = build_dataset(config)
        _, _, test_dl, _ = build_loaders(dataset, config)
        model = load_saved_model(ckpt_path, input_mode=config.input_mode, device=device)
        metrics = evaluate_test_metrics(model, test_dl, config, device, threshold=0.5)
        metrics["selected_threshold"] = 0.5
        metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        processed += 1
        print(f"{prefix}: wrote {metrics_path.name}")

    summary_dir.mkdir(parents=True, exist_ok=True)
    print(f"re-evaluation finished: processed={processed}, skipped={skipped}")
    print(f"next: summarize_test_results.py --base-dir {base_dir} --output-dir {summary_dir} --metrics-file-suffix _0.5")


if __name__ == "__main__":
    main()
