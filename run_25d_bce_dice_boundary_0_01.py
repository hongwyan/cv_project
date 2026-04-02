import json
from pathlib import Path

from experiment_utils import evaluate_and_save_threshold_metrics, make_experiment_config, run_experiment


EXPERIMENT_NAME = "25d_bce_dice_boundary_0.01"
THRESHOLD = 0.5


def main():
    for dataset_seed in range(5):
        config = make_experiment_config(EXPERIMENT_NAME, dataset_seed=dataset_seed)
        out_dir = Path(config.results_root) / config.artifact_prefix
        metrics_05_path = out_dir / f"{config.artifact_prefix}_metrics_test_{THRESHOLD:g}.json"
        if metrics_05_path.exists():
            print(f"SKIP seed={dataset_seed} name={EXPERIMENT_NAME} (found {metrics_05_path.name})")
            print(metrics_05_path.read_text(encoding='utf-8'))
            continue
        print(f"START seed={dataset_seed} name={EXPERIMENT_NAME}")
        result = run_experiment(config)
        metrics_05, metrics_path = evaluate_and_save_threshold_metrics(config, threshold=THRESHOLD)
        print(f"seed={dataset_seed} threshold={THRESHOLD}")
        print(json.dumps(metrics_05, indent=2))
        print(f"saved_0.5_metrics: {metrics_path}")
        print(f"output_dir: {result['output_dir']}")


if __name__ == "__main__":
    main()
