import json

from experiment_utils import evaluate_saved_experiment


def main():
    experiment_names = [
        "2d_baseline",
        "2d_weightedBCE",
        "2d_boundary_combo",
    ]

    for name in experiment_names:
        metrics = evaluate_saved_experiment(name, input_mode="2d")
        print(f"{name}: {json.dumps(metrics, indent=2)}")


if __name__ == "__main__":
    main()
