import json

from experiment_utils import evaluate_saved_experiment


def main():
    experiment_names = [
        "25d_baseline",
        "25d_weightedBCE",
        "25d_boundary_combo",
    ]

    for name in experiment_names:
        metrics = evaluate_saved_experiment(name, input_mode="2p5d")
        print(f"{name}: {json.dumps(metrics, indent=2)}")


if __name__ == "__main__":
    main()
