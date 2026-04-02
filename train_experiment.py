import argparse

from experiment_utils import EXPERIMENT_ORDER, make_experiment_config, run_experiment


def parse_args():
    parser = argparse.ArgumentParser(description="Run one experiment by name.")
    parser.add_argument("--name", required=True, choices=EXPERIMENT_ORDER)
    parser.add_argument("--dataset-seed", type=int, default=3)
    return parser.parse_args()


def main():
    args = parse_args()
    config = make_experiment_config(args.name, dataset_seed=args.dataset_seed)
    run_experiment(config)


if __name__ == "__main__":
    main()
