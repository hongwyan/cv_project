from experiment_utils import make_experiment_config, parse_dataset_seed, run_experiment


def main():
    dataset_seed = parse_dataset_seed()
    config = make_experiment_config("25d_baseline", dataset_seed=dataset_seed)
    run_experiment(config)


if __name__ == "__main__":
    main()
