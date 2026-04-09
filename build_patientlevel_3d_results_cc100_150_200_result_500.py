import argparse
import subprocess
import sys
from pathlib import Path


DEFAULT_RESULTS_ROOT = Path("result_500")
THRESHOLDS = [400, 450, 500]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build patient-level 3D postprocessed Dice/HD95 results for cc400, cc450 and cc500."
    )
    parser.add_argument("--results-root", default=str(DEFAULT_RESULTS_ROOT))
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--seed-end", type=int, default=2)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--python", default=sys.executable)
    return parser.parse_args()


def main():
    args = parse_args()
    results_root = Path(args.results_root)

    for min_component_size in THRESHOLDS:
        output_dir = results_root / "summary_0.5" / f"results_3D_cc{min_component_size}"
        cmd = [
            args.python,
            "build_patientlevel_3d_results_cc50_result_500.py",
            "--results-root",
            str(results_root),
            "--output-dir",
            str(output_dir),
            "--seed-start",
            str(args.seed_start),
            "--seed-end",
            str(args.seed_end),
            "--threshold",
            str(args.threshold),
            "--batch-size",
            str(args.batch_size),
            "--min-component-size",
            str(min_component_size),
        ]
        print("Running:", " ".join(cmd))
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
