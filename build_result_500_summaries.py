import argparse
import subprocess
import sys
from pathlib import Path


RESULTS_ROOT = Path("result_500")
SUMMARY_05_DIR = RESULTS_ROOT / "summary_0.5"
SUMMARY_DYN_DIR = RESULTS_ROOT / "summary_dynthres"


def run(cmd: list[str]):
    print("RUN:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Build summary_0.5 and summary_dynthres for result_500.")
    parser.add_argument("--python", default=sys.executable, help="Python executable to use for child scripts.")
    parser.add_argument("--results-root", default=str(RESULTS_ROOT))
    return parser.parse_args()


def main():
    args = parse_args()
    python_exec = args.python
    results_root = Path(args.results_root)
    summary_05_dir = results_root / "summary_0.5"
    summary_dyn_dir = results_root / "summary_dynthres"
    summary_05_dir.mkdir(parents=True, exist_ok=True)
    summary_dyn_dir.mkdir(parents=True, exist_ok=True)

    run([python_exec, "build_summary_0_5.py", "--base-dir", str(results_root), "--summary-dir", str(summary_05_dir)])
    run(
        [
            python_exec,
            "summarize_test_results.py",
            "--base-dir",
            str(results_root),
            "--output-dir",
            str(summary_05_dir),
            "--metrics-file-suffix",
            "_0.5",
        ]
    )
    run(
        [
            python_exec,
            "tune_thresholds_result_100.py",
            "--base-dir",
            str(results_root),
            "--summary-dir",
            str(summary_dyn_dir),
        ]
    )


if __name__ == "__main__":
    main()
