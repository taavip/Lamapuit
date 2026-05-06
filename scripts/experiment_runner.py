#!/usr/bin/env python
"""
Experiment runner to systematically test different training configurations
and find optimal settings for small CDW dataset.

Experiments:
1. Model size comparison (nano, small, medium)
2. Augmentation levels (none, light, moderate, heavy)
3. Learning rate schedules
4. Batch size effects
"""

import subprocess
import json
from pathlib import Path
from datetime import datetime
import time


class ExperimentRunner:
    def __init__(self, base_dir="experiments"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        self.results = []

    def run_experiment(self, name, config):
        """Run a single training experiment"""
        print(f"\n{'='*70}")
        print(f"EXPERIMENT: {name}")
        print(f"{'='*70}")
        print(f"Config: {json.dumps(config, indent=2)}")

        start_time = time.time()

        # Build command
        cmd = [
            "python",
            "scripts/train_experiment.py",
            "--name",
            name,
            "--model",
            config.get("model", "yolo11s-seg.pt"),
            "--epochs",
            str(config.get("epochs", 200)),
            "--batch",
            str(config.get("batch", 8)),
            "--patience",
            str(config.get("patience", 30)),
        ]

        # Add augmentation level
        if "augmentation" in config:
            cmd.extend(["--augmentation", config["augmentation"]])

        # Run training
        result = subprocess.run(cmd, capture_output=True, text=True)

        elapsed = time.time() - start_time

        # Parse results
        exp_result = {
            "name": name,
            "config": config,
            "elapsed_time": elapsed,
            "success": result.returncode == 0,
            "stdout": result.stdout[-500:] if result.stdout else "",  # Last 500 chars
            "stderr": result.stderr[-500:] if result.stderr else "",
        }

        # Extract metrics if available
        results_file = Path(f"runs/cdw_detect/{name}/results.csv")
        if results_file.exists():
            exp_result["results_file"] = str(results_file)

        self.results.append(exp_result)

        # Save results after each experiment
        self._save_results()

        return exp_result

    def _save_results(self):
        """Save experiment results to JSON"""
        results_file = (
            self.base_dir / f"experiments_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        with open(results_file, "w") as f:
            json.dump(self.results, f, indent=2)
        print(f"\n✓ Results saved to {results_file}")


def main():
    runner = ExperimentRunner()

    # Experiment 1: Model size comparison
    print("\n" + "=" * 70)
    print("EXPERIMENT SET 1: Model Size Comparison")
    print("=" * 70)

    for model in ["yolo11n-seg.pt", "yolo11s-seg.pt"]:
        runner.run_experiment(
            name=f"exp_model_{model.split('-')[0].replace('.pt', '')}",
            config={
                "model": model,
                "epochs": 150,
                "batch": 8 if "s" in model else 16,
                "patience": 30,
                "augmentation": "light",
            },
        )

    # Experiment 2: Augmentation levels
    print("\n" + "=" * 70)
    print("EXPERIMENT SET 2: Augmentation Levels")
    print("=" * 70)

    for aug_level in ["none", "light", "moderate"]:
        runner.run_experiment(
            name=f"exp_aug_{aug_level}",
            config={
                "model": "yolo11s-seg.pt",
                "epochs": 150,
                "batch": 8,
                "patience": 30,
                "augmentation": aug_level,
            },
        )

    # Experiment 3: Batch size effects
    print("\n" + "=" * 70)
    print("EXPERIMENT SET 3: Batch Size Effects")
    print("=" * 70)

    for batch_size in [4, 8, 16]:
        runner.run_experiment(
            name=f"exp_batch_{batch_size}",
            config={
                "model": "yolo11s-seg.pt",
                "epochs": 150,
                "batch": batch_size,
                "patience": 30,
                "augmentation": "light",
            },
        )

    print("\n" + "=" * 70)
    print("ALL EXPERIMENTS COMPLETE!")
    print("=" * 70)
    print(f"Total experiments: {len(runner.results)}")
    print(f"Results saved to: {runner.base_dir}")


if __name__ == "__main__":
    main()
