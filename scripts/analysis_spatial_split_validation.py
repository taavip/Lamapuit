#!/usr/bin/env python3
"""
Spatial Split Validation
Validates E07 spatial splits for correct data separation and no leakage.
Verifies methodology and produces split statistics.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
import logging
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SpatialSplitValidator:
    def __init__(self, labels_csv_path, split_json_paths):
        """Initialize validator with labels and split metadata."""
        self.labels_path = Path(labels_csv_path)
        self.split_paths = [Path(p) for p in split_json_paths]
        self.df = pd.read_csv(self.labels_path)
        self.splits = {}
        self.validation_results = {}

        logger.info(f"Loaded {len(self.df)} labels")
        self._load_splits()

    def _load_splits(self):
        """Load split JSON files."""
        for split_path in self.split_paths:
            if split_path.exists():
                with open(split_path) as f:
                    split_data = json.load(f)
                    # Use seed as key
                    seed = split_path.stem.split('_')[-1]  # extract seed number
                    self.splits[seed] = split_data
                    logger.info(f"Loaded split with seed {seed}")

    def validate_e07_split_structure(self, split_data):
        """Validate E07 split contains required fields."""
        required_keys = {'buffer', 'train', 'test'}
        missing = required_keys - set(split_data.keys())

        if missing:
            logger.error(f"Missing keys in split: {missing}")
            return False

        logger.info("✓ Split structure valid (buffer, train, test present)")
        return True

    def validate_split_coverage(self, split_data):
        """Verify that train/test/buffer cover all unique tiles without overlap."""
        train_set = set(split_data.get('train', []))
        test_set = set(split_data.get('test', []))
        buffer_set = set(split_data.get('buffer', []))

        # Check for overlaps
        train_test_overlap = train_set & test_set
        train_buffer_overlap = train_set & buffer_set
        test_buffer_overlap = test_set & buffer_set

        if train_test_overlap:
            logger.error(f"Train-Test overlap: {len(train_test_overlap)} items")
            return False
        if train_buffer_overlap:
            logger.error(f"Train-Buffer overlap: {len(train_buffer_overlap)} items")
            return False
        if test_buffer_overlap:
            logger.error(f"Test-Buffer overlap: {len(test_buffer_overlap)} items")
            return False

        logger.info("✓ No overlaps between train/test/buffer sets")

        # Check coverage
        unique_rasters = self.df['raster'].unique()
        coverage = train_set | test_set | buffer_set
        missing = set(unique_rasters) - coverage

        if missing:
            logger.warning(f"Not all rasters covered: {len(missing)} missing")
            return False

        logger.info(f"✓ All {len(unique_rasters)} rasters covered by split")
        return True

    def compute_split_statistics(self):
        """Compute statistics for all loaded splits."""
        for seed, split_data in self.splits.items():
            train_set = set(split_data.get('train', []))
            test_set = set(split_data.get('test', []))
            buffer_set = set(split_data.get('buffer', []))

            # Count samples in each set
            train_samples = self.df[self.df['raster'].isin(train_set)]
            test_samples = self.df[self.df['raster'].isin(test_set)]
            buffer_samples = self.df[self.df['raster'].isin(buffer_set)]

            stats = {
                'seed': seed,
                'rasters': {
                    'train': len(train_set),
                    'test': len(test_set),
                    'buffer': len(buffer_set),
                    'total': len(train_set) + len(test_set) + len(buffer_set)
                },
                'samples': {
                    'train': len(train_samples),
                    'train_cdw': int((train_samples['label'] == 'cdw').sum()),
                    'train_no_cdw': int((train_samples['label'] == 'no_cdw').sum()),
                    'test': len(test_samples),
                    'test_cdw': int((test_samples['label'] == 'cdw').sum()),
                    'test_no_cdw': int((test_samples['label'] == 'no_cdw').sum()),
                    'buffer': len(buffer_samples),
                    'buffer_cdw': int((buffer_samples['label'] == 'cdw').sum()),
                    'buffer_no_cdw': int((buffer_samples['label'] == 'no_cdw').sum()),
                    'total': len(train_samples) + len(test_samples) + len(buffer_samples)
                },
                'percentages': {
                    'train_pct': float(len(train_samples) / (len(train_samples) + len(test_samples) + len(buffer_samples)) * 100),
                    'test_pct': float(len(test_samples) / (len(train_samples) + len(test_samples) + len(buffer_samples)) * 100),
                    'buffer_pct': float(len(buffer_samples) / (len(train_samples) + len(test_samples) + len(buffer_samples)) * 100),
                    'train_cdw_pct': float(stats['samples']['train_cdw'] / max(stats['samples']['train'], 1) * 100) if 'train_cdw' in stats.get('samples', {}) else 0
                }
            }

            # Compute percentages correctly
            total_samples = len(train_samples) + len(test_samples) + len(buffer_samples)
            stats['percentages']['train_pct'] = float(len(train_samples) / max(total_samples, 1) * 100)
            stats['percentages']['test_pct'] = float(len(test_samples) / max(total_samples, 1) * 100)
            stats['percentages']['buffer_pct'] = float(len(buffer_samples) / max(total_samples, 1) * 100)
            stats['percentages']['train_cdw_pct'] = float(stats['samples']['train_cdw'] / max(stats['samples']['train'], 1) * 100)

            self.validation_results[seed] = stats

        return self.validation_results

    def log_split_statistics(self):
        """Log comprehensive split statistics."""
        logger.info("\n" + "="*100)
        logger.info("SPATIAL SPLIT VALIDATION STATISTICS")
        logger.info("="*100)

        for seed, stats in sorted(self.validation_results.items(), key=lambda x: int(x[0])):
            logger.info(f"\nSeed {seed}:")
            logger.info(f"  Rasters - Train: {stats['rasters']['train']}, "
                       f"Test: {stats['rasters']['test']}, Buffer: {stats['rasters']['buffer']}")
            logger.info(f"  Samples - Train: {stats['samples']['train']:,} "
                       f"(CWD: {stats['samples']['train_cdw']:,}, bg: {stats['samples']['train_no_cdw']:,})")
            logger.info(f"           Test: {stats['samples']['test']:,} "
                       f"(CWD: {stats['samples']['test_cdw']:,}, bg: {stats['samples']['test_no_cdw']:,})")
            logger.info(f"           Buffer: {stats['samples']['buffer']:,} "
                       f"(CWD: {stats['samples']['buffer_cdw']:,}, bg: {stats['samples']['buffer_no_cdw']:,})")
            logger.info(f"  Distribution - Train: {stats['percentages']['train_pct']:.2f}%, "
                       f"Test: {stats['percentages']['test_pct']:.2f}%, "
                       f"Buffer: {stats['percentages']['buffer_pct']:.2f}%")

        logger.info("="*100)

    def visualize_split_statistics(self, output_dir):
        """Create visualizations of split statistics."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if not self.validation_results:
            logger.warning("No split statistics to visualize")
            return

        # Convert to DataFrame for easier plotting
        split_list = list(self.validation_results.values())

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # 1. Sample distribution across splits (stacked bar)
        ax = axes[0, 0]
        seeds = [str(s['seed']) for s in split_list]
        train_counts = [s['samples']['train'] for s in split_list]
        test_counts = [s['samples']['test'] for s in split_list]
        buffer_counts = [s['samples']['buffer'] for s in split_list]

        x = np.arange(len(seeds))
        width = 0.6
        ax.bar(x, train_counts, width, label='Train', color='steelblue', alpha=0.8, edgecolor='black')
        ax.bar(x, test_counts, width, bottom=train_counts, label='Test', color='orange', alpha=0.8, edgecolor='black')
        ax.bar(x, buffer_counts, width, bottom=np.array(train_counts)+np.array(test_counts),
              label='Buffer', color='lightcoral', alpha=0.8, edgecolor='black')

        ax.set_ylabel('Number of Samples', fontsize=12, fontweight='bold')
        ax.set_xlabel('Seed', fontsize=12, fontweight='bold')
        ax.set_title('Sample Distribution Across Train/Test/Buffer (All Seeds)', fontsize=13, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(seeds)
        ax.legend()
        ax.set_yscale('log')

        # 2. Class distribution in train set
        ax = axes[0, 1]
        train_cdw = [s['samples']['train_cdw'] for s in split_list]
        train_no_cdw = [s['samples']['train_no_cdw'] for s in split_list]

        x = np.arange(len(seeds))
        width = 0.35
        ax.bar(x - width/2, train_cdw, width, label='CWD', color='#d62728', alpha=0.8, edgecolor='black')
        ax.bar(x + width/2, train_no_cdw, width, label='Background', color='#2ca02c', alpha=0.8, edgecolor='black')

        ax.set_ylabel('Number of Samples', fontsize=12, fontweight='bold')
        ax.set_xlabel('Seed', fontsize=12, fontweight='bold')
        ax.set_title('Training Set Class Distribution (All Seeds)', fontsize=13, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(seeds)
        ax.legend()
        ax.set_yscale('log')

        # 3. Split percentages
        ax = axes[1, 0]
        train_pcts = [s['percentages']['train_pct'] for s in split_list]
        test_pcts = [s['percentages']['test_pct'] for s in split_list]
        buffer_pcts = [s['percentages']['buffer_pct'] for s in split_list]

        x = np.arange(len(seeds))
        ax.plot(x, train_pcts, marker='o', linewidth=2, markersize=8, label='Train %', color='steelblue')
        ax.plot(x, test_pcts, marker='s', linewidth=2, markersize=8, label='Test %', color='orange')
        ax.plot(x, buffer_pcts, marker='^', linewidth=2, markersize=8, label='Buffer %', color='lightcoral')

        ax.set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
        ax.set_xlabel('Seed', fontsize=12, fontweight='bold')
        ax.set_title('Split Proportions Consistency Across Seeds', fontsize=13, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(seeds)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim([-5, 105])

        # 4. CWD percentage in training set
        ax = axes[1, 1]
        train_cdw_pcts = [s['percentages']['train_cdw_pct'] for s in split_list]
        ax.bar(range(len(seeds)), train_cdw_pcts, color='#d62728', alpha=0.7, edgecolor='black', linewidth=1.5)
        ax.set_ylabel('CWD Percentage in Training (%)', fontsize=12, fontweight='bold')
        ax.set_xlabel('Seed', fontsize=12, fontweight='bold')
        ax.set_title('CWD Class Percentage in Training Set (All Seeds)', fontsize=13, fontweight='bold')
        ax.set_xticks(range(len(seeds)))
        ax.set_xticklabels(seeds)
        ax.axhline(y=np.mean(train_cdw_pcts), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(train_cdw_pcts):.2f}%')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        output_path = output_dir / 'spatial_split_validation.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"✓ Saved visualization to {output_path}")
        plt.close()

    def run_validation(self, output_dir=None):
        """Run complete validation."""
        logger.info("Starting spatial split validation...")

        # Validate each split
        for seed, split_data in self.splits.items():
            if not self.validate_e07_split_structure(split_data):
                logger.error(f"Structure validation failed for seed {seed}")
                continue

            if not self.validate_split_coverage(split_data):
                logger.error(f"Coverage validation failed for seed {seed}")
                continue

        # Compute and log statistics
        self.compute_split_statistics()
        self.log_split_statistics()

        # Visualize
        if output_dir:
            self.visualize_split_statistics(output_dir)

        logger.info("Validation complete!")
        return self.validation_results

    def save_results(self, output_path):
        """Save validation results."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(self.validation_results, f, indent=2)

        logger.info(f"✓ Results saved to {output_path}")


if __name__ == '__main__':
    labels_csv = Path('/home/tpipar/project/Lamapuit/output/onboarding_labels_v2_drop13_standardized/labels_canonical.csv')

    # Find E07 split files
    import glob
    e07_splits = sorted(glob.glob('/home/tpipar/project/Lamapuit/output/spatial_split_experiments/E07_v3_blocks3_buf2/split_seed*.json'))

    output_dir = Path('/home/tpipar/project/Lamapuit/output/analysis_reports')

    validator = SpatialSplitValidator(labels_csv, e07_splits)
    validator.run_validation(output_dir)
    validator.save_results(output_dir / 'spatial_split_validation_results.json')
