#!/usr/bin/env python3
"""
Class Distribution Analysis
Analyzes CWD vs background imbalance, per-area breakdown, and visualization data.
Produces statistics for thesis argument about data characteristics.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3

class ClassDistributionAnalyzer:
    def __init__(self, labels_csv_path):
        """Initialize analyzer."""
        self.labels_path = Path(labels_csv_path)
        self.df = pd.read_csv(self.labels_path)
        self.results = {}

        logger.info(f"Loaded {len(self.df)} records")

    def global_class_balance(self):
        """Calculate global class balance."""
        counts = self.df['label'].value_counts()
        total = len(self.df)

        balance = {
            'cdw': int(counts.get('cdw', 0)),
            'no_cdw': int(counts.get('no_cdw', 0)),
            'cdw_percent': float(counts.get('cdw', 0) / total * 100),
            'no_cdw_percent': float(counts.get('no_cdw', 0) / total * 100),
            'imbalance_ratio': float(counts.get('no_cdw', 1) / (counts.get('cdw', 1) + 1e-6))
        }

        logger.info("\n" + "="*60)
        logger.info("GLOBAL CLASS BALANCE")
        logger.info("="*60)
        logger.info(f"CWD (cdw): {balance['cdw']:,} ({balance['cdw_percent']:.2f}%)")
        logger.info(f"Background (no_cdw): {balance['no_cdw']:,} ({balance['no_cdw_percent']:.2f}%)")
        logger.info(f"Imbalance ratio: {balance['imbalance_ratio']:.2f}:1 (bg:cdw)")
        logger.info("="*60)

        self.results['global_balance'] = balance
        return balance

    def per_area_balance(self):
        """Analyze class distribution per map sheet."""
        per_area = self.df.groupby('map_sheet')['label'].value_counts().unstack(fill_value=0)

        if 'cdw' in per_area.columns:
            per_area['total'] = per_area.sum(axis=1)
            per_area['cdw_percent'] = (per_area['cdw'] / per_area['total'] * 100).round(2)
            per_area['imbalance_ratio'] = (per_area.get('no_cdw', 0) / per_area['cdw']).round(2)
            per_area = per_area.sort_values('cdw_percent', ascending=False)

            logger.info("\nCLASS BALANCE BY MAP SHEET (sorted by CWD percentage)")
            logger.info("-" * 80)
            logger.info(f"{'Map Sheet':>10} {'CWD':>8} {'Background':>12} {'Total':>8} {'CWD%':>8} {'Imbalance':>10}")
            logger.info("-" * 80)

            for idx, row in per_area.iterrows():
                logger.info(f"{idx:>10d} {int(row['cdw']):>8d} {int(row.get('no_cdw', 0)):>12d} "
                           f"{int(row['total']):>8d} {row['cdw_percent']:>7.2f}% {row['imbalance_ratio']:>10.2f}:1")

            self.results['per_area_balance'] = per_area.to_dict('index')
            return per_area
        else:
            logger.warning("No CWD samples found in any area!")
            return per_area

    def per_year_balance(self):
        """Analyze class distribution per year."""
        per_year = self.df.groupby('year')['label'].value_counts().unstack(fill_value=0)

        if 'cdw' in per_year.columns:
            per_year['total'] = per_year.sum(axis=1)
            per_year['cdw_percent'] = (per_year['cdw'] / per_year['total'] * 100).round(2)
            per_year['imbalance_ratio'] = (per_year.get('no_cdw', 0) / per_year['cdw']).round(2)

            logger.info("\nCLASS BALANCE BY YEAR")
            logger.info("-" * 80)
            logger.info(f"{'Year':>6} {'CWD':>8} {'Background':>12} {'Total':>8} {'CWD%':>8} {'Imbalance':>10}")
            logger.info("-" * 80)

            for year in sorted(per_year.index):
                row = per_year.loc[year]
                logger.info(f"{int(year):>6d} {int(row['cdw']):>8d} {int(row.get('no_cdw', 0)):>12d} "
                           f"{int(row['total']):>8d} {row['cdw_percent']:>7.2f}% {row['imbalance_ratio']:>10.2f}:1")

            self.results['per_year_balance'] = per_year.to_dict('index')
            return per_year
        else:
            return per_year

    def per_source_balance(self):
        """Analyze class distribution by label source."""
        per_source = self.df.groupby('source')['label'].value_counts().unstack(fill_value=0)

        if 'cdw' in per_source.columns:
            per_source['total'] = per_source.sum(axis=1)
            per_source['cdw_percent'] = (per_source['cdw'] / per_source['total'] * 100).round(2)
            per_source = per_source.sort_values('cdw_percent', ascending=False)

            logger.info("\nCLASS BALANCE BY LABEL SOURCE")
            logger.info("-" * 80)
            logger.info(f"{'Source':>15} {'CWD':>8} {'Background':>12} {'Total':>8} {'CWD%':>8}")
            logger.info("-" * 80)

            for source in per_source.index:
                row = per_source.loc[source]
                logger.info(f"{source:>15} {int(row['cdw']):>8d} {int(row.get('no_cdw', 0)):>12d} "
                           f"{int(row['total']):>8d} {row['cdw_percent']:>7.2f}%")

            self.results['per_source_balance'] = per_source.to_dict('index')
            return per_source
        else:
            return per_source

    def visualize_class_distribution(self, output_dir):
        """Create comprehensive visualization of class distribution."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # 1. Global class distribution
        ax = axes[0, 0]
        counts = self.df['label'].value_counts()
        colors = ['#d62728', '#2ca02c']  # Red for CWD, Green for background
        bars = ax.bar(counts.index, counts.values, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
        ax.set_ylabel('Number of Samples', fontsize=12, fontweight='bold')
        ax.set_title('Global Class Distribution', fontsize=13, fontweight='bold')
        ax.set_yscale('log')

        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height):,}',
                   ha='center', va='bottom', fontweight='bold')

        # 2. Imbalance ratio per map sheet
        ax = axes[0, 1]
        per_area = self.df.groupby('map_sheet')['label'].value_counts().unstack(fill_value=0)
        if 'cdw' in per_area.columns:
            per_area['ratio'] = per_area.get('no_cdw', 0) / per_area['cdw']
            per_area = per_area.sort_values('ratio', ascending=False)
            ax.barh(range(len(per_area)), per_area['ratio'], color='steelblue', alpha=0.7, edgecolor='black')
            ax.set_yticks(range(len(per_area)))
            ax.set_yticklabels([f"Map {int(idx)}" for idx in per_area.index], fontsize=9)
            ax.set_xlabel('Imbalance Ratio (no_cdw / cdw)', fontsize=11, fontweight='bold')
            ax.set_title('Class Imbalance by Map Sheet', fontsize=13, fontweight='bold')
            ax.axvline(x=self.results['global_balance']['imbalance_ratio'], color='red', linestyle='--',
                      linewidth=2, label='Global ratio')
            ax.legend()

        # 3. Temporal trend
        ax = axes[1, 0]
        per_year = self.df.groupby('year')['label'].value_counts().unstack(fill_value=0)
        if 'cdw' in per_year.columns:
            per_year['cdw_percent'] = per_year['cdw'] / per_year.sum(axis=1) * 100
            ax.plot(per_year.index, per_year['cdw_percent'], marker='o', linewidth=2.5,
                   markersize=8, color='#d62728', label='CWD %')
            ax.fill_between(per_year.index, per_year['cdw_percent'], alpha=0.3, color='#d62728')
            ax.set_xlabel('Year', fontsize=11, fontweight='bold')
            ax.set_ylabel('CWD Percentage (%)', fontsize=11, fontweight='bold')
            ax.set_title('CWD Percentage Trend Over Years', fontsize=13, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend()

        # 4. Source distribution
        ax = axes[1, 1]
        per_source = self.df['source'].value_counts()
        colors_source = plt.cm.Set3(range(len(per_source)))
        wedges, texts, autotexts = ax.pie(per_source.values, labels=per_source.index, autopct='%1.1f%%',
                                          colors=colors_source, startangle=90, textprops={'fontsize': 10})
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        ax.set_title('Label Source Distribution', fontsize=13, fontweight='bold')

        plt.tight_layout()
        output_path = output_dir / 'class_distribution_analysis.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"✓ Saved visualization to {output_path}")
        plt.close()

    def run_analysis(self, output_dir=None):
        """Run complete analysis."""
        logger.info("Starting class distribution analysis...")

        self.global_class_balance()
        self.per_area_balance()
        self.per_year_balance()
        self.per_source_balance()

        if output_dir:
            self.visualize_class_distribution(output_dir)

        logger.info("Analysis complete!")
        return self.results

    def save_results(self, output_path):
        """Save analysis results to JSON."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Serialize results recursively
        def convert_to_serializable(obj):
            if isinstance(obj, dict):
                return {str(k): convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_to_serializable(item) for item in obj]
            elif isinstance(obj, (np.integer, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64)):
                return float(obj)
            elif isinstance(obj, pd.DataFrame):
                return obj.to_dict('index')
            elif isinstance(obj, pd.Series):
                return obj.to_dict()
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return obj

        serializable = convert_to_serializable(self.results)

        with open(output_path, 'w') as f:
            json.dump(serializable, f, indent=2)

        logger.info(f"✓ Results saved to {output_path}")


if __name__ == '__main__':
    labels_csv = Path('/home/tpipar/project/Lamapuit/output/onboarding_labels_v2_drop13_standardized/labels_canonical.csv')
    output_dir = Path('/home/tpipar/project/Lamapuit/output/analysis_reports')

    analyzer = ClassDistributionAnalyzer(labels_csv)
    analyzer.run_analysis(output_dir)
    analyzer.save_results(output_dir / 'class_distribution_results.json')
