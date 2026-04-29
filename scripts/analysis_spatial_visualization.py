#!/usr/bin/env python3
"""
Spatial Analysis and Visualization
Visualizes geographic distribution of labels, creates maps, spatial statistics.
Produces publication-ready figures for thesis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SpatialAnalyzer:
    def __init__(self, labels_csv_path):
        """Initialize spatial analyzer."""
        self.labels_path = Path(labels_csv_path)
        self.df = pd.read_csv(self.labels_path)
        self.results = {}

        # Parse map sheet and year from raster names
        self._parse_raster_info()

        logger.info(f"Loaded {len(self.df)} records from {self.labels_path.name}")

    def _parse_raster_info(self):
        """Parse map sheet, year, and landscape type from raster names."""
        # Format: "MAPSHEET_YEAR_TYPE_chm_max_hag_20cm.tif"
        self.df['mapsheet'] = self.df['raster'].str.split('_').str[0]
        self.df['year_parsed'] = self.df['raster'].str.split('_').str[1]
        self.df['landscape'] = self.df['raster'].str.split('_').str[2]  # e.g., 'tava', 'mets', 'madal'

    def mapsheet_statistics(self):
        """Calculate statistics per map sheet."""
        stats = []

        for mapsheet in sorted(self.df['mapsheet'].unique()):
            subset = self.df[self.df['mapsheet'] == mapsheet]
            cdw_count = (subset['label'] == 'cdw').sum()
            total = len(subset)
            cdw_pct = 100 * cdw_count / total if total > 0 else 0

            stats.append({
                'map_sheet': int(mapsheet),
                'total_samples': total,
                'cdw_samples': int(cdw_count),
                'cdw_percentage': float(cdw_pct),
                'num_rasters': subset['raster'].nunique(),
                'num_years': subset['year'].nunique(),
                'year_range': f"{int(subset['year'].min())}-{int(subset['year'].max())}"
            })

        stats_df = pd.DataFrame(stats).sort_values('cdw_percentage', ascending=False)

        logger.info("\n" + "="*90)
        logger.info("SPATIAL DISTRIBUTION BY MAP SHEET")
        logger.info("="*90)
        logger.info(f"{'Map Sheet':>12} {'Total':>10} {'CWD':>8} {'CWD%':>8} {'Rasters':>10} {'Years':>8}")
        logger.info("-"*90)

        for _, row in stats_df.iterrows():
            logger.info(f"{int(row['map_sheet']):>12d} {row['total_samples']:>10d} "
                       f"{int(row['cdw_samples']):>8d} {row['cdw_percentage']:>7.2f}% "
                       f"{int(row['num_rasters']):>10d} {row['year_range']:>8}")

        self.results['mapsheet_stats'] = stats
        return stats_df

    def landscape_type_statistics(self):
        """Analyze distribution by landscape type (if available)."""
        landscape_types = self.df['landscape'].unique()

        stats = []
        logger.info("\nCLASS DISTRIBUTION BY LANDSCAPE TYPE")
        logger.info("-" * 70)

        for landscape in sorted(landscape_types):
            subset = self.df[self.df['landscape'] == landscape]
            cdw_count = (subset['label'] == 'cdw').sum()
            total = len(subset)
            cdw_pct = 100 * cdw_count / total if total > 0 else 0

            stats.append({
                'landscape': landscape,
                'total_samples': total,
                'cdw_samples': int(cdw_count),
                'cdw_percentage': float(cdw_pct)
            })

            logger.info(f"{landscape:>15}: {total:>10,} samples, "
                       f"CWD: {int(cdw_count):>8,} ({cdw_pct:>6.2f}%)")

        self.results['landscape_stats'] = stats
        return pd.DataFrame(stats)

    def spatial_density_heatmap(self, output_dir):
        """Create heatmap of CWD density by map sheet and year."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create pivot table: map sheets x years, values = CWD percentage
        pivot_data = self.df.copy()
        pivot_data['mapsheet_int'] = pivot_data['mapsheet'].astype(int)
        pivot_data['year_int'] = pivot_data['year'].astype(int)

        cdw_pct = pivot_data[pivot_data['label'] == 'cdw'].groupby(['mapsheet_int', 'year_int']).size()
        total = pivot_data.groupby(['mapsheet_int', 'year_int']).size()
        density = (cdw_pct / total * 100).unstack(fill_value=0)

        fig, ax = plt.subplots(figsize=(14, 10))

        # Create heatmap using imshow
        im = ax.imshow(density.values, cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=100)

        # Set labels and ticks
        ax.set_xticks(range(len(density.columns)))
        ax.set_yticks(range(len(density.index)))
        ax.set_xticklabels(density.columns)
        ax.set_yticklabels(density.index)

        # Annotate cells with values
        for i in range(len(density.index)):
            for j in range(len(density.columns)):
                val = density.values[i, j]
                if val > 0:
                    text_color = 'white' if val > 50 else 'black'
                    ax.text(j, i, f'{val:.1f}', ha='center', va='center', color=text_color, fontsize=9)

        ax.set_xlabel('Year', fontsize=12, fontweight='bold')
        ax.set_ylabel('Map Sheet', fontsize=12, fontweight='bold')
        ax.set_title('CWD Sample Density (%) by Map Sheet and Year', fontsize=14, fontweight='bold')

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('CWD Percentage (%)', fontsize=11, fontweight='bold')

        plt.tight_layout()
        output_path = output_dir / 'spatial_density_heatmap.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"✓ Saved heatmap to {output_path}")
        plt.close()

    def coordinate_distribution_plot(self, output_dir):
        """Visualize spatial distribution of samples within tiles."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # CWD samples spatial distribution
        ax = axes[0]
        cdw_data = self.df[self.df['label'] == 'cdw']
        ax.hexbin(cdw_data['col_off'], cdw_data['row_off'], gridsize=30, cmap='Reds', mincnt=1)
        ax.set_xlabel('Column Offset (pixels)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Row Offset (pixels)', fontsize=11, fontweight='bold')
        ax.set_title(f'CWD Sample Spatial Distribution (n={len(cdw_data):,})', fontsize=12, fontweight='bold')
        ax.set_aspect('equal')

        # Background samples spatial distribution
        ax = axes[1]
        bg_data = self.df[self.df['label'] == 'no_cdw']
        ax.hexbin(bg_data['col_off'], bg_data['row_off'], gridsize=30, cmap='Blues', mincnt=1)
        ax.set_xlabel('Column Offset (pixels)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Row Offset (pixels)', fontsize=11, fontweight='bold')
        ax.set_title(f'Background Sample Spatial Distribution (n={len(bg_data):,})', fontsize=12, fontweight='bold')
        ax.set_aspect('equal')

        plt.tight_layout()
        output_path = output_dir / 'coordinate_distribution.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"✓ Saved coordinate distribution to {output_path}")
        plt.close()

    def coverage_summary_visualization(self, output_dir):
        """Create summary visualization of data coverage."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # 1. Samples per map sheet
        ax = axes[0, 0]
        mapsheet_counts = self.df['mapsheet'].value_counts().sort_index()
        mapsheet_int = mapsheet_counts.index.astype(int)
        colors = plt.cm.viridis(np.linspace(0, 1, len(mapsheet_counts)))
        ax.bar(range(len(mapsheet_counts)), mapsheet_counts.values, color=colors, edgecolor='black', linewidth=1)
        ax.set_xticks(range(len(mapsheet_counts)))
        ax.set_xticklabels([f"{int(ms)}" for ms in mapsheet_int], rotation=45, ha='right')
        ax.set_ylabel('Number of Samples', fontsize=11, fontweight='bold')
        ax.set_xlabel('Map Sheet', fontsize=11, fontweight='bold')
        ax.set_title('Sample Distribution Across Map Sheets', fontsize=12, fontweight='bold')
        ax.set_yscale('log')

        # 2. Samples per year
        ax = axes[0, 1]
        year_counts = self.df['year'].value_counts().sort_index()
        ax.plot(year_counts.index, year_counts.values, marker='o', linewidth=2.5, markersize=8, color='steelblue')
        ax.fill_between(year_counts.index, year_counts.values, alpha=0.3, color='steelblue')
        ax.set_ylabel('Number of Samples', fontsize=11, fontweight='bold')
        ax.set_xlabel('Year', fontsize=11, fontweight='bold')
        ax.set_title('Temporal Coverage (Samples per Year)', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')

        # 3. Rasters per map sheet
        ax = axes[1, 0]
        rasters_per_sheet = self.df.groupby('mapsheet')['raster'].nunique().sort_index()
        rasters_per_sheet.index = rasters_per_sheet.index.astype(int)
        ax.barh(range(len(rasters_per_sheet)), rasters_per_sheet.values, color='coral', edgecolor='black')
        ax.set_yticks(range(len(rasters_per_sheet)))
        ax.set_yticklabels([f"Map {int(ms)}" for ms in rasters_per_sheet.index], fontsize=10)
        ax.set_xlabel('Number of Rasters', fontsize=11, fontweight='bold')
        ax.set_title('Raster Coverage per Map Sheet', fontsize=12, fontweight='bold')

        # 4. Geographic extent (map sheet coordinates)
        ax = axes[1, 1]
        mapsheet_ints = self.df['mapsheet'].astype(int).unique()
        mapsheet_ints = np.sort(mapsheet_ints)

        # Simple visualization: map sheet ID as proxy for location
        counts_by_sheet = self.df['mapsheet'].astype(int).value_counts().sort_index()
        x_pos = np.arange(len(counts_by_sheet))
        colors_spatial = plt.cm.RdYlGn_r(np.linspace(0, 1, len(counts_by_sheet)))
        ax.scatter(x_pos, counts_by_sheet.values, s=500, c=range(len(counts_by_sheet)), cmap='tab20',
                  edgecolor='black', linewidth=1.5, alpha=0.7)
        ax.set_xticks(x_pos)
        ax.set_xticklabels([f"{int(ms)}" for ms in counts_by_sheet.index], rotation=45, ha='right')
        ax.set_ylabel('Number of Samples (log scale)', fontsize=11, fontweight='bold')
        ax.set_xlabel('Map Sheet', fontsize=11, fontweight='bold')
        ax.set_title('Geographic Coverage Overview', fontsize=12, fontweight='bold')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        output_path = output_dir / 'coverage_summary.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"✓ Saved coverage summary to {output_path}")
        plt.close()

    def run_analysis(self, output_dir=None):
        """Run complete spatial analysis."""
        logger.info("Starting spatial analysis...")

        self.mapsheet_statistics()
        self.landscape_type_statistics()

        if output_dir:
            output_dir = Path(output_dir)
            self.spatial_density_heatmap(output_dir)
            self.coordinate_distribution_plot(output_dir)
            self.coverage_summary_visualization(output_dir)

        logger.info("Spatial analysis complete!")
        return self.results

    def save_results(self, output_path):
        """Save results to JSON."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        def convert_to_serializable(obj):
            if isinstance(obj, dict):
                return {str(k): convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_to_serializable(item) for item in obj]
            elif isinstance(obj, (np.integer, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return str(obj) if not isinstance(obj, (str, bool, type(None))) else obj

        serializable = convert_to_serializable(self.results)

        with open(output_path, 'w') as f:
            json.dump(serializable, f, indent=2)

        logger.info(f"✓ Results saved to {output_path}")


if __name__ == '__main__':
    labels_csv = Path('/home/tpipar/project/Lamapuit/output/onboarding_labels_v2_drop13_standardized/labels_canonical.csv')
    output_dir = Path('/home/tpipar/project/Lamapuit/output/analysis_reports')

    analyzer = SpatialAnalyzer(labels_csv)
    analyzer.run_analysis(output_dir)
    analyzer.save_results(output_dir / 'spatial_analysis_results.json')
