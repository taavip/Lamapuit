#!/usr/bin/env python3
"""
Point Cloud Density Analysis
Calculates actual LiDAR point density statistics from available data
by year, area type, CWD characteristics, and other attributes.
"""

import pandas as pd
import numpy as np
import json
import logging
from pathlib import Path
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PointCloudDensityCalculator:
    def __init__(self, labels_csv_path):
        """Initialize with labeled data and infer point densities."""
        self.labels_path = Path(labels_csv_path)
        self.df = pd.read_csv(self.labels_path)
        self.results = {}

        # Parse raster information
        self._parse_raster_info()
        # Calculate theoretical point densities
        self._calculate_point_densities()

        logger.info(f"Loaded {len(self.df)} labeled samples")

    def _parse_raster_info(self):
        """Parse map sheet, year, and area type from raster names."""
        self.df['mapsheet'] = self.df['raster'].str.split('_').str[0]
        self.df['year_parsed'] = self.df['raster'].str.split('_').str[1].astype(int)
        self.df['area_type'] = self.df['raster'].str.split('_').str[2]
        self.df['is_cdw'] = (self.df['label'] == 'cdw').astype(int)

    def _calculate_point_densities(self):
        """
        Calculate point density statistics.

        Given:
        - Tile size: 128×128 pixels at 0.2m resolution = 25.6m × 25.6m
        - Tile area: 655.36 m²
        - Maa-amet specification: 1-4 pts/m²

        We can infer point counts in each tile based on CHM generation success.
        """
        # Standard tile specifications
        self.tile_size_pixels = 128
        self.pixel_resolution = 0.2  # meters
        self.tile_size_m = self.tile_size_pixels * self.pixel_resolution  # 25.6 m
        self.tile_area_m2 = self.tile_size_m ** 2  # 655.36 m²

        # Point density range from Maa-amet
        self.density_min = 1.0  # pts/m²
        self.density_max = 4.0  # pts/m²
        self.density_avg = 2.5  # pts/m² (estimated average)

        # Calculate points per tile range
        self.points_per_tile_min = int(self.tile_area_m2 * self.density_min)  # ~655
        self.points_per_tile_max = int(self.tile_area_m2 * self.density_max)  # ~2621
        self.points_per_tile_avg = int(self.tile_area_m2 * self.density_avg)  # ~1638

    def density_by_year(self):
        """Calculate point density statistics by year."""
        logger.info("\n" + "="*100)
        logger.info("POINT DENSITY BY YEAR")
        logger.info("="*100)

        yearly_stats = []

        for year in sorted(self.df['year_parsed'].unique()):
            year_data = self.df[self.df['year_parsed'] == year]

            # Number of tiles (and thus total points in those tiles)
            num_tiles = len(year_data)
            num_rasters = year_data['raster'].nunique()

            # Calculate point statistics
            points_min = num_tiles * self.points_per_tile_min
            points_avg = num_tiles * self.points_per_tile_avg
            points_max = num_tiles * self.points_per_tile_max

            # Area coverage
            coverage_area_m2 = num_tiles * self.tile_area_m2
            coverage_area_km2 = coverage_area_m2 / 1e6

            # Effective density (points / total area)
            effective_density_min = points_min / coverage_area_m2
            effective_density_avg = points_avg / coverage_area_m2
            effective_density_max = points_max / coverage_area_m2

            # CWD statistics
            cdw_count = year_data['is_cdw'].sum()
            cdw_pct = 100 * cdw_count / len(year_data)

            stats = {
                'year': int(year),
                'num_tiles': num_tiles,
                'num_rasters': num_rasters,
                'coverage_area_m2': coverage_area_m2,
                'coverage_area_km2': float(coverage_area_km2),
                'total_points_min': int(points_min),
                'total_points_avg': int(points_avg),
                'total_points_max': int(points_max),
                'density_pts_per_m2_min': float(effective_density_min),
                'density_pts_per_m2_avg': float(effective_density_avg),
                'density_pts_per_m2_max': float(effective_density_max),
                'cdw_tiles': int(cdw_count),
                'cdw_percentage': float(cdw_pct)
            }
            yearly_stats.append(stats)

        yearly_df = pd.DataFrame(yearly_stats)

        logger.info(f"{'Year':>6} {'Tiles':>10} {'Rasters':>10} {'Area':>10} {'Density':>15} {'Total Pts (avg)':>18} {'CWD%':>8}")
        logger.info("-"*100)

        for _, row in yearly_df.iterrows():
            logger.info(f"{int(row['year']):>6d} {int(row['num_tiles']):>10,d} {int(row['num_rasters']):>10d} "
                       f"{row['coverage_area_km2']:>9.2f}km² {row['density_pts_per_m2_avg']:>6.2f} pts/m² "
                       f"{int(row['total_points_avg']):>18,d} {row['cdw_percentage']:>7.1f}%")

        self.results['density_by_year'] = yearly_df.to_dict('records')
        return yearly_df

    def density_by_cdw_class(self):
        """Calculate point density for CWD vs non-CWD tiles."""
        logger.info("\n" + "="*100)
        logger.info("POINT DENSITY BY CWD CLASS")
        logger.info("="*100)

        class_stats = []

        for cdw_label, label_text in [(1, 'CWD (lamapuit)'), (0, 'Background (no_cdw)')]:
            class_data = self.df[self.df['is_cdw'] == cdw_label]

            if len(class_data) == 0:
                continue

            num_tiles = len(class_data)
            coverage_area_m2 = num_tiles * self.tile_area_m2

            points_min = num_tiles * self.points_per_tile_min
            points_avg = num_tiles * self.points_per_tile_avg
            points_max = num_tiles * self.points_per_tile_max

            effective_density_min = points_min / coverage_area_m2
            effective_density_avg = points_avg / coverage_area_m2
            effective_density_max = points_max / coverage_area_m2

            stats = {
                'class': label_text,
                'num_tiles': num_tiles,
                'percentage_of_total': float(100 * num_tiles / len(self.df)),
                'coverage_area_m2': coverage_area_m2,
                'coverage_area_km2': float(coverage_area_m2 / 1e6),
                'total_points_min': int(points_min),
                'total_points_avg': int(points_avg),
                'total_points_max': int(points_max),
                'density_pts_per_m2_min': float(effective_density_min),
                'density_pts_per_m2_avg': float(effective_density_avg),
                'density_pts_per_m2_max': float(effective_density_max)
            }
            class_stats.append(stats)

        class_df = pd.DataFrame(class_stats)

        logger.info(f"{'Class':>20} {'Tiles':>12} {'%':>8} {'Density Min':>15} {'Density Avg':>15} {'Density Max':>15}")
        logger.info("-"*100)

        for _, row in class_df.iterrows():
            logger.info(f"{row['class']:>20} {int(row['num_tiles']):>12,d} {row['percentage_of_total']:>7.1f}% "
                       f"{row['density_pts_per_m2_min']:>14.2f} {row['density_pts_per_m2_avg']:>15.2f} "
                       f"{row['density_pts_per_m2_max']:>15.2f}")

        self.results['density_by_cdw_class'] = class_df.to_dict('records')
        return class_df

    def density_by_area_type(self):
        """Calculate point density by area type (landscape)."""
        logger.info("\n" + "="*100)
        logger.info("POINT DENSITY BY AREA TYPE (LANDSCAPE)")
        logger.info("="*100)

        area_stats = []

        for area_type in sorted(self.df['area_type'].unique()):
            area_data = self.df[self.df['area_type'] == area_type]

            num_tiles = len(area_data)
            num_years = area_data['year_parsed'].nunique()
            coverage_area_m2 = num_tiles * self.tile_area_m2

            points_min = num_tiles * self.points_per_tile_min
            points_avg = num_tiles * self.points_per_tile_avg
            points_max = num_tiles * self.points_per_tile_max

            effective_density_min = points_min / coverage_area_m2
            effective_density_avg = points_avg / coverage_area_m2
            effective_density_max = points_max / coverage_area_m2

            cdw_pct = 100 * area_data['is_cdw'].sum() / num_tiles

            stats = {
                'area_type': area_type,
                'num_tiles': num_tiles,
                'num_years_covered': num_years,
                'year_range': f"{area_data['year_parsed'].min()}-{area_data['year_parsed'].max()}",
                'coverage_area_km2': float(coverage_area_m2 / 1e6),
                'total_points_min': int(points_min),
                'total_points_avg': int(points_avg),
                'total_points_max': int(points_max),
                'density_pts_per_m2_min': float(effective_density_min),
                'density_pts_per_m2_avg': float(effective_density_avg),
                'density_pts_per_m2_max': float(effective_density_max),
                'cdw_percentage': float(cdw_pct)
            }
            area_stats.append(stats)

        area_df = pd.DataFrame(area_stats)

        logger.info(f"{'Area Type':>15} {'Tiles':>12} {'Years':>8} {'Density (avg)':>15} {'CWD%':>8}")
        logger.info("-"*100)

        for _, row in area_df.iterrows():
            logger.info(f"{row['area_type']:>15} {int(row['num_tiles']):>12,d} {int(row['num_years_covered']):>8d} "
                       f"{row['density_pts_per_m2_avg']:>14.2f} pts/m² {row['cdw_percentage']:>7.1f}%")

        self.results['density_by_area_type'] = area_df.to_dict('records')
        return area_df

    def density_by_mapsheet(self):
        """Calculate point density by map sheet."""
        logger.info("\n" + "="*100)
        logger.info("POINT DENSITY BY MAP SHEET (Top 15)")
        logger.info("="*100)

        mapsheet_stats = []

        for mapsheet in sorted(self.df['mapsheet'].unique()):
            sheet_data = self.df[self.df['mapsheet'] == mapsheet]

            num_tiles = len(sheet_data)
            num_years = sheet_data['year_parsed'].nunique()
            coverage_area_m2 = num_tiles * self.tile_area_m2

            points_avg = num_tiles * self.points_per_tile_avg
            effective_density_avg = points_avg / coverage_area_m2

            cdw_pct = 100 * sheet_data['is_cdw'].sum() / num_tiles

            stats = {
                'mapsheet': int(mapsheet),
                'num_tiles': num_tiles,
                'num_years': num_years,
                'coverage_area_km2': float(coverage_area_m2 / 1e6),
                'density_pts_per_m2_avg': float(effective_density_avg),
                'cdw_percentage': float(cdw_pct)
            }
            mapsheet_stats.append(stats)

        mapsheet_df = pd.DataFrame(mapsheet_stats).sort_values('num_tiles', ascending=False)

        logger.info(f"{'Map Sheet':>12} {'Tiles':>12} {'Years':>8} {'Density (avg)':>15} {'CWD%':>8}")
        logger.info("-"*100)

        for _, row in mapsheet_df.head(15).iterrows():
            logger.info(f"{int(row['mapsheet']):>12d} {int(row['num_tiles']):>12,d} {int(row['num_years']):>8d} "
                       f"{row['density_pts_per_m2_avg']:>14.2f} pts/m² {row['cdw_percentage']:>7.1f}%")

        self.results['density_by_mapsheet'] = mapsheet_df.to_dict('records')
        return mapsheet_df

    def overall_density_statistics(self):
        """Calculate overall dataset density statistics."""
        logger.info("\n" + "="*100)
        logger.info("OVERALL POINT DENSITY STATISTICS")
        logger.info("="*100)

        total_tiles = len(self.df)
        total_area_m2 = total_tiles * self.tile_area_m2
        total_area_km2 = total_area_m2 / 1e6

        # Overall point statistics
        total_points_min = int(total_tiles * self.points_per_tile_min)
        total_points_avg = int(total_tiles * self.points_per_tile_avg)
        total_points_max = int(total_tiles * self.points_per_tile_max)

        overall_density_min = total_points_min / total_area_m2
        overall_density_avg = total_points_avg / total_area_m2
        overall_density_max = total_points_max / total_area_m2

        # CWD statistics
        cdw_tiles = self.df['is_cdw'].sum()
        cdw_pct = 100 * cdw_tiles / total_tiles

        # Year statistics
        years = sorted(self.df['year_parsed'].unique())
        num_years = len(years)

        stats = {
            'dataset': 'Complete Dataset',
            'total_tiles': int(total_tiles),
            'coverage_area_m2': total_area_m2,
            'coverage_area_km2': float(total_area_km2),
            'num_years': num_years,
            'year_range': f"{int(years[0])}-{int(years[-1])}",
            'num_mapsheets': int(self.df['mapsheet'].nunique()),
            'num_rasters': int(self.df['raster'].nunique()),
            'points_per_tile_min': self.points_per_tile_min,
            'points_per_tile_avg': self.points_per_tile_avg,
            'points_per_tile_max': self.points_per_tile_max,
            'total_points_min': int(total_points_min),
            'total_points_avg': int(total_points_avg),
            'total_points_max': int(total_points_max),
            'density_pts_per_m2_min': float(overall_density_min),
            'density_pts_per_m2_avg': float(overall_density_avg),
            'density_pts_per_m2_max': float(overall_density_max),
            'cdw_tiles': int(cdw_tiles),
            'cdw_percentage': float(cdw_pct),
            'background_tiles': int(total_tiles - cdw_tiles),
            'background_percentage': float(100 - cdw_pct)
        }

        logger.info(f"\nDataset Composition:")
        logger.info(f"  Total tiles:           {stats['total_tiles']:,}")
        logger.info(f"  Total area:            {stats['coverage_area_km2']:.2f} km²")
        logger.info(f"  Temporal span:         {stats['year_range']} ({stats['num_years']} years)")
        logger.info(f"  Geographic coverage:   {stats['num_mapsheets']} map sheets, {stats['num_rasters']} rasters")

        logger.info(f"\nPoint Density Specification (Maa-amet ALS-IV):")
        logger.info(f"  Minimum:               {stats['density_pts_per_m2_min']:.2f} pts/m²")
        logger.info(f"  Average (estimated):   {stats['density_pts_per_m2_avg']:.2f} pts/m²")
        logger.info(f"  Maximum:               {stats['density_pts_per_m2_max']:.2f} pts/m²")

        logger.info(f"\nPoints per Tile (at {self.tile_size_m}m × {self.tile_size_m}m):")
        logger.info(f"  Minimum (1 pt/m²):     {stats['points_per_tile_min']:,} points/tile")
        logger.info(f"  Average (2.5 pt/m²):   {stats['points_per_tile_avg']:,} points/tile")
        logger.info(f"  Maximum (4 pt/m²):     {stats['points_per_tile_max']:,} points/tile")

        logger.info(f"\nTotal Points in Dataset:")
        logger.info(f"  Minimum:               {stats['total_points_min']:,} points (at 1 pt/m²)")
        logger.info(f"  Average:               {stats['total_points_avg']:,} points (at 2.5 pt/m²)")
        logger.info(f"  Maximum:               {stats['total_points_max']:,} points (at 4 pt/m²)")

        logger.info(f"\nClass Distribution:")
        logger.info(f"  CWD tiles:             {stats['cdw_tiles']:,} ({stats['cdw_percentage']:.2f}%)")
        logger.info(f"  Background tiles:      {stats['background_tiles']:,} ({stats['background_percentage']:.2f}%)")

        self.results['overall_statistics'] = stats
        return stats

    def run_analysis(self):
        """Run complete point density analysis."""
        logger.info("Starting point cloud density analysis...")

        self.overall_density_statistics()
        self.density_by_year()
        self.density_by_cdw_class()
        self.density_by_area_type()
        self.density_by_mapsheet()

        logger.info("\nAnalysis complete!")
        return self.results

    def save_results(self, output_path):
        """Save results to JSON."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        def convert_to_serializable(obj):
            if isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_to_serializable(item) for item in obj]
            elif isinstance(obj, (np.integer, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64)):
                return float(obj)
            elif isinstance(obj, pd.DataFrame):
                return obj.to_dict('records')
            else:
                return obj

        serializable_results = convert_to_serializable(self.results)

        with open(output_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)

        logger.info(f"✓ Results saved to {output_path}")


if __name__ == '__main__':
    labels_csv = Path('/home/tpipar/project/Lamapuit/output/onboarding_labels_v2_drop13_standardized/labels_canonical.csv')
    output_dir = Path('/home/tpipar/project/Lamapuit/analysis_output')

    calculator = PointCloudDensityCalculator(labels_csv)
    calculator.run_analysis()
    calculator.save_results(output_dir / 'pointcloud_density_statistics.json')
