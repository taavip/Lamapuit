#!/usr/bin/env python3
"""
Data Validation Script
Confirms data integrity, structure, and quality metrics for CWD detection thesis.
Generates validation report with key statistics.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from collections import Counter
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataValidator:
    def __init__(self, labels_csv_path):
        """Initialize validator with canonical labels CSV."""
        self.labels_path = Path(labels_csv_path)
        self.df = None
        self.validation_results = {}

    def load_data(self):
        """Load and validate CSV structure."""
        logger.info(f"Loading data from {self.labels_path}")
        self.df = pd.read_csv(self.labels_path)
        logger.info(f"Loaded {len(self.df)} records")
        return self.df

    def validate_structure(self):
        """Validate required columns exist."""
        required_cols = ['raster', 'row_off', 'col_off', 'label', 'source', 'year', 'map_sheet']
        missing = [col for col in required_cols if col not in self.df.columns]

        if missing:
            logger.error(f"Missing required columns: {missing}")
            return False

        logger.info("✓ All required columns present")
        self.validation_results['structure_valid'] = True
        return True

    def validate_coordinates(self):
        """Check coordinate validity (non-negative integers)."""
        valid = (self.df['row_off'] >= 0) & (self.df['col_off'] >= 0)
        invalid_count = (~valid).sum()

        if invalid_count > 0:
            logger.warning(f"Found {invalid_count} invalid coordinates")
            self.validation_results['invalid_coordinates'] = invalid_count
        else:
            logger.info("✓ All coordinates valid")

        return invalid_count

    def validate_labels(self):
        """Check label values and distribution."""
        valid_labels = {'cdw', 'no_cdw'}
        invalid = ~self.df['label'].isin(valid_labels)
        invalid_count = invalid.sum()

        if invalid_count > 0:
            logger.warning(f"Found {invalid_count} invalid labels: {self.df[invalid]['label'].unique()}")
        else:
            logger.info("✓ All labels valid (cdw or no_cdw)")

        label_counts = self.df['label'].value_counts().to_dict()
        logger.info(f"Label distribution: {label_counts}")
        self.validation_results['label_counts'] = label_counts

        return invalid_count, label_counts

    def validate_duplicates(self):
        """Check for duplicate entries (same raster, row, col)."""
        dup_cols = ['raster', 'row_off', 'col_off']
        duplicates = self.df.duplicated(subset=dup_cols, keep=False)
        dup_count = duplicates.sum() // 2  # Each pair counted twice

        if dup_count > 0:
            logger.warning(f"Found {dup_count} duplicate tile locations")
            self.validation_results['duplicates'] = dup_count
        else:
            logger.info("✓ No duplicate tile locations")

        return dup_count

    def temporal_distribution(self):
        """Analyze temporal coverage."""
        year_counts = self.df['year'].value_counts().sort_index().to_dict()
        years_range = (self.df['year'].min(), self.df['year'].max())

        logger.info(f"Temporal range: {years_range[0]}-{years_range[1]}")
        logger.info(f"Year distribution: {year_counts}")

        self.validation_results['temporal_range'] = years_range
        self.validation_results['year_distribution'] = year_counts

        return year_counts

    def geographic_distribution(self):
        """Analyze geographic coverage (map sheets)."""
        mapsheet_counts = self.df['map_sheet'].value_counts().to_dict()
        num_mapsheets = len(mapsheet_counts)

        logger.info(f"Coverage: {num_mapsheets} map sheets")
        logger.info(f"Map sheet distribution: {dict(sorted(mapsheet_counts.items(),
                                                          key=lambda x: x[1], reverse=True)[:10])}")

        self.validation_results['num_mapsheets'] = num_mapsheets
        self.validation_results['mapsheet_counts'] = mapsheet_counts

        return mapsheet_counts

    def source_distribution(self):
        """Analyze label source (manual, auto, etc)."""
        source_counts = self.df['source'].value_counts().to_dict()

        logger.info(f"Label sources: {source_counts}")
        self.validation_results['source_distribution'] = source_counts

        return source_counts

    def missing_values_check(self):
        """Check for missing values in critical columns."""
        critical_cols = ['raster', 'row_off', 'col_off', 'label', 'year']
        missing = {}

        for col in critical_cols:
            missing_count = self.df[col].isna().sum()
            if missing_count > 0:
                logger.warning(f"Found {missing_count} missing values in {col}")
                missing[col] = missing_count

        if not missing:
            logger.info("✓ No missing values in critical columns")

        self.validation_results['missing_values'] = missing
        return missing

    def data_quality_summary(self):
        """Generate comprehensive quality summary."""
        summary = {
            'total_records': len(self.df),
            'unique_rasters': self.df['raster'].nunique(),
            'unique_mapsheets': self.df['map_sheet'].nunique(),
            'temporal_range': (int(self.df['year'].min()), int(self.df['year'].max())),
            'class_balance': {
                'cdw': int((self.df['label'] == 'cdw').sum()),
                'no_cdw': int((self.df['label'] == 'no_cdw').sum()),
                'imbalance_ratio': float((self.df['label'] == 'no_cdw').sum() /
                                        ((self.df['label'] == 'cdw').sum() + 1))
            },
            'coordinate_range_rows': (int(self.df['row_off'].min()), int(self.df['row_off'].max())),
            'coordinate_range_cols': (int(self.df['col_off'].min()), int(self.df['col_off'].max())),
        }

        logger.info("\n" + "="*60)
        logger.info("DATA QUALITY SUMMARY")
        logger.info("="*60)
        logger.info(f"Total records: {summary['total_records']:,}")
        logger.info(f"Unique rasters: {summary['unique_rasters']}")
        logger.info(f"Unique map sheets: {summary['unique_mapsheets']}")
        logger.info(f"Temporal range: {summary['temporal_range'][0]}-{summary['temporal_range'][1]}")
        logger.info(f"CWD samples: {summary['class_balance']['cdw']:,}")
        logger.info(f"Background samples: {summary['class_balance']['no_cdw']:,}")
        logger.info(f"Class imbalance ratio: {summary['class_balance']['imbalance_ratio']:.2f}:1")
        logger.info("="*60)

        self.validation_results['quality_summary'] = summary
        return summary

    def run_all_validations(self):
        """Run complete validation suite."""
        if self.df is None:
            self.load_data()

        logger.info("Starting validation suite...")
        self.validate_structure()
        self.validate_coordinates()
        self.validate_labels()
        self.validate_duplicates()
        self.temporal_distribution()
        self.geographic_distribution()
        self.source_distribution()
        self.missing_values_check()
        self.data_quality_summary()

        logger.info("Validation complete!")
        return self.validation_results

    def save_report(self, output_path):
        """Save validation report to JSON."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert non-serializable types recursively
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
                return obj

        report = convert_to_serializable(self.validation_results)

        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)

        logger.info(f"Report saved to {output_path}")


if __name__ == '__main__':
    # Paths
    labels_csv = Path('/home/tpipar/project/Lamapuit/output/onboarding_labels_v2_drop13_standardized/labels_canonical.csv')
    output_report = Path('/home/tpipar/project/Lamapuit/output/analysis_reports/data_validation_report.json')

    validator = DataValidator(labels_csv)
    validator.run_all_validations()
    validator.save_report(output_report)
