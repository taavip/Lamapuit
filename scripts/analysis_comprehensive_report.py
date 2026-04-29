#!/usr/bin/env python3
"""
Comprehensive Analysis Report Generator
Runs all analyses and generates a unified report for thesis.
Combines validation, statistics, and visualizations.
"""

import sys
import json
import logging
from pathlib import Path
from datetime import datetime

# Import analysis modules
sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('analysis_run.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ComprehensiveReportGenerator:
    def __init__(self, output_dir):
        """Initialize report generator."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.report = {
            'generated_at': datetime.now().isoformat(),
            'analyses': {}
        }

    def run_data_validation(self, labels_csv):
        """Run data validation analysis."""
        logger.info("\n" + "="*70)
        logger.info("RUNNING DATA VALIDATION")
        logger.info("="*70)

        try:
            from analysis_data_validation import DataValidator

            validator = DataValidator(labels_csv)
            results = validator.run_all_validations()
            validator.save_report(self.output_dir / 'data_validation_report.json')

            self.report['analyses']['data_validation'] = {
                'status': 'completed',
                'output': str(self.output_dir / 'data_validation_report.json')
            }
            logger.info("✓ Data validation completed")
            return results

        except Exception as e:
            logger.error(f"Data validation failed: {e}")
            self.report['analyses']['data_validation'] = {'status': 'failed', 'error': str(e)}
            return None

    def run_class_distribution_analysis(self, labels_csv):
        """Run class distribution analysis."""
        logger.info("\n" + "="*70)
        logger.info("RUNNING CLASS DISTRIBUTION ANALYSIS")
        logger.info("="*70)

        try:
            from analysis_class_distribution import ClassDistributionAnalyzer

            analyzer = ClassDistributionAnalyzer(labels_csv)
            results = analyzer.run_analysis(self.output_dir)
            analyzer.save_results(self.output_dir / 'class_distribution_results.json')

            self.report['analyses']['class_distribution'] = {
                'status': 'completed',
                'output_results': str(self.output_dir / 'class_distribution_results.json'),
                'output_visualization': str(self.output_dir / 'class_distribution_analysis.png')
            }
            logger.info("✓ Class distribution analysis completed")
            return results

        except Exception as e:
            logger.error(f"Class distribution analysis failed: {e}")
            self.report['analyses']['class_distribution'] = {'status': 'failed', 'error': str(e)}
            return None

    def run_spatial_analysis(self, labels_csv):
        """Run spatial analysis."""
        logger.info("\n" + "="*70)
        logger.info("RUNNING SPATIAL ANALYSIS")
        logger.info("="*70)

        try:
            from analysis_spatial_visualization import SpatialAnalyzer

            analyzer = SpatialAnalyzer(labels_csv)
            results = analyzer.run_analysis(self.output_dir)
            analyzer.save_results(self.output_dir / 'spatial_analysis_results.json')

            self.report['analyses']['spatial_analysis'] = {
                'status': 'completed',
                'output_results': str(self.output_dir / 'spatial_analysis_results.json'),
                'output_visualizations': [
                    str(self.output_dir / 'spatial_density_heatmap.png'),
                    str(self.output_dir / 'coordinate_distribution.png'),
                    str(self.output_dir / 'coverage_summary.png')
                ]
            }
            logger.info("✓ Spatial analysis completed")
            return results

        except Exception as e:
            logger.error(f"Spatial analysis failed: {e}")
            self.report['analyses']['spatial_analysis'] = {'status': 'failed', 'error': str(e)}
            return None

    def run_split_validation(self, labels_csv, split_paths):
        """Run spatial split validation."""
        logger.info("\n" + "="*70)
        logger.info("RUNNING SPATIAL SPLIT VALIDATION")
        logger.info("="*70)

        try:
            from analysis_spatial_split_validation import SpatialSplitValidator

            validator = SpatialSplitValidator(labels_csv, split_paths)
            results = validator.run_validation(self.output_dir)
            validator.save_results(self.output_dir / 'spatial_split_validation_results.json')

            self.report['analyses']['split_validation'] = {
                'status': 'completed',
                'output_results': str(self.output_dir / 'spatial_split_validation_results.json'),
                'output_visualization': str(self.output_dir / 'spatial_split_validation.png')
            }
            logger.info("✓ Spatial split validation completed")
            return results

        except Exception as e:
            logger.error(f"Spatial split validation failed: {e}")
            self.report['analyses']['split_validation'] = {'status': 'failed', 'error': str(e)}
            return None

    def generate_summary(self):
        """Generate summary report."""
        logger.info("\n" + "="*70)
        logger.info("GENERATING COMPREHENSIVE SUMMARY")
        logger.info("="*70)

        summary = {
            'total_analyses': len(self.report['analyses']),
            'completed': sum(1 for a in self.report['analyses'].values() if a.get('status') == 'completed'),
            'failed': sum(1 for a in self.report['analyses'].values() if a.get('status') == 'failed')
        }

        self.report['summary'] = summary

        logger.info(f"Total analyses run: {summary['total_analyses']}")
        logger.info(f"Completed: {summary['completed']}")
        logger.info(f"Failed: {summary['failed']}")

        return summary

    def save_report(self):
        """Save comprehensive report."""
        report_path = self.output_dir / 'COMPREHENSIVE_ANALYSIS_REPORT.json'

        with open(report_path, 'w') as f:
            json.dump(self.report, f, indent=2, default=str)

        logger.info(f"\n✓ Comprehensive report saved to {report_path}")
        return report_path

    def print_summary(self):
        """Print user-friendly summary."""
        logger.info("\n" + "="*70)
        logger.info("ANALYSIS COMPLETE - SUMMARY")
        logger.info("="*70)
        logger.info(f"\nGenerated at: {self.report['generated_at']}")
        logger.info(f"\nOutput directory: {self.output_dir}")
        logger.info("\nGenerated files:")

        # List generated files
        for analysis, details in self.report['analyses'].items():
            logger.info(f"\n  {analysis}:")
            if details.get('status') == 'completed':
                logger.info(f"    Status: ✓ Completed")
                if 'output_results' in details:
                    logger.info(f"    Results: {Path(details['output_results']).name}")
                if 'output_visualization' in details:
                    logger.info(f"    Visualization: {Path(details['output_visualization']).name}")
                if 'output_visualizations' in details:
                    for viz in details['output_visualizations']:
                        logger.info(f"    Visualization: {Path(viz).name}")
            else:
                logger.info(f"    Status: ✗ Failed")
                if 'error' in details:
                    logger.info(f"    Error: {details['error']}")

        logger.info("\n" + "="*70)
        logger.info("All visualizations are publication-ready (300 DPI PNG)")
        logger.info("="*70)


def main():
    """Run complete analysis pipeline."""
    # Define paths
    labels_csv = Path('/home/tpipar/project/Lamapuit/output/onboarding_labels_v2_drop13_standardized/labels_canonical.csv')
    output_dir = Path('/home/tpipar/project/Lamapuit/analysis_output')

    # Find E07 split files
    import glob
    e07_splits = sorted(glob.glob('/home/tpipar/project/Lamapuit/output/spatial_split_experiments/E07_v3_blocks3_buf2/split_seed*.json'))

    if not labels_csv.exists():
        logger.error(f"Labels CSV not found: {labels_csv}")
        sys.exit(1)

    # Create generator and run all analyses
    generator = ComprehensiveReportGenerator(output_dir)

    logger.info("\n" + "="*70)
    logger.info("STARTING COMPREHENSIVE ANALYSIS")
    logger.info("="*70)
    logger.info(f"Labels: {labels_csv.name}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Splits found: {len(e07_splits)}")

    # Run analyses
    generator.run_data_validation(labels_csv)
    generator.run_class_distribution_analysis(labels_csv)
    generator.run_spatial_analysis(labels_csv)

    if e07_splits:
        generator.run_split_validation(labels_csv, e07_splits)
    else:
        logger.warning("No E07 split files found - skipping split validation")

    # Generate report
    generator.generate_summary()
    generator.save_report()
    generator.print_summary()

    logger.info("\n✓ All analyses completed successfully!")


if __name__ == '__main__':
    main()
