#!/usr/bin/env python3
"""
Comprehensive CHM variant generator.

Generates all CHM variants from LAZ input with clear naming:
  1. baseline_chm_0p2m/          (20cm CHM from original sparse LiDAR)
  2. harmonized_raw_0p2m/        (Harmonized DEM + raw CHM, no smoothing)
  3. harmonized_gauss_kernel0p8m_0p2m/  (Gaussian kernel 0.8m, resolution 0.2m)
  4. composite_4band_raw_base/   (Gauss + Raw + Baseline + Mask, mask=Raw+Base)
  5. masked_raw_2band/           (Raw + Mask channel)

Usage:
  python scripts/generate_all_chm_variants.py \\
    --laz-dir /path/to/laz/folder \\
    --output-dir /path/to/output \\
    --variants baseline,raw,gaussian,composite,masked-raw

Options:
  --laz-dir           Path to LAZ files (required)
  --output-dir        Output base directory (required)
  --variants          Comma-separated list: baseline,raw,gaussian,composite,masked-raw
                      (default: all)
  --resolution        Output CHM resolution in meters (default: 0.2)
  --gaussian-kernel   Gaussian smoothing kernel size (default: 0.8)
  --harmonize-dem     Use harmonized DEM for raw CHM (default: True)
  --max-tiles         Max tiles to process (0=all, default: 0)
  --skip-existing     Skip tiles that already exist (default: True)
  --verbose           Verbose output (default: False)
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Optional


class CHMVariantGenerator:
    """Generate all CHM variants from LAZ data."""

    VARIANTS = {
        'baseline': 'Baseline CHM (0.2m from original sparse LiDAR)',
        'raw': 'Harmonized raw CHM (0.2m, no smoothing)',
        'gaussian': 'Harmonized Gaussian CHM (0.8m kernel, 0.2m resolution)',
        'composite': '4-band composite (Gauss+Raw+Base+Mask)',
        'masked-raw': '2-band masked raw CHM (Raw+Mask)',
    }

    def __init__(
        self,
        laz_dir: str,
        output_dir: str,
        resolution: float = 0.2,
        gaussian_kernel: float = 0.8,
        harmonize_dem: bool = True,
        max_tiles: int = 0,
        skip_existing: bool = True,
        verbose: bool = False,
    ):
        self.laz_dir = Path(laz_dir)
        self.output_dir = Path(output_dir)
        self.resolution = resolution
        self.gaussian_kernel = gaussian_kernel
        self.harmonize_dem = harmonize_dem
        self.max_tiles = max_tiles
        self.skip_existing = skip_existing
        self.verbose = verbose

        # Validate input
        if not self.laz_dir.exists():
            raise FileNotFoundError(f"LAZ directory not found: {self.laz_dir}")

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def print_info(self, msg: str):
        """Print info message."""
        if self.verbose:
            print(f"[INFO] {msg}")

    def print_status(self, msg: str):
        """Print status message."""
        print(f"[STATUS] {msg}")

    def print_error(self, msg: str):
        """Print error message."""
        print(f"[ERROR] {msg}", file=sys.stderr)

    def run_command(self, cmd: List[str], description: str) -> bool:
        """Run a shell command and return success status."""
        self.print_status(f"{description}...")
        if self.verbose:
            self.print_info(f"Command: {' '.join(cmd)}")

        try:
            result = subprocess.run(
                cmd,
                capture_output=not self.verbose,
                text=True,
                check=False,
            )
            if result.returncode != 0:
                self.print_error(f"{description} failed")
                if result.stderr:
                    self.print_error(result.stderr)
                return False
            return True
        except Exception as e:
            self.print_error(f"{description} error: {e}")
            return False

    def generate_baseline(self) -> bool:
        """Generate baseline CHM (from original sparse LiDAR)."""
        output_subdir = self.output_dir / "baseline_chm_0p2m"

        if output_subdir.exists() and self.skip_existing:
            self.print_status(f"Baseline CHM already exists: {output_subdir}")
            return True

        output_subdir.mkdir(parents=True, exist_ok=True)

        # This would typically call the original CHM generation pipeline
        # For now, provide instructions
        cmd = [
            "python",
            "scripts/process_laz_to_chm.py",
            "--input",
            str(self.laz_dir),
            "--output",
            str(output_subdir),
            "--resolution",
            str(self.resolution),
            "--mode",
            "drop",  # or "clip"
        ]

        if self.max_tiles > 0:
            cmd.extend(["--max-tiles", str(self.max_tiles)])

        return self.run_command(cmd, "Generating baseline CHM")

    def generate_harmonized_raw(self) -> bool:
        """Generate harmonized raw CHM (no smoothing)."""
        output_subdir = self.output_dir / "harmonized_raw_0p2m"

        if output_subdir.exists() and self.skip_existing:
            self.print_status(f"Harmonized raw CHM already exists: {output_subdir}")
            return True

        output_subdir.mkdir(parents=True, exist_ok=True)

        # Call harmonized CHM pipeline
        cmd = [
            "python",
            "experiments/laz_to_chm_harmonized_0p8m/build_dataset.py",
            "--laz-dir",
            str(self.laz_dir),
            "--output-dir",
            str(output_subdir),
            "--resolution",
            str(self.resolution),
            "--no-gaussian",  # Raw only
        ]

        if self.max_tiles > 0:
            cmd.extend(["--max-tiles", str(self.max_tiles)])

        return self.run_command(cmd, "Generating harmonized raw CHM")

    def generate_harmonized_gaussian(self) -> bool:
        """Generate harmonized Gaussian CHM (with smoothing)."""
        output_subdir = (
            self.output_dir / f"harmonized_gauss_kernel{self.gaussian_kernel:.1f}m_res{self.resolution:.1f}m"
        )

        if output_subdir.exists() and self.skip_existing:
            self.print_status(
                f"Harmonized Gaussian CHM already exists: {output_subdir}"
            )
            return True

        output_subdir.mkdir(parents=True, exist_ok=True)

        # Call harmonized CHM pipeline with Gaussian smoothing
        cmd = [
            "python",
            "experiments/laz_to_chm_harmonized_0p8m/build_dataset.py",
            "--laz-dir",
            str(self.laz_dir),
            "--output-dir",
            str(output_subdir),
            "--resolution",
            str(self.resolution),
            "--gaussian-kernel",
            str(self.gaussian_kernel),
        ]

        if self.max_tiles > 0:
            cmd.extend(["--max-tiles", str(self.max_tiles)])

        return self.run_command(cmd, "Generating harmonized Gaussian CHM")

    def generate_composite_4band(self) -> bool:
        """Generate 4-band composite (Gauss+Raw+Base+Mask)."""
        output_subdir = self.output_dir / "composite_4band_raw_base_mask"

        if output_subdir.exists() and self.skip_existing:
            self.print_status(f"4-band composite already exists: {output_subdir}")
            return True

        output_subdir.mkdir(parents=True, exist_ok=True)

        # Requires baseline, raw, and gaussian to exist
        gauss_dir = self.output_dir / f"harmonized_gauss_kernel{self.gaussian_kernel:.1f}m_res{self.resolution:.1f}m"
        raw_dir = self.output_dir / "harmonized_raw_0p2m"
        base_dir = self.output_dir / "baseline_chm_0p2m"

        if not all(d.exists() for d in [gauss_dir, raw_dir, base_dir]):
            self.print_error(
                "4-band composite requires baseline, raw, and gaussian CHMs"
            )
            return False

        cmd = [
            "python",
            "scripts/build_composite_3band_with_masks.py",
            str(self.max_tiles) if self.max_tiles > 0 else "0",
            "--gauss-dir",
            str(gauss_dir),
            "--raw-dir",
            str(raw_dir),
            "--base-dir",
            str(base_dir),
            "--output-dir",
            str(output_subdir),
        ]

        return self.run_command(cmd, "Generating 4-band composite")

    def generate_masked_raw_2band(self) -> bool:
        """Generate 2-band masked raw CHM (Raw+Mask)."""
        output_subdir = self.output_dir / "masked_raw_2band_0p2m"

        if output_subdir.exists() and self.skip_existing:
            self.print_status(f"2-band masked raw CHM already exists: {output_subdir}")
            return True

        output_subdir.mkdir(parents=True, exist_ok=True)

        # Requires raw CHM to exist
        raw_dir = self.output_dir / "harmonized_raw_0p2m"

        if not raw_dir.exists():
            self.print_error("2-band masked raw requires harmonized raw CHM")
            return False

        cmd = [
            "python",
            "scripts/build_2band_masked_chm.py",
            str(self.max_tiles) if self.max_tiles > 0 else "0",
            "--input-dir",
            str(raw_dir),
            "--output-dir",
            str(output_subdir),
        ]

        return self.run_command(cmd, "Generating 2-band masked raw CHM")

    def generate(self, variants: Optional[List[str]] = None) -> dict:
        """
        Generate selected CHM variants.

        Args:
            variants: List of variant names ('baseline', 'raw', 'gaussian', 'composite', 'masked-raw')
                     If None, generate all.

        Returns:
            Dict with generation results.
        """
        if variants is None:
            variants = list(self.VARIANTS.keys())

        # Validate variant names
        invalid = [v for v in variants if v not in self.VARIANTS]
        if invalid:
            raise ValueError(f"Invalid variants: {invalid}")

        self.print_status("=" * 70)
        self.print_status("CHM Variant Generator")
        self.print_status("=" * 70)
        self.print_status(f"LAZ input: {self.laz_dir}")
        self.print_status(f"Output base: {self.output_dir}")
        self.print_status(f"Resolution: {self.resolution:.1f} m")
        self.print_status(f"Gaussian kernel: {self.gaussian_kernel:.1f} m")
        self.print_status(f"Variants to generate: {', '.join(variants)}")
        self.print_status("=" * 70)
        self.print_status("")

        results = {}

        # Generate in dependency order
        if "baseline" in variants:
            results["baseline"] = self.generate_baseline()

        if "raw" in variants:
            results["raw"] = self.generate_harmonized_raw()

        if "gaussian" in variants:
            results["gaussian"] = self.generate_harmonized_gaussian()

        if "composite" in variants:
            # Depends on baseline, raw, gaussian
            if not all(results.get(v, False) for v in ["baseline", "raw", "gaussian"]):
                self.print_error(
                    "Composite requires baseline, raw, and gaussian. Skipping."
                )
                results["composite"] = False
            else:
                results["composite"] = self.generate_composite_4band()

        if "masked-raw" in variants:
            # Depends on raw
            if not results.get("raw", False):
                self.print_error("Masked raw requires raw CHM. Skipping.")
                results["masked-raw"] = False
            else:
                results["masked-raw"] = self.generate_masked_raw_2band()

        return results

    def print_summary(self, results: dict):
        """Print generation summary."""
        self.print_status("=" * 70)
        self.print_status("Generation Summary")
        self.print_status("=" * 70)

        for variant, success in results.items():
            status = "✓ SUCCESS" if success else "✗ FAILED"
            desc = self.VARIANTS.get(variant, variant)
            self.print_status(f"{status:12} {variant:15} {desc}")

        self.print_status("=" * 70)
        self.print_status("")
        self.print_status(f"Output directory: {self.output_dir}")
        self.print_status("")

        # List generated directories
        self.print_status("Generated directories:")
        for subdir in sorted(self.output_dir.iterdir()):
            if subdir.is_dir():
                file_count = len(list(subdir.glob("*.tif")))
                self.print_status(f"  {subdir.name:45} ({file_count} .tif files)")


def main():
    parser = argparse.ArgumentParser(
        description="Generate all CHM variants from LAZ data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate all variants
  python scripts/generate_all_chm_variants.py \\
    --laz-dir /data/laz \\
    --output-dir /data/chm_variants

  # Generate only baseline and raw
  python scripts/generate_all_chm_variants.py \\
    --laz-dir /data/laz \\
    --output-dir /data/chm_variants \\
    --variants baseline,raw

  # Generate with custom resolution and Gaussian kernel
  python scripts/generate_all_chm_variants.py \\
    --laz-dir /data/laz \\
    --output-dir /data/chm_variants \\
    --resolution 0.5 \\
    --gaussian-kernel 1.0 \\
    --max-tiles 10

Output directory structure:
  /output-dir/
  ├── baseline_chm_0p2m/
  ├── harmonized_raw_0p2m/
  ├── harmonized_gauss_kernel0p8m_0p2m/
  ├── composite_4band_raw_base_mask/
  └── masked_raw_2band_0p2m/
        """,
    )

    parser.add_argument("--laz-dir", required=True, help="Input LAZ directory")
    parser.add_argument("--output-dir", required=True, help="Output base directory")
    parser.add_argument(
        "--variants",
        default="baseline,raw,gaussian,composite,masked-raw",
        help="Comma-separated variants (default: all)",
    )
    parser.add_argument(
        "--resolution", type=float, default=0.2, help="CHM resolution in meters"
    )
    parser.add_argument(
        "--gaussian-kernel",
        type=float,
        default=0.8,
        help="Gaussian smoothing kernel size",
    )
    parser.add_argument(
        "--no-harmonize-dem",
        action="store_true",
        help="Don't use harmonized DEM",
    )
    parser.add_argument(
        "--max-tiles", type=int, default=0, help="Max tiles to process (0=all)"
    )
    parser.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="Don't skip existing outputs",
    )
    parser.add_argument("--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args()

    # Parse variants
    variants = [v.strip() for v in args.variants.split(",")]

    try:
        generator = CHMVariantGenerator(
            laz_dir=args.laz_dir,
            output_dir=args.output_dir,
            resolution=args.resolution,
            gaussian_kernel=args.gaussian_kernel,
            harmonize_dem=not args.no_harmonize_dem,
            max_tiles=args.max_tiles,
            skip_existing=not args.no_skip_existing,
            verbose=args.verbose,
        )

        results = generator.generate(variants)
        generator.print_summary(results)

        # Exit with error if any failed
        if not all(results.values()):
            sys.exit(1)

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
