"""
CHM Variant Generator Orchestrator

Main interface for generating all CHM variants from LAZ data.

Generates 5 variants with clear naming:
  1. baseline_chm_0p2m/                      (Original sparse LiDAR)
  2. harmonized_raw_0p2m/                    (No smoothing)
  3. harmonized_gauss_kernel0p8m_0p2m/       (Gaussian kernel 0.8m, resolution 0.2m)
  4. composite_4band_raw_base_mask/          (Gauss + Raw + Base + Mask)
  5. masked_raw_2band_0p2m/                  (Raw + Mask)

Usage:
    from src.cdw_detect.chm_variants import CHMVariantGenerator

    generator = CHMVariantGenerator(
        laz_dir="/data/laz",
        output_dir="/data/chm_variants"
    )
    results = generator.generate(variants=['baseline', 'raw', 'gaussian', 'composite', 'masked-raw'])
"""

import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Dict


class CHMVariantGenerator:
    """Orchestrate generation of all CHM variants."""

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
        max_tiles: int = 0,
        skip_existing: bool = True,
        verbose: bool = False,
    ):
        """
        Initialize CHM variant generator.

        Args:
            laz_dir: Input LAZ directory
            output_dir: Output base directory
            resolution: CHM resolution in meters (default: 0.2)
            gaussian_kernel: Gaussian smoothing kernel size (default: 0.8)
            max_tiles: Max tiles to process (0=all)
            skip_existing: Skip existing outputs
            verbose: Verbose output
        """
        self.laz_dir = Path(laz_dir)
        self.output_dir = Path(output_dir)
        self.resolution = resolution
        self.gaussian_kernel = gaussian_kernel
        self.max_tiles = max_tiles
        self.skip_existing = skip_existing
        self.verbose = verbose

        if not self.laz_dir.exists():
            raise FileNotFoundError(f"LAZ directory not found: {self.laz_dir}")

        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _print(self, level: str, msg: str):
        """Print status/info/error messages."""
        if level == "status":
            print(f"[STATUS] {msg}")
        elif level == "info" and self.verbose:
            print(f"[INFO] {msg}")
        elif level == "error":
            print(f"[ERROR] {msg}", file=sys.stderr)

    def _run_command(self, cmd: List[str], description: str) -> bool:
        """Run shell command, return success."""
        self._print("status", f"{description}...")
        try:
            result = subprocess.run(
                cmd,
                capture_output=not self.verbose,
                text=True,
                check=False,
            )
            if result.returncode != 0:
                self._print("error", f"{description} failed")
                return False
            return True
        except Exception as e:
            self._print("error", f"{description}: {e}")
            return False

    def generate(self, variants: Optional[List[str]] = None) -> Dict[str, bool]:
        """
        Generate selected CHM variants.

        Args:
            variants: List of variant names. If None, generates all.

        Returns:
            Dict with generation results {variant: success}.
        """
        if variants is None:
            variants = list(self.VARIANTS.keys())

        # Validate
        invalid = [v for v in variants if v not in self.VARIANTS]
        if invalid:
            raise ValueError(f"Invalid variants: {invalid}")

        self._print("status", "=" * 70)
        self._print("status", "CHM Variant Generator")
        self._print("status", "=" * 70)
        self._print("status", f"LAZ input: {self.laz_dir}")
        self._print("status", f"Output base: {self.output_dir}")
        self._print("status", f"Resolution: {self.resolution:.1f} m")
        self._print("status", f"Gaussian kernel: {self.gaussian_kernel:.1f} m")
        self._print("status", f"Variants: {', '.join(variants)}")
        self._print("status", "=" * 70)

        results = {}

        # Generate in dependency order
        if "baseline" in variants:
            self._print("status", "")
            results["baseline"] = self._generate_baseline()

        if "raw" in variants:
            self._print("status", "")
            results["raw"] = self._generate_harmonized_raw()

        if "gaussian" in variants:
            self._print("status", "")
            results["gaussian"] = self._generate_harmonized_gaussian()

        if "composite" in variants:
            self._print("status", "")
            if not all(results.get(v, False) for v in ["baseline", "raw", "gaussian"]):
                self._print("error", "Composite requires baseline, raw, gaussian. Skipping.")
                results["composite"] = False
            else:
                results["composite"] = self._generate_composite_4band()

        if "masked-raw" in variants:
            self._print("status", "")
            if not results.get("raw", False):
                self._print("error", "Masked raw requires raw CHM. Skipping.")
                results["masked-raw"] = False
            else:
                results["masked-raw"] = self._generate_masked_raw_2band()

        self._print_summary(results)
        return results

    def _generate_baseline(self) -> bool:
        """Generate baseline CHM from original sparse LiDAR."""
        output_subdir = self.output_dir / "baseline_chm_0p2m"

        if output_subdir.exists() and self.skip_existing:
            self._print("status", f"Baseline CHM already exists: {output_subdir}")
            return True

        output_subdir.mkdir(parents=True, exist_ok=True)

        # Call baseline CHM generation
        cmd = [
            "python",
            "scripts/process_laz_to_chm.py",
            "--input",
            str(self.laz_dir),
            "--output",
            str(output_subdir),
            "--resolution",
            str(self.resolution),
        ]

        if self.max_tiles > 0:
            cmd.extend(["--max-tiles", str(self.max_tiles)])

        return self._run_command(cmd, "Generating baseline CHM")

    def _generate_harmonized_raw(self) -> bool:
        """Generate harmonized raw CHM (no smoothing)."""
        output_subdir = self.output_dir / "harmonized_raw_0p2m"

        if output_subdir.exists() and self.skip_existing:
            self._print("status", f"Harmonized raw CHM already exists: {output_subdir}")
            return True

        output_subdir.mkdir(parents=True, exist_ok=True)

        cmd = [
            "python",
            "experiments/laz_to_chm_harmonized_0p8m/build_dataset.py",
            "--laz-dir",
            str(self.laz_dir),
            "--output-dir",
            str(output_subdir),
            "--resolution",
            str(self.resolution),
        ]

        if self.max_tiles > 0:
            cmd.extend(["--max-tiles", str(self.max_tiles)])

        return self._run_command(cmd, "Generating harmonized raw CHM")

    def _generate_harmonized_gaussian(self) -> bool:
        """Generate harmonized Gaussian CHM (with smoothing)."""
        output_subdir = (
            self.output_dir / f"harmonized_gauss_kernel{self.gaussian_kernel:.1f}m_res{self.resolution:.1f}m"
        )

        if output_subdir.exists() and self.skip_existing:
            self._print("status", f"Harmonized Gaussian CHM already exists: {output_subdir}")
            return True

        output_subdir.mkdir(parents=True, exist_ok=True)

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

        return self._run_command(cmd, "Generating harmonized Gaussian CHM")

    def _generate_composite_4band(self) -> bool:
        """Generate 4-band composite (Gauss+Raw+Base+Mask)."""
        from .composite import CompositeGenerator

        output_subdir = self.output_dir / "composite_4band_raw_base_mask"

        if output_subdir.exists() and self.skip_existing:
            self._print("status", f"4-band composite already exists: {output_subdir}")
            return True

        gauss_dir = self.output_dir / f"harmonized_gauss_kernel{self.gaussian_kernel:.1f}m_res{self.resolution:.1f}m"
        raw_dir = self.output_dir / "harmonized_raw_0p2m"
        base_dir = self.output_dir / "baseline_chm_0p2m"

        if not all(d.exists() for d in [gauss_dir, raw_dir, base_dir]):
            self._print("error", "4-band composite requires baseline, raw, gaussian CHMs")
            return False

        try:
            gen = CompositeGenerator(str(gauss_dir), str(raw_dir), str(base_dir), str(output_subdir))
            processed = gen.generate(max_tiles=self.max_tiles)
            return processed > 0
        except Exception as e:
            self._print("error", f"Composite generation failed: {e}")
            return False

    def _generate_masked_raw_2band(self) -> bool:
        """Generate 2-band masked raw CHM (Raw+Mask)."""
        from .masked import MaskedCHMGenerator

        output_subdir = self.output_dir / "masked_raw_2band_0p2m"

        if output_subdir.exists() and self.skip_existing:
            self._print("status", f"2-band masked raw CHM already exists: {output_subdir}")
            return True

        raw_dir = self.output_dir / "harmonized_raw_0p2m"

        if not raw_dir.exists():
            self._print("error", "2-band masked raw requires harmonized raw CHM")
            return False

        try:
            gen = MaskedCHMGenerator(str(raw_dir), str(output_subdir))
            processed = gen.generate(max_tiles=self.max_tiles)
            return processed > 0
        except Exception as e:
            self._print("error", f"2-band masked generation failed: {e}")
            return False

    def _print_summary(self, results: Dict[str, bool]):
        """Print generation summary."""
        self._print("status", "=" * 70)
        self._print("status", "Generation Summary")
        self._print("status", "=" * 70)

        for variant, success in results.items():
            status = "✓ SUCCESS" if success else "✗ FAILED"
            desc = self.VARIANTS.get(variant, variant)
            self._print("status", f"{status:12} {variant:15} {desc}")

        self._print("status", "=" * 70)
        self._print("status", f"Output directory: {self.output_dir}")
        self._print("status", "")

        # List generated directories
        self._print("status", "Generated directories:")
        for subdir in sorted(self.output_dir.iterdir()):
            if subdir.is_dir():
                file_count = len(list(subdir.glob("*.tif")))
                self._print("status", f"  {subdir.name:45} ({file_count} .tif files)")
