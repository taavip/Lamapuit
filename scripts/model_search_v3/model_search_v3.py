#!/usr/bin/env python3
"""Model Search V3 launcher.

V3 profile goals:
- Use only high-performing backbones from previous runs.
- Add new candidate backbones requested for evaluation.
- Enforce spatial fences to reduce train/test and cross-year location leakage.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

DEFAULT_FORCE_MODELS = [
    "convnext_small",
    "convnextv2_small",
    "deep_cnn_attn_dropout_tuned",
    "efficientnet_b2",
    "maxvit_small",
    "eva02_small",
]


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="Launch Model Search V3 (compact high-performance + new backbones profile)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--output", default=str(root / "output/model_search_v3"))
    parser.add_argument(
        "--drop-labels",
        default=str(root / "output/onboarding_labels_v3_drop13_diverse_top3"),
        help="Drop13 label directory used by model_search_v2 curation",
    )
    parser.add_argument(
        "--force-models",
        default=",".join(DEFAULT_FORCE_MODELS),
        help="Comma-separated Stage-1 model pool",
    )
    parser.add_argument("--n-models", type=int, default=len(DEFAULT_FORCE_MODELS))

    parser.add_argument("--stage2-strategies", default="mixup_swa,tta")
    parser.add_argument("--max-extended", type=int, default=10)

    parser.add_argument(
        "--spatial-fence-m",
        type=float,
        default=26.0,
        help="Train/test spatial fence buffer in meters (26m ~= one 128px tile at 20cm)",
    )
    parser.add_argument("--cv-group-block-size", type=int, default=128)
    parser.add_argument("--cv-spatial-block-m", type=float, default=0.0)
    parser.add_argument(
        "--auto-cv-block-size",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Auto-probe candidate CV block sizes in meters",
    )
    parser.add_argument("--cv-block-candidates-m", default="26,39,52,78,104")

    parser.add_argument(
        "--split-mode",
        choices=["legacy", "spatial_blocks"],
        default="spatial_blocks",
        help="Use spatial_blocks to keep same-place different years in one split side",
    )
    parser.add_argument("--test-fraction", type=float, default=0.20)
    parser.add_argument("--split-block-size-places", type=int, default=2)

    parser.add_argument("--extra-test-fraction", type=float, default=0.10)
    parser.add_argument("--max-extra-test", type=int, default=500)
    parser.add_argument("--t-high", type=float, default=0.98)
    parser.add_argument("--t-low", type=float, default=0.05)

    parser.add_argument("--augment-random-nodata-frac", type=float, default=0.50)
    parser.add_argument("--augment-pattern-nodata-frac", type=float, default=0.75)

    parser.add_argument("--stage2-epochs", type=int, default=60)
    parser.add_argument("--stage2-patience", type=int, default=10)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=2026)

    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")

    args, passthrough = parser.parse_known_args()
    return args, passthrough


def main() -> int:
    args, passthrough = parse_args()
    root = Path(__file__).resolve().parents[2]
    v2_script = root / "scripts" / "model_search_v2" / "model_search_v2.py"

    cmd = [
        sys.executable,
        str(v2_script),
        "--output",
        str(args.output),
        "--n-models",
        str(args.n_models),
        "--force-models",
        args.force_models,
        "--stage2-strategies",
        args.stage2_strategies,
        "--max-extended",
        str(args.max_extended),
        "--drop-labels",
        str(args.drop_labels),
        "--split-mode",
        str(args.split_mode),
        "--test-fraction",
        str(args.test_fraction),
        "--split-block-size-places",
        str(args.split_block_size_places),
        "--spatial-fence-m",
        str(args.spatial_fence_m),
        "--cv-group-block-size",
        str(args.cv_group_block_size),
        "--cv-spatial-block-m",
        str(args.cv_spatial_block_m),
        "--cv-block-candidates-m",
        str(args.cv_block_candidates_m),
        "--extra-test-fraction",
        str(args.extra_test_fraction),
        "--max-extra-test",
        str(args.max_extra_test),
        "--t-high",
        str(args.t_high),
        "--t-low",
        str(args.t_low),
        "--augment-random-nodata-frac",
        str(args.augment_random_nodata_frac),
        "--augment-pattern-nodata-frac",
        str(args.augment_pattern_nodata_frac),
        "--stage2-epochs",
        str(args.stage2_epochs),
        "--stage2-patience",
        str(args.stage2_patience),
        "--n-folds",
        str(args.n_folds),
        "--batch-size",
        str(args.batch_size),
        "--seed",
        str(args.seed),
    ]

    if args.prepare_only:
        cmd.append("--prepare-only")
    if args.auto_cv_block_size:
        cmd.append("--auto-cv-block-size")
    if args.smoke_test:
        cmd.append("--smoke-test")

    cmd.extend(passthrough)

    print("Running:")
    print(" ".join(cmd))
    proc = subprocess.run(cmd, check=False)
    return int(proc.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
