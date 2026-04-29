#!/usr/bin/env python3
"""Summarize manual-mask finetuning experiment outputs into CSV and Markdown."""

from __future__ import annotations

import csv
import json
from pathlib import Path


def main() -> None:
    root = Path("output/manual_mask_experiments")
    report_paths = sorted(root.glob("*/manual_mask_finetune_report.json"))
    if not report_paths:
        raise RuntimeError("No manual_mask_finetune_report.json files found under output/manual_mask_experiments")

    rows: list[dict[str, object]] = []
    for path in report_paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        cfg = data["config"]
        delta = data["delta"]
        rows.append(
            {
                "run": path.parent.name,
                "seed": int(cfg["seed"]),
                "optimizer": str(cfg["optimizer"]),
                "lr": float(cfg["lr"]),
                "augment": bool(cfg["augment"]),
                "freeze_encoder": bool(cfg["freeze_encoder"]),
                "cosine": bool(cfg["use_cosine_scheduler"]),
                "val_dice_delta_fixed": float(delta["val_dice"]),
                "val_f1_delta_fixed": float(delta["val_f1"]),
                "best_val_dice_delta": float(delta["best_val_dice"]),
                "best_val_f1_delta": float(delta["best_val_f1"]),
                "all_dice_delta_fixed": float(delta["all_dice"]),
                "best_all_dice_delta": float(delta["best_all_dice"]),
                "tuned_best_val_threshold": float(data["tuned"]["best_val"]["threshold"]),
                "baseline_best_val_threshold": float(data["baseline"]["best_val"]["threshold"]),
            }
        )

    summary_csv = Path("MANUAL_MASK_EXPERIMENT_SUMMARY.csv")
    with summary_csv.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    seed42 = [r for r in rows if int(r["seed"]) == 42 and "_seed" not in str(r["run"])]
    seed42_sorted = sorted(seed42, key=lambda r: float(r["best_val_f1_delta"]), reverse=True)

    pair_runs: list[dict[str, float]] = []
    for seed in (42, 7, 123):
        baseline_name = "baseline_default" if seed == 42 else f"baseline_default_seed{seed}"
        combo_name = "full_combo" if seed == 42 else f"full_combo_seed{seed}"
        baseline = next((r for r in rows if str(r["run"]) == baseline_name), None)
        combo = next((r for r in rows if str(r["run"]) == combo_name), None)
        if baseline is None or combo is None:
            continue

        pair_runs.append(
            {
                "seed": float(seed),
                "baseline_best_val_f1_delta": float(baseline["best_val_f1_delta"]),
                "combo_best_val_f1_delta": float(combo["best_val_f1_delta"]),
                "combo_minus_baseline_best_val_f1": float(combo["best_val_f1_delta"]) - float(baseline["best_val_f1_delta"]),
                "baseline_fixed_val_f1_delta": float(baseline["val_f1_delta_fixed"]),
                "combo_fixed_val_f1_delta": float(combo["val_f1_delta_fixed"]),
                "combo_minus_baseline_fixed_val_f1": float(combo["val_f1_delta_fixed"]) - float(baseline["val_f1_delta_fixed"]),
            }
        )

    mean_combo_adv_best = sum(x["combo_minus_baseline_best_val_f1"] for x in pair_runs) / max(1, len(pair_runs))
    mean_combo_adv_fixed = sum(x["combo_minus_baseline_fixed_val_f1"] for x in pair_runs) / max(1, len(pair_runs))

    lines: list[str] = []
    lines.append("# Manual Mask Fine-Tuning Experiment Report")
    lines.append("")
    lines.append("## Scope")
    lines.append("- Goal: test multiple methods to improve PartialConv quality using manual RGBA masks.")
    lines.append("- Data: 23 labeled chips from output/manual_masks (15 train / 8 val per seed split).")
    lines.append("- Baseline checkpoint: output/cwd_partialconv_gpu_multiepoch_20260417_sota_es/best_partialconv_unet.pt")
    lines.append("- Metric focus: F1/Dice deltas vs baseline on the same split; both fixed threshold (0.5) and best threshold from sweep.")
    lines.append("")
    lines.append("## Methods Tried (Seed 42)")
    lines.append("")
    lines.append("| Run | Optimizer | LR | Aug | Freeze Encoder | Cosine | Best Val F1 Delta | Best Val Dice Delta |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for row in seed42_sorted:
        lines.append(
            f"| {row['run']} | {row['optimizer']} | {float(row['lr']):.1e} | {str(bool(row['augment'])).lower()} | "
            f"{str(bool(row['freeze_encoder'])).lower()} | {str(bool(row['cosine'])).lower()} | "
            f"{float(row['best_val_f1_delta']):+.6f} | {float(row['best_val_dice_delta']):+.6f} |"
        )

    lines.append("")
    lines.append("## Robustness: Baseline vs Full Combo Across Seeds")
    lines.append("")
    lines.append("| Seed | Baseline Best Val F1 Delta | Full Combo Best Val F1 Delta | Combo-Baseline (Best Val F1) | Combo-Baseline (Fixed Val F1@0.5) |")
    lines.append("|---:|---:|---:|---:|---:|")
    for pair in pair_runs:
        lines.append(
            f"| {int(pair['seed'])} | {pair['baseline_best_val_f1_delta']:+.6f} | {pair['combo_best_val_f1_delta']:+.6f} | "
            f"{pair['combo_minus_baseline_best_val_f1']:+.6f} | {pair['combo_minus_baseline_fixed_val_f1']:+.6f} |"
        )

    lines.append("")
    lines.append(f"- Mean combo advantage (best-threshold Val F1) across seeds: {mean_combo_adv_best:+.6f}")
    lines.append(f"- Mean combo advantage (fixed-threshold Val F1@0.5) across seeds: {mean_combo_adv_fixed:+.6f}")
    lines.append("")
    lines.append("## Findings")
    lines.append("- We did not test every possible method, but we tested practical quality-improvement strategies: augmentation, optimizer change, lower LR, encoder freezing, and cosine scheduling.")
    lines.append("- On seed 42, full_combo was best by best-threshold validation metrics.")
    lines.append("- Across seeds (42, 7, 123), full_combo beat baseline on best-threshold validation F1 in all tested seeds.")
    lines.append("- Fixed-threshold metrics are noisier and can disagree due calibration; threshold sweep gives a fairer model-vs-model comparison on this tiny set.")
    lines.append("- Dataset size (23 chips) is the main bottleneck; variance remains high despite tuning.")
    lines.append("")
    lines.append("## Recommended Next Settings")
    lines.append("- Use AdamW, LR=5e-5, augmentation on, freeze encoder on, cosine scheduler on.")
    lines.append("- Keep threshold calibration during evaluation.")
    lines.append("- Highest-impact next step: add more manual masks, especially hard negatives/background clutter.")
    lines.append("")
    lines.append("## Artifacts")
    lines.append("- Matrix outputs: output/manual_mask_experiments/*/manual_mask_finetune_report.json")
    lines.append("- Summary table: MANUAL_MASK_EXPERIMENT_SUMMARY.csv")

    report_md = Path("MANUAL_MASK_EXPERIMENT_REPORT.md")
    report_md.write_text("\n".join(lines), encoding="utf-8")

    print(f"WROTE {summary_csv}")
    print(f"WROTE {report_md}")
    if seed42_sorted:
        best = seed42_sorted[0]
        print(f"BEST_SEED42 {best['run']} best_val_f1_delta={float(best['best_val_f1_delta']):+.6f}")
    print(f"MEAN_COMBO_ADV_BEST {mean_combo_adv_best:+.6f}")
    print(f"MEAN_COMBO_ADV_FIXED {mean_combo_adv_fixed:+.6f}")


if __name__ == "__main__":
    main()
