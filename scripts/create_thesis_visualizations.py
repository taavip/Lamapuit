#!/usr/bin/env python3
"""
Create publication-ready visualizations for Option B thesis results.

Generates:
1. Top 10 probability change examples (CHM tiles + before/after)
2. Ensemble model architecture diagram
3. Probability distribution comparisons
4. Test metrics visualization
5. Class-wise performance analysis
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
import seaborn as sns
from scipy import stats

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9


def load_chm_window(chm_dir: Path, raster_name: str, row_off: int, col_off: int) -> np.ndarray | None:
    """Load 128×128 CHM window from GeoTIFF."""
    chm_path = chm_dir / raster_name
    if not chm_path.exists():
        return None
    try:
        with rasterio.open(chm_path) as src:
            window = Window(col_off, row_off, 128, 128)
            data = src.read(1, window=window).astype(np.float32)
            if src.nodata is not None:
                data[data == src.nodata] = np.nan
            data = np.nan_to_num(data, nan=0.0)
            return data
    except Exception as e:
        print(f"Error loading {raster_name}: {e}")
        return None


def create_top10_changes_visualization(output_dir: Path):
    """Create visualization of top 10 probability changes."""
    print("\n[Creating top 10 changes visualization...]")

    # Load data
    df_orig = pd.read_csv("data/chm_variants/labels_canonical_with_splits.csv")
    df_retrain = pd.read_csv("data/chm_variants/labels_canonical_with_splits_retrained_ensemble.csv")

    # Merge
    df_merged = pd.merge(
        df_orig[['raster', 'row_off', 'col_off', 'label', 'split', 'model_prob']].rename(
            columns={'model_prob': 'original_prob'}
        ),
        df_retrain[['raster', 'row_off', 'col_off', 'model_prob']].rename(
            columns={'model_prob': 'retrained_prob'}
        ),
        on=['raster', 'row_off', 'col_off'],
    )

    # Calculate differences
    df_merged['abs_diff'] = np.abs(df_merged['original_prob'] - df_merged['retrained_prob'])

    # Get top 10
    top10 = df_merged.nlargest(10, 'abs_diff')

    # Create figure (5 rows × 2 cols)
    fig = plt.figure(figsize=(14, 18))
    gs = GridSpec(10, 3, figure=fig, hspace=0.4, wspace=0.3)

    chm_dir = Path("data/lamapuit/chm_max_hag_13_drop")

    for idx, (_, row) in enumerate(top10.iterrows()):
        # Load CHM tile
        chm = load_chm_window(chm_dir, row['raster'], int(row['row_off']), int(row['col_off']))
        if chm is None:
            continue

        # Subplot: CHM heatmap
        ax_chm = fig.add_subplot(gs[idx, 0])
        im = ax_chm.imshow(chm, cmap='viridis', aspect='auto')
        ax_chm.set_title(f"#{idx+1}: {row['label'].upper()}", fontsize=10, fontweight='bold')
        ax_chm.set_xlabel("X (pixels)")
        ax_chm.set_ylabel("Y (pixels)")
        ax_chm.grid(False)
        plt.colorbar(im, ax=ax_chm, label="Height (m)")

        # Subplot: Before/After probabilities
        ax_bar = fig.add_subplot(gs[idx, 1:])
        x_pos = [0, 1]
        probs = [row['original_prob'], row['retrained_prob']]
        colors = ['#1f77b4', '#ff7f0e']
        bars = ax_bar.bar(x_pos, probs, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)

        # Add value labels on bars
        for bar, prob in zip(bars, probs):
            height = bar.get_height()
            ax_bar.text(bar.get_x() + bar.get_width()/2., height,
                       f'{prob:.4f}',
                       ha='center', va='bottom', fontsize=9, fontweight='bold')

        ax_bar.set_ylabel("P(CDW)")
        ax_bar.set_ylim([0, 1.0])
        ax_bar.set_xticks(x_pos)
        ax_bar.set_xticklabels(['Original', 'Retrained'], fontsize=9)
        ax_bar.grid(axis='y', alpha=0.3)

        # Add change annotation
        change_pct = row['abs_diff'] * 100
        ax_bar.text(0.5, 0.95, f'Δ = {change_pct:.2f}%',
                   transform=ax_bar.transAxes,
                   ha='center', va='top',
                   bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5),
                   fontsize=9, fontweight='bold')

    plt.suptitle('Top 10 Largest Probability Changes (Option A → Option B)',
                 fontsize=14, fontweight='bold', y=0.995)

    output_path = output_dir / "top10_probability_changes.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_path}")
    plt.close()


def create_ensemble_architecture_diagram(output_dir: Path):
    """Create ensemble model architecture diagram."""
    print("\n[Creating ensemble architecture diagram...]")

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Title
    ax.text(5, 9.5, 'Option B: 4-Model Ensemble Architecture',
           ha='center', fontsize=14, fontweight='bold')

    # Input
    input_box = mpatches.FancyBboxPatch((3.5, 8), 3, 0.6,
                                        boxstyle="round,pad=0.1",
                                        edgecolor='black', facecolor='lightblue', linewidth=2)
    ax.add_patch(input_box)
    ax.text(5, 8.3, 'CHM Input (128×128×1)', ha='center', va='center', fontsize=10, fontweight='bold')

    # TTA Augmentation
    ax.text(5, 7.5, '8× Test-Time Augmentation', ha='center', fontsize=10, style='italic')
    ax.text(5, 7.1, '(4 rotations × 2 flips)', ha='center', fontsize=9)

    # Models
    models = [
        ('CNN-seed42\n(50 epochs)', 0.5, 5.5),
        ('CNN-seed43\n(50 epochs)', 2.5, 5.5),
        ('CNN-seed44\n(50 epochs)', 4.5, 5.5),
        ('EfficientNet-B2\n(30 epochs)', 6.5, 5.5),
    ]

    for model_name, x, y in models:
        box = mpatches.FancyBboxPatch((x-0.7, y-0.5), 1.4, 1,
                                      boxstyle="round,pad=0.05",
                                      edgecolor='darkgreen', facecolor='lightgreen', linewidth=2)
        ax.add_patch(box)
        ax.text(x, y, model_name, ha='center', va='center', fontsize=9, fontweight='bold')

        # Arrows from input to models
        ax.annotate('', xy=(x, y+0.5), xytext=(5, 7.5),
                   arrowprops=dict(arrowstyle='->', lw=1.5, color='gray', alpha=0.5))

    # Soft voting
    ax.text(5, 4.5, 'Soft Voting (Average Probabilities)', ha='center', fontsize=10, style='italic')
    ax.text(5, 4.0, r'$P_{ens}(CDW|x) = \frac{1}{4}\sum_{i=1}^{4} P_i(CDW|x)$',
           ha='center', fontsize=11, bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))

    # Arrows from models to voting
    for model_name, x, y in models:
        ax.annotate('', xy=(5, 4.5), xytext=(x, y-0.5),
                   arrowprops=dict(arrowstyle='->', lw=1.5, color='gray', alpha=0.5))

    # Output
    output_box = mpatches.FancyBboxPatch((3.5, 2.5), 3, 0.6,
                                         boxstyle="round,pad=0.1",
                                         edgecolor='darkred', facecolor='lightcoral', linewidth=2)
    ax.add_patch(output_box)
    ax.text(5, 2.8, 'P(CDW) ∈ [0, 1]', ha='center', va='center', fontsize=10, fontweight='bold')

    # Arrow from voting to output
    ax.annotate('', xy=(5, 2.5), xytext=(5, 4.0),
               arrowprops=dict(arrowstyle='->', lw=2, color='black'))

    # Training info box
    info_text = (
        'Training Configuration:\n'
        '• Batch size: 16 (disk streaming)\n'
        '• Optimizer: AdamW (5e-4 head, 5e-5 backbone)\n'
        '• Regularization: Label smoothing (0.05) + Mixup (α=0.3)\n'
        '• Scheduling: Cosine annealing'
    )
    ax.text(0.5, 1.0, info_text, fontsize=8, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

    # Results box
    results_text = (
        'Test Performance (56,521 held-out):\n'
        '• AUC: 0.9885  • F1: 0.9819 @ t=0.40\n'
        '• CDW accuracy: 99.2%  • NO_CDW accuracy: 97.1%'
    )
    ax.text(8.0, 1.0, results_text, fontsize=8, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.7))

    output_path = output_dir / "ensemble_architecture_diagram.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_path}")
    plt.close()


def create_probability_distribution_comparison(output_dir: Path):
    """Create probability distribution comparison visualization."""
    print("\n[Creating probability distribution comparison...]")

    df_orig = pd.read_csv("data/chm_variants/labels_canonical_with_splits.csv")
    df_retrain = pd.read_csv("data/chm_variants/labels_canonical_with_splits_retrained_ensemble.csv")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Overall distributions
    ax = axes[0, 0]
    ax.hist(df_orig['model_prob'].dropna(), bins=50, alpha=0.6, label='Original (A)', color='blue', edgecolor='black')
    ax.hist(df_retrain['model_prob'].dropna(), bins=50, alpha=0.6, label='Retrained (B)', color='orange', edgecolor='black')
    ax.set_xlabel('P(CDW)')
    ax.set_ylabel('Count')
    ax.set_title('Overall Probability Distribution', fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

    # By class label
    ax = axes[0, 1]
    for label_val in ['cdw', 'no_cdw']:
        mask_orig = df_orig['label'] == label_val
        mask_retrain = df_retrain['label'] == label_val

        color = 'red' if label_val == 'cdw' else 'green'
        label_name = 'CWD' if label_val == 'cdw' else 'Background'

        ax.hist(df_retrain[mask_retrain]['model_prob'], bins=40, alpha=0.5,
               label=label_name, color=color, edgecolor='black')

    ax.set_xlabel('P(CDW)')
    ax.set_ylabel('Count')
    ax.set_title('Retrained Ensemble - By Class', fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

    # Probability change distribution
    ax = axes[1, 0]
    df_merged = pd.merge(
        df_orig[['raster', 'row_off', 'col_off', 'model_prob']].rename(columns={'model_prob': 'original_prob'}),
        df_retrain[['raster', 'row_off', 'col_off', 'model_prob']].rename(columns={'model_prob': 'retrained_prob'}),
        on=['raster', 'row_off', 'col_off']
    )
    df_merged['abs_diff'] = np.abs(df_merged['original_prob'] - df_merged['retrained_prob'])

    ax.hist(df_merged['abs_diff'], bins=60, alpha=0.7, color='purple', edgecolor='black')
    ax.axvline(df_merged['abs_diff'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {df_merged["abs_diff"].mean():.4f}')
    ax.axvline(df_merged['abs_diff'].median(), color='green', linestyle='--', linewidth=2, label=f'Median: {df_merged["abs_diff"].median():.4f}')
    ax.set_xlabel('|P_original - P_retrained|')
    ax.set_ylabel('Count')
    ax.set_title('Probability Change Magnitude', fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

    # Q-Q plot
    ax = axes[1, 1]
    orig_sample = df_orig['model_prob'].dropna().sample(min(5000, len(df_orig)))
    retrain_sample = df_retrain['model_prob'].dropna().sample(min(5000, len(df_retrain)))

    stats.probplot(orig_sample, dist="norm", plot=ax)
    ax.set_title('Q-Q Plot: Original Distribution vs Normal', fontweight='bold')
    ax.grid(alpha=0.3)

    plt.suptitle('Probability Distribution Analysis: Option A vs Option B',
                fontsize=14, fontweight='bold')

    output_path = output_dir / "probability_distribution_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_path}")
    plt.close()


def create_test_metrics_visualization(output_dir: Path):
    """Create test metrics visualization."""
    print("\n[Creating test metrics visualization...]")

    with open("output/tile_labels_spatial_splits/training_metadata.json") as f:
        meta = json.load(f)

    tm = meta['test_metrics']

    fig = plt.figure(figsize=(14, 5))

    # AUC
    ax1 = plt.subplot(1, 3, 1)
    ax1.bar([0], [tm['ensemble_auc']], color='steelblue', alpha=0.7, edgecolor='black', linewidth=2)
    ax1.set_ylabel('AUC Score')
    ax1.set_ylim([0.95, 1.0])
    ax1.set_xticks([0])
    ax1.set_xticklabels(['Ensemble'])
    ax1.set_title('Test Set AUC', fontweight='bold', fontsize=11)
    ax1.text(0, tm['ensemble_auc'] - 0.01, f"{tm['ensemble_auc']:.4f}",
            ha='center', va='top', fontweight='bold', fontsize=11)
    ax1.grid(axis='y', alpha=0.3)

    # F1 Score
    ax2 = plt.subplot(1, 3, 2)
    ax2.bar([0], [tm['ensemble_f1']], color='darkorange', alpha=0.7, edgecolor='black', linewidth=2)
    ax2.set_ylabel('F1 Score')
    ax2.set_ylim([0.95, 1.0])
    ax2.set_xticks([0])
    ax2.set_xticklabels(['Ensemble'])
    ax2.set_title('Test Set F1 @ Optimal Threshold', fontweight='bold', fontsize=11)
    ax2.text(0, tm['ensemble_f1'] - 0.01, f"{tm['ensemble_f1']:.4f}",
            ha='center', va='top', fontweight='bold', fontsize=11)
    ax2.grid(axis='y', alpha=0.3)

    # Class distribution in test set
    ax3 = plt.subplot(1, 3, 3)
    sizes = [tm['n_cdw'], tm['n_test'] - tm['n_cdw']]
    labels = [f"CDW\n({tm['n_cdw']:,})", f"Background\n({tm['n_test'] - tm['n_cdw']:,})"]
    colors = ['#ff9999', '#66b3ff']
    wedges, texts, autotexts = ax3.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
                                        startangle=90, textprops={'fontsize': 10, 'fontweight': 'bold'})
    ax3.set_title(f'Test Set Distribution\n(Total: {tm["n_test"]:,})', fontweight='bold', fontsize=11)

    plt.suptitle(f'Option B Test Performance (Threshold = {tm["ensemble_thresh"]:.2f})',
                fontsize=12, fontweight='bold')

    output_path = output_dir / "test_metrics_visualization.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_path}")
    plt.close()


def create_option_comparison_chart(output_dir: Path):
    """Create Option A vs Option B comparison chart."""
    print("\n[Creating Option A vs Option B comparison...]")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Data
    categories = ['Training\nData', 'Test Set', 'Distribution\nShift', 'Test AUC', 'Test F1']
    option_a = [19.812, 2.186, 6.0, 0.0, 0.0]  # placeholders for A
    option_b = [67.290, 56.521, 5.51, 0.9885, 0.9819]

    x = np.arange(len(categories[:3]))
    width = 0.35

    ax = axes[0]
    bars1 = ax.bar(x - width/2, [option_a[0], option_a[1], option_a[2]], width,
                   label='Option A', color='lightcoral', alpha=0.7, edgecolor='black')
    bars2 = ax.bar(x + width/2, [option_b[0], option_b[1], option_b[2]], width,
                   label='Option B', color='lightgreen', alpha=0.7, edgecolor='black')

    ax.set_ylabel('Value', fontweight='bold')
    ax.set_title('Data Strategy Comparison', fontweight='bold', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(['Training Data\n(K tiles)', 'Test Set\n(K tiles)', 'Distribution\nShift (%)'])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.1f}', ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.1f}', ha='center', va='bottom', fontsize=9)

    # Metrics comparison
    ax = axes[1]
    metrics = ['AUC', 'F1 Score']
    values_b = [0.9885, 0.9819]
    bars = ax.bar(metrics, values_b, color=['steelblue', 'darkorange'], alpha=0.7, edgecolor='black', linewidth=2)
    ax.set_ylabel('Score', fontweight='bold')
    ax.set_title('Option B Test Performance', fontweight='bold', fontsize=12)
    ax.set_ylim([0.95, 1.0])
    ax.grid(axis='y', alpha=0.3)

    for bar, val in zip(bars, values_b):
        ax.text(bar.get_x() + bar.get_width()/2., val - 0.005,
               f'{val:.4f}', ha='center', va='top', fontweight='bold', fontsize=10)

    plt.suptitle('Option A vs Option B: Comprehensive Comparison',
                fontsize=13, fontweight='bold')

    output_path = output_dir / "option_comparison_chart.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_path}")
    plt.close()


def main():
    output_dir = Path("output/thesis_visualizations")
    output_dir.mkdir(exist_ok=True)

    print("="*80)
    print("CREATING THESIS VISUALIZATIONS")
    print("="*80)

    create_ensemble_architecture_diagram(output_dir)
    create_test_metrics_visualization(output_dir)
    create_option_comparison_chart(output_dir)
    create_probability_distribution_comparison(output_dir)
    create_top10_changes_visualization(output_dir)

    print("\n" + "="*80)
    print("ALL VISUALIZATIONS CREATED SUCCESSFULLY")
    print("="*80)
    print(f"\nOutputs saved to: {output_dir}")
    print("\nGenerated files:")
    for f in sorted(output_dir.glob("*.png")):
        print(f"  - {f.name}")


if __name__ == "__main__":
    main()
