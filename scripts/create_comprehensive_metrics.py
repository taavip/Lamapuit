#!/usr/bin/env python3
"""
Create comprehensive validation and test metrics visualizations.

Generates:
1. Confusion matrix & detailed metrics (TP, TN, FP, FN)
2. ROC curve with AUC
3. Precision-Recall curve
4. Calibration curve
5. Per-threshold performance metrics
6. False positive and true negative analysis
7. Learning-like curves (performance vs threshold)
8. Class-specific metrics table
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, confusion_matrix,
    accuracy_score, precision_score, recall_score, f1_score,
    matthews_corrcoef
)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 10


def load_predictions_and_labels():
    """Load all model predictions and true labels."""
    print("\n[Loading predictions and labels...]")

    # Load retrained probabilities
    df = pd.read_csv("data/chm_variants/labels_canonical_with_splits_retrained_ensemble.csv")

    # Filter to test set
    df_test = df[df['split'] == 'test'].copy()

    # Convert labels to binary
    df_test['y_true'] = (df_test['label'] == 'cdw').astype(int)
    df_test['y_prob'] = df_test['model_prob']

    y_true = df_test['y_true'].values
    y_prob = df_test['y_prob'].values

    print(f"  Loaded {len(y_true):,} test samples")
    print(f"  CDW: {np.sum(y_true):,} (positive)")
    print(f"  Background: {len(y_true) - np.sum(y_true):,} (negative)")

    return y_true, y_prob, df_test


def compute_metrics_at_threshold(y_true, y_prob, threshold):
    """Compute all metrics at a given threshold."""
    y_pred = (y_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0  # recall / TPR
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0  # TNR
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0  # PPV
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0  # negative predictive value
    accuracy = (tp + tn) / len(y_true)
    f1 = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
    mcc = matthews_corrcoef(y_true, y_pred)

    return {
        'threshold': threshold,
        'tp': int(tp),
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn),
        'sensitivity': float(sensitivity),  # TPR / Recall
        'specificity': float(specificity),  # TNR
        'precision': float(precision),  # PPV
        'npv': float(npv),
        'accuracy': float(accuracy),
        'f1': float(f1),
        'mcc': float(mcc),
    }


def create_confusion_matrix_detailed(y_true, y_prob):
    """Create detailed confusion matrix visualization."""
    print("\n[Creating confusion matrix visualization...]")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Optimal threshold (from previous results: 0.40)
    threshold = 0.40
    y_pred = (y_prob >= threshold).astype(int)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    ax = axes[0]
    sns.heatmap(cm, annot=True, fmt=',.0f', cmap='Blues', cbar=False, ax=ax,
               xticklabels=['Predicted\nBackground', 'Predicted\nCDW'],
               yticklabels=['Actual\nBackground', 'Actual\nCDW'],
               annot_kws={'size': 12, 'weight': 'bold'})
    ax.set_title(f'Confusion Matrix (Threshold = {threshold:.2f})', fontweight='bold', fontsize=12)
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')

    # Detailed metrics table
    ax = axes[1]
    ax.axis('off')

    metrics = compute_metrics_at_threshold(y_true, y_prob, threshold)

    # Create detailed metrics text
    metrics_text = f"""
DETAILED PERFORMANCE METRICS (Threshold = {threshold:.2f})

CONFUSION MATRIX:
  True Positives (TP):  {metrics['tp']:>10,}    (correctly identified CWD)
  True Negatives (TN):  {metrics['tn']:>10,}    (correctly identified background)
  False Positives (FP): {metrics['fp']:>10,}    (background predicted as CWD)
  False Negatives (FN): {metrics['fn']:>10,}    (CWD predicted as background)

CLASSIFICATION METRICS:
  Accuracy (ACC):       {metrics['accuracy']:>10.4f}  (TP+TN)/(TP+TN+FP+FN)
  Precision (PPV):      {metrics['precision']:>10.4f}  TP/(TP+FP)
  Recall (TPR/Sens):    {metrics['sensitivity']:>10.4f}  TP/(TP+FN)
  Specificity (TNR):    {metrics['specificity']:>10.4f}  TN/(TN+FP)
  NPV:                  {metrics['npv']:>10.4f}  TN/(TN+FN)
  F1-Score:             {metrics['f1']:>10.4f}  2×(Prec×Recall)/(Prec+Recall)
  Matthews Corr Coeff:  {metrics['mcc']:>10.4f}  Balanced measure

STATISTICS:
  Total samples:        {len(y_true):>10,}
  Positive samples:     {np.sum(y_true):>10,}    ({100*np.sum(y_true)/len(y_true):.1f}%)
  Negative samples:     {len(y_true) - np.sum(y_true):>10,}    ({100*(1-np.sum(y_true)/len(y_true)):.1f}%)
"""

    ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes,
           fontfamily='monospace', fontsize=9, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.suptitle('Option B: Detailed Confusion Matrix and Metrics',
                fontsize=13, fontweight='bold')

    output_path = Path("output/thesis_visualizations/confusion_matrix_detailed.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_path}")
    plt.close()


def create_roc_and_pr_curves(y_true, y_prob):
    """Create ROC and Precision-Recall curves."""
    print("\n[Creating ROC and PR curves...]")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # ROC Curve
    fpr, tpr, thresholds_roc = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    ax = axes[0]
    ax.plot(fpr, tpr, color='darkorange', lw=2.5, label=f'ROC curve (AUC = {roc_auc:.4f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')

    # Mark optimal threshold
    optimal_idx = np.argmax(tpr - fpr)
    ax.plot(fpr[optimal_idx], tpr[optimal_idx], 'ro', markersize=10,
           label=f'Optimal Point (t={thresholds_roc[optimal_idx]:.2f})')

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate (1 - Specificity)', fontweight='bold')
    ax.set_ylabel('True Positive Rate (Sensitivity)', fontweight='bold')
    ax.set_title('ROC Curve', fontweight='bold', fontsize=12)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(alpha=0.3)

    # Precision-Recall Curve
    precision, recall, thresholds_pr = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(recall, precision)

    ax = axes[1]
    ax.plot(recall, precision, color='green', lw=2.5, label=f'PR curve (AUC = {pr_auc:.4f})')
    ax.axhline(y=np.sum(y_true) / len(y_true), color='gray', linestyle='--', lw=2,
              label=f'Baseline ({np.sum(y_true)/len(y_true):.2%})')

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall (True Positive Rate)', fontweight='bold')
    ax.set_ylabel('Precision', fontweight='bold')
    ax.set_title('Precision-Recall Curve', fontweight='bold', fontsize=12)
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(alpha=0.3)

    plt.suptitle('Option B: ROC and Precision-Recall Curves',
                fontsize=13, fontweight='bold')

    output_path = Path("output/thesis_visualizations/roc_pr_curves.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_path}")
    plt.close()


def create_threshold_analysis(y_true, y_prob):
    """Create performance metrics across different thresholds."""
    print("\n[Creating threshold analysis...]")

    # Compute metrics at multiple thresholds
    thresholds = np.linspace(0.0, 1.0, 101)
    metrics_list = [compute_metrics_at_threshold(y_true, y_prob, t) for t in thresholds]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Sensitivity vs Specificity
    ax = axes[0, 0]
    sensitivity = [m['sensitivity'] for m in metrics_list]
    specificity = [m['specificity'] for m in metrics_list]
    ax.plot(thresholds, sensitivity, label='Sensitivity (TPR)', lw=2.5, color='red')
    ax.plot(thresholds, specificity, label='Specificity (TNR)', lw=2.5, color='blue')
    ax.axvline(x=0.40, color='green', linestyle='--', linewidth=2, label='Optimal (t=0.40)')
    ax.set_xlabel('Decision Threshold')
    ax.set_ylabel('Score')
    ax.set_title('Sensitivity vs Specificity', fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

    # Precision and Recall
    ax = axes[0, 1]
    precision = [m['precision'] for m in metrics_list]
    recall = [m['sensitivity'] for m in metrics_list]  # Same as sensitivity
    ax.plot(thresholds, precision, label='Precision (PPV)', lw=2.5, color='orange')
    ax.plot(thresholds, recall, label='Recall (Sensitivity)', lw=2.5, color='purple')
    ax.axvline(x=0.40, color='green', linestyle='--', linewidth=2, label='Optimal (t=0.40)')
    ax.set_xlabel('Decision Threshold')
    ax.set_ylabel('Score')
    ax.set_title('Precision vs Recall', fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

    # F1 Score and Accuracy
    ax = axes[1, 0]
    f1 = [m['f1'] for m in metrics_list]
    accuracy = [m['accuracy'] for m in metrics_list]
    ax.plot(thresholds, f1, label='F1 Score', lw=2.5, color='red')
    ax.plot(thresholds, accuracy, label='Accuracy', lw=2.5, color='blue')
    ax.axvline(x=0.40, color='green', linestyle='--', linewidth=2, label='Optimal (t=0.40)')
    ax.set_xlabel('Decision Threshold')
    ax.set_ylabel('Score')
    ax.set_title('F1 Score vs Accuracy', fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

    # Matthews Correlation Coefficient
    ax = axes[1, 1]
    mcc = [m['mcc'] for m in metrics_list]
    ax.plot(thresholds, mcc, lw=2.5, color='darkgreen', label='MCC')
    ax.axvline(x=0.40, color='red', linestyle='--', linewidth=2, label='Optimal (t=0.40)')
    ax.fill_between(thresholds, mcc, alpha=0.3, color='darkgreen')
    ax.set_xlabel('Decision Threshold')
    ax.set_ylabel('MCC')
    ax.set_title('Matthews Correlation Coefficient', fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

    plt.suptitle('Option B: Performance Metrics Across Decision Thresholds',
                fontsize=13, fontweight='bold')

    output_path = Path("output/thesis_visualizations/threshold_analysis.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_path}")
    plt.close()

    return metrics_list


def create_calibration_curve(y_true, y_prob):
    """Create calibration curve."""
    print("\n[Creating calibration curve...]")

    fig, ax = plt.subplots(figsize=(10, 8))

    # Compute calibration using bins
    n_bins = 10
    prob_true, prob_pred = [], []

    for i in range(n_bins):
        bin_start = i / n_bins
        bin_end = (i + 1) / n_bins
        mask = (y_prob >= bin_start) & (y_prob < bin_end)

        if np.sum(mask) > 0:
            prob_true.append(np.mean(y_true[mask]))
            prob_pred.append(np.mean(y_prob[mask]))

    # Plot calibration curve
    ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Perfectly calibrated')
    ax.plot(prob_pred, prob_true, 'o-', lw=2, markersize=8, color='steelblue',
           label='Option B Ensemble')

    # Add reference lines
    ax.axhline(y=np.mean(y_true), color='gray', linestyle=':', lw=1.5, alpha=0.7,
              label=f'Positive class rate ({np.mean(y_true):.2%})')

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0])
    ax.set_xlabel('Mean Predicted Probability', fontweight='bold', fontsize=11)
    ax.set_ylabel('Fraction of Positives', fontweight='bold', fontsize=11)
    ax.set_title('Calibration Curve (Reliability Diagram)', fontweight='bold', fontsize=12)
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(alpha=0.3)

    output_path = Path("output/thesis_visualizations/calibration_curve.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_path}")
    plt.close()


def create_anomaly_analysis(y_true, y_prob, df_test):
    """Analyze high false positives and low true negatives."""
    print("\n[Creating anomaly analysis...]")

    y_pred = (y_prob >= 0.40).astype(int)

    # False positives: predicted CDW but actually background
    fp_mask = (y_pred == 1) & (y_true == 0)
    fp_indices = np.where(fp_mask)[0]
    fp_probs = y_prob[fp_mask]

    # True negatives: correctly predicted background
    tn_mask = (y_pred == 0) & (y_true == 0)
    tn_indices = np.where(tn_mask)[0]
    tn_probs = y_prob[tn_mask]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # False positive distribution
    ax = axes[0, 0]
    ax.hist(fp_probs, bins=40, color='red', alpha=0.7, edgecolor='black')
    ax.axvline(x=np.mean(fp_probs), color='darkred', linestyle='--', linewidth=2,
              label=f'Mean: {np.mean(fp_probs):.4f}')
    ax.set_xlabel('Predicted Probability')
    ax.set_ylabel('Count')
    ax.set_title(f'False Positives Distribution (N={len(fp_probs):,})', fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

    # True negative distribution
    ax = axes[0, 1]
    ax.hist(tn_probs, bins=40, color='blue', alpha=0.7, edgecolor='black')
    ax.axvline(x=np.mean(tn_probs), color='darkblue', linestyle='--', linewidth=2,
              label=f'Mean: {np.mean(tn_probs):.4f}')
    ax.set_xlabel('Predicted Probability')
    ax.set_ylabel('Count')
    ax.set_title(f'True Negatives Distribution (N={len(tn_probs):,})', fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

    # High FP examples (top 20)
    ax = axes[1, 0]
    top_fp_idx = np.argsort(fp_probs)[-20:][::-1]
    top_fp_probs = fp_probs[top_fp_idx]
    ax.barh(range(len(top_fp_probs)), top_fp_probs, color='red', alpha=0.7, edgecolor='black')
    ax.set_xlabel('Predicted Probability')
    ax.set_ylabel('Rank')
    ax.set_title('Top 20 False Positives (Highest Confidence)', fontweight='bold')
    ax.invert_yaxis()
    ax.grid(alpha=0.3, axis='x')

    # Low TN examples (bottom 20 out of TNs, i.e., highest FN-like)
    ax = axes[1, 1]
    bottom_tn_idx = np.argsort(tn_probs)[-20:][::-1]
    bottom_tn_probs = tn_probs[bottom_tn_idx]
    ax.barh(range(len(bottom_tn_probs)), bottom_tn_probs, color='blue', alpha=0.7, edgecolor='black')
    ax.set_xlabel('Predicted Probability')
    ax.set_ylabel('Rank')
    ax.set_title('Top 20 True Negatives (Highest Confidence)', fontweight='bold')
    ax.invert_yaxis()
    ax.grid(alpha=0.3, axis='x')

    plt.suptitle('Option B: False Positive and True Negative Analysis',
                fontsize=13, fontweight='bold')

    output_path = Path("output/thesis_visualizations/anomaly_analysis.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_path}")
    plt.close()


def create_metrics_summary_table(y_true, y_prob):
    """Create comprehensive metrics summary table."""
    print("\n[Creating metrics summary table...]")

    # Compute metrics at optimal threshold
    optimal_threshold = 0.40
    metrics = compute_metrics_at_threshold(y_true, y_prob, optimal_threshold)

    # Create summary CSV
    summary_df = pd.DataFrame([
        {
            'Metric': 'True Positives (TP)',
            'Value': metrics['tp'],
            'Description': 'Correctly identified CWD samples'
        },
        {
            'Metric': 'True Negatives (TN)',
            'Value': metrics['tn'],
            'Description': 'Correctly identified background samples'
        },
        {
            'Metric': 'False Positives (FP)',
            'Value': metrics['fp'],
            'Description': 'Background incorrectly classified as CWD'
        },
        {
            'Metric': 'False Negatives (FN)',
            'Value': metrics['fn'],
            'Description': 'CWD incorrectly classified as background'
        },
        {
            'Metric': 'Accuracy',
            'Value': f"{metrics['accuracy']:.4f}",
            'Description': '(TP+TN) / (TP+TN+FP+FN)'
        },
        {
            'Metric': 'Sensitivity (Recall/TPR)',
            'Value': f"{metrics['sensitivity']:.4f}",
            'Description': 'TP / (TP+FN) - ability to find CWD'
        },
        {
            'Metric': 'Specificity (TNR)',
            'Value': f"{metrics['specificity']:.4f}",
            'Description': 'TN / (TN+FP) - ability to identify background'
        },
        {
            'Metric': 'Precision (PPV)',
            'Value': f"{metrics['precision']:.4f}",
            'Description': 'TP / (TP+FP) - confidence in positive prediction'
        },
        {
            'Metric': 'Negative Predictive Value (NPV)',
            'Value': f"{metrics['npv']:.4f}",
            'Description': 'TN / (TN+FN) - confidence in negative prediction'
        },
        {
            'Metric': 'F1 Score',
            'Value': f"{metrics['f1']:.4f}",
            'Description': '2×(Precision×Recall)/(Precision+Recall)'
        },
        {
            'Metric': 'Matthews Correlation Coefficient',
            'Value': f"{metrics['mcc']:.4f}",
            'Description': 'Balanced measure for binary classification'
        },
    ])

    summary_df.to_csv("output/thesis_visualizations/metrics_summary.csv", index=False)
    print(f"  ✓ Saved: output/thesis_visualizations/metrics_summary.csv")

    return summary_df


def main():
    output_dir = Path("output/thesis_visualizations")
    output_dir.mkdir(exist_ok=True)

    print("="*80)
    print("CREATING COMPREHENSIVE METRICS VISUALIZATIONS")
    print("="*80)

    # Load data
    y_true, y_prob, df_test = load_predictions_and_labels()

    # Create visualizations
    create_confusion_matrix_detailed(y_true, y_prob)
    create_roc_and_pr_curves(y_true, y_prob)
    metrics_list = create_threshold_analysis(y_true, y_prob)
    create_calibration_curve(y_true, y_prob)
    create_anomaly_analysis(y_true, y_prob, df_test)
    metrics_df = create_metrics_summary_table(y_true, y_prob)

    print("\n" + "="*80)
    print("COMPREHENSIVE METRICS CREATED SUCCESSFULLY")
    print("="*80)
    print(f"\nNew visualizations:")
    print("  - confusion_matrix_detailed.png")
    print("  - roc_pr_curves.png")
    print("  - threshold_analysis.png")
    print("  - calibration_curve.png")
    print("  - anomaly_analysis.png")
    print("  - metrics_summary.csv")

    print("\n" + "="*80)
    print("METRICS SUMMARY")
    print("="*80)
    print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()
