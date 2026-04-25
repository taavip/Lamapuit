"""
Train YOLO CDW detector with best practices:
- 50+ epochs
- Early stopping
- Proper train/val/test split
- Model checkpointing
- Learning rate scheduling
- Class balancing
"""
from ultralytics import YOLO
import yaml
from pathlib import Path
import torch
import matplotlib.pyplot as plt
import pandas as pd


def train_cdw_detector(
    data_yaml='C:/temp/Lamapuit/yolo_dataset_split/dataset.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    model='yolo11n-seg.pt',
    patience=20,
    project='C:/temp/Lamapuit/runs/cdw_yolo_best',
    name='cdw_detector_v1'
):
    """
    Train YOLO CDW detector with best practices
    """
    print("="*70)
    print("TRAINING YOLO CDW DETECTOR - BEST PRACTICES")
    print("="*70)
    
    # Check CUDA availability
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")
    if device == 'cpu':
        print("  WARNING: Training on CPU will be slow!")
        print("  Consider reducing batch size and epochs for testing")
    
    # Load dataset info
    with open(data_yaml, 'r') as f:
        data_config = yaml.safe_load(f)
    
    print(f"\nDataset: {data_config['path']}")
    print(f"  Classes: {data_config['names']}")
    print(f"  Train: {data_config['train']}")
    print(f"  Val: {data_config['val']}")
    print(f"  Test: {data_config['test']}")
    
    # Training configuration
    print(f"\nTraining Configuration:")
    print(f"  Model: {model}")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch}")
    print(f"  Image size: {imgsz}")
    print(f"  Patience: {patience} (early stopping)")
    
    # Load model
    print(f"\nLoading model...")
    yolo_model = YOLO(model)
    
    # Training parameters (best practices for CDW detection)
    train_args = {
        'data': data_yaml,
        'epochs': epochs,
        'imgsz': imgsz,
        'batch': batch,
        'device': device,
        'project': project,
        'name': name,
        'exist_ok': True,
        'patience': patience,  # Early stopping
        'save': True,  # Save checkpoints
        'save_period': 10,  # Save every 10 epochs
        'cache': False,  # Don't cache (large dataset)
        'pretrained': True,
        'optimizer': 'AdamW',  # Best for segmentation
        'lr0': 0.001,  # Initial learning rate
        'lrf': 0.01,  # Final learning rate (lr0 * lrf)
        'momentum': 0.937,
        'weight_decay': 0.0005,
        'warmup_epochs': 3,  # Warmup for stable training
        'warmup_momentum': 0.8,
        'box': 7.5,  # Box loss gain
        'cls': 0.5,  # Class loss gain
        'dfl': 1.5,  # DFL loss gain
        'close_mosaic': 10,  # Disable mosaic last N epochs
        'amp': True,  # Automatic Mixed Precision
        'fraction': 1.0,  # Use 100% of data
        'workers': 4,  # Data loading workers
        'verbose': True,
        'plots': True,  # Save plots
        # Augmentation (moderate for thin objects)
        'hsv_h': 0.015,
        'hsv_s': 0.4,
        'hsv_v': 0.2,
        'degrees': 90.0,  # Allow 90° rotation
        'translate': 0.1,
        'scale': 0.2,
        'shear': 0.0,  # No shear (preserves shape)
        'perspective': 0.0,  # No perspective (preserves shape)
        'flipud': 0.5,  # Vertical flip
        'fliplr': 0.5,  # Horizontal flip
        'mosaic': 0.5,  # Reduced mosaic
        'mixup': 0.0,  # No mixup for segmentation
        'copy_paste': 0.0,  # No copy-paste
    }
    
    print(f"\nStarting training...")
    print(f"{'='*70}\n")
    
    # Train
    results = yolo_model.train(**train_args)
    
    print(f"\n{'='*70}")
    print(f"TRAINING COMPLETE")
    print(f"{'='*70}")
    
    # Get results directory
    results_dir = Path(project) / name
    print(f"\nResults saved to: {results_dir}")
    
    # Evaluate on test set
    print(f"\nEvaluating on test set...")
    test_metrics = yolo_model.val(
        data=data_yaml,
        split='test',
        imgsz=imgsz,
        batch=batch,
        save_json=True,
        plots=True,
        project=project,
        name=f"{name}_test"
    )
    
    print(f"\nTest Set Results:")
    print(f"  mAP50-95(B): {test_metrics.box.map:.4f}")
    print(f"  mAP50(B): {test_metrics.box.map50:.4f}")
    print(f"  mAP75(B): {test_metrics.box.map75:.4f}")
    if hasattr(test_metrics, 'seg'):
        print(f"  mAP50-95(M): {test_metrics.seg.map:.4f}")
        print(f"  mAP50(M): {test_metrics.seg.map50:.4f}")
        print(f"  mAP75(M): {test_metrics.seg.map75:.4f}")
    
    # Analyze training curves
    analyze_training_results(results_dir)
    
    return yolo_model, results_dir


def analyze_training_results(results_dir):
    """
    Analyze training results and provide recommendations
    """
    results_csv = Path(results_dir) / 'results.csv'
    
    if not results_csv.exists():
        print(f"\nWARNING: results.csv not found in {results_dir}")
        return
    
    print(f"\n{'='*70}")
    print(f"TRAINING ANALYSIS")
    print(f"{'='*70}")
    
    # Load results
    df = pd.read_csv(results_csv)
    df.columns = df.columns.str.strip()  # Remove whitespace
    
    # Get final metrics
    final_epoch = df.iloc[-1]
    best_epoch = df['metrics/mAP50-95(B)'].idxmax()
    best_metrics = df.iloc[best_epoch]
    
    print(f"\nBest Epoch: {best_epoch + 1}")
    print(f"  Box mAP50-95: {best_metrics['metrics/mAP50-95(B)']:.4f}")
    print(f"  Box mAP50: {best_metrics['metrics/mAP50(B)']:.4f}")
    if 'metrics/mAP50-95(M)' in df.columns:
        print(f"  Mask mAP50-95: {best_metrics['metrics/mAP50-95(M)']:.4f}")
        print(f"  Mask mAP50: {best_metrics['metrics/mAP50(M)']:.4f}")
    
    print(f"\nFinal Epoch: {len(df)}")
    print(f"  Box mAP50-95: {final_epoch['metrics/mAP50-95(B)']:.4f}")
    print(f"  Box mAP50: {final_epoch['metrics/mAP50(B)']:.4f}")
    
    # Check convergence
    last_10_map = df['metrics/mAP50-95(B)'].tail(10).values
    map_std = last_10_map.std()
    map_trend = last_10_map[-1] - last_10_map[0]
    
    print(f"\nConvergence Analysis:")
    print(f"  Last 10 epochs mAP std: {map_std:.6f}")
    print(f"  Last 10 epochs mAP trend: {map_trend:+.6f}")
    
    if map_std < 0.001 and abs(map_trend) < 0.001:
        print(f"  Status: CONVERGED ✓")
    elif map_trend > 0.001:
        print(f"  Status: STILL IMPROVING - Consider more epochs")
    else:
        print(f"  Status: PLATEAUED")
    
    # Overfitting check
    if 'metrics/mAP50-95(B)' in df.columns and 'train/box_loss' in df.columns:
        train_loss_trend = df['train/box_loss'].tail(10).mean()
        val_map_trend = df['metrics/mAP50-95(B)'].tail(10).mean()
        
        print(f"\nOverfitting Check:")
        print(f"  Avg train loss (last 10): {train_loss_trend:.4f}")
        print(f"  Avg val mAP (last 10): {val_map_trend:.4f}")
        
        if best_epoch < len(df) - 10:
            print(f"  WARNING: Best epoch was {len(df) - best_epoch - 1} epochs ago")
            print(f"           Model may be overfitting")
    
    # Recommendations
    print(f"\n{'='*70}")
    print(f"RECOMMENDATIONS")
    print(f"{'='*70}")
    
    if best_metrics['metrics/mAP50(B)'] < 0.3:
        print("  ⚠ Low mAP50 (<30%)")
        print("     → Increase buffer width")
        print("     → Add more augmentation")
        print("     → Train longer")
    elif best_metrics['metrics/mAP50(B)'] < 0.5:
        print("  ✓ Moderate mAP50 (30-50%)")
        print("     → Fine-tune hyperparameters")
        print("     → Consider larger model (yolo11s-seg)")
    else:
        print("  ✓✓ Good mAP50 (>50%)")
        print("     → Model is performing well")
        print("     → Test on full rasters")
    
    # Create summary plot
    create_summary_plot(df, results_dir)


def create_summary_plot(df, results_dir):
    """Create summary training plot"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Training Summary', fontsize=16, fontweight='bold')
    
    epochs = df.index + 1
    
    # Plot 1: Box mAP
    ax1 = axes[0, 0]
    ax1.plot(epochs, df['metrics/mAP50(B)'], label='mAP50', linewidth=2)
    ax1.plot(epochs, df['metrics/mAP50-95(B)'], label='mAP50-95', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('mAP')
    ax1.set_title('Box mAP Progress')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Losses
    ax2 = axes[0, 1]
    ax2.plot(epochs, df['train/box_loss'], label='Box Loss', linewidth=2)
    ax2.plot(epochs, df['train/cls_loss'], label='Cls Loss', linewidth=2)
    ax2.plot(epochs, df['train/dfl_loss'], label='DFL Loss', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.set_title('Training Losses')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Precision/Recall
    ax3 = axes[1, 0]
    ax3.plot(epochs, df['metrics/precision(B)'], label='Precision', linewidth=2)
    ax3.plot(epochs, df['metrics/recall(B)'], label='Recall', linewidth=2)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Score')
    ax3.set_title('Precision & Recall')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Learning rate
    ax4 = axes[1, 1]
    lr_cols = [col for col in df.columns if 'lr' in col.lower()]
    if lr_cols:
        ax4.plot(epochs, df[lr_cols[0]], linewidth=2)
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Learning Rate')
        ax4.set_title('Learning Rate Schedule')
        ax4.set_yscale('log')
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    summary_path = Path(results_dir) / 'training_summary.png'
    plt.savefig(summary_path, dpi=150, bbox_inches='tight')
    print(f"\nSummary plot saved: {summary_path}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch', type=int, default=16, help='Batch size')
    parser.add_argument('--imgsz', type=int, default=640, help='Image size')
    parser.add_argument('--patience', type=int, default=20, help='Early stopping patience')
    
    args = parser.parse_args()
    
    train_cdw_detector(
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        patience=args.patience
    )
