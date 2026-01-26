"""Fine-tune existing model with additional epochs.

This script continues training from a checkpoint for model improvement.

Usage:
    python scripts/finetune_model.py \\
        --model runs/cdw_detect/cdw_lamapuit_robust/weights/best.pt \\
        --data yolo_dataset_lamapuit_robust/dataset.yaml \\
        --epochs 10
"""

import argparse
import os
import sys
from pathlib import Path
import yaml
import shutil
from datetime import datetime


def load_config(config_path: str = None) -> dict:
    """Load configuration from YAML file."""
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    return {}


def finetune_model(
    model_path: str,
    data_yaml: str,
    epochs: int = 10,
    batch: int = 4,
    imgsz: int = 640,
    project: str = "runs/cdw_detect",
    name: str = "finetune_v1.0.0",
    **kwargs
):
    """Fine-tune model from checkpoint.
    
    Args:
        model_path: Path to existing model weights (best.pt)
        data_yaml: Path to dataset YAML
        epochs: Number of additional epochs
        batch: Batch size
        imgsz: Image size
        project: Project directory
        name: Experiment name (use semantic versioning)
        **kwargs: Additional YOLO training arguments
    """
    # Import here to avoid startup overhead
    from ultralytics import YOLO
    import torch
    import gc
    
    print("=" * 70)
    print(f"FINE-TUNING MODEL: {name}")
    print("=" * 70)
    print(f"Base model: {model_path}")
    print(f"Dataset: {data_yaml}")
    print(f"Additional epochs: {epochs}")
    print(f"Device: {'GPU' if torch.cuda.is_available() else 'CPU'}")
    print("=" * 70)
    
    # Memory cleanup before training
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Load model from checkpoint
    model = YOLO(model_path)
    
    # Training arguments with CPU optimizations
    train_args = {
        'data': data_yaml,
        'epochs': epochs,
        'batch': batch,
        'imgsz': imgsz,
        'patience': 5,  # Shorter patience for fine-tuning
        'device': 'cpu',
        'workers': 0,
        'project': project,
        'name': name,
        'exist_ok': True,
        'pretrained': False,  # Already using fine-tuned weights
        'optimizer': 'SGD',
        'lr0': 0.001,  # Lower learning rate for fine-tuning
        'lrf': 0.01,
        'momentum': 0.937,
        'weight_decay': 0.0005,
        'warmup_epochs': 1,
        'warmup_momentum': 0.8,
        'warmup_bias_lr': 0.1,
        'box': 7.5,
        'cls': 0.5,
        'dfl': 1.5,
        'plots': True,
        'save': True,
        'save_period': -1,
        'cache': 'disk',
        'amp': False,  # Critical: Disable AMP on CPU
        'verbose': True,
    }
    
    # Override with any custom args
    train_args.update(kwargs)
    
    print("\nTraining configuration:")
    for key, value in train_args.items():
        print(f"  {key}: {value}")
    print()
    
    # Train
    try:
        results = model.train(**train_args)
        
        print("\n" + "=" * 70)
        print("FINE-TUNING COMPLETED!")
        print("=" * 70)
        
        # Get output paths
        output_dir = Path(project) / name
        best_model = output_dir / 'weights' / 'best.pt'
        
        print(f"Best model: {best_model}")
        print(f"Model size: {best_model.stat().st_size / (1024*1024):.2f} MB")
        
        # Create version info file
        version_info = {
            'version': name.split('_')[-1] if '_v' in name else '1.0.0',
            'timestamp': datetime.now().isoformat(),
            'base_model': str(model_path),
            'epochs_trained': epochs,
            'dataset': str(data_yaml),
            'final_metrics': {
                'box_map50': float(results.results_dict.get('metrics/mAP50(B)', 0)),
                'mask_map50': float(results.results_dict.get('metrics/mAP50(M)', 0)),
            }
        }
        
        version_file = output_dir / 'version_info.yaml'
        with open(version_file, 'w') as f:
            yaml.dump(version_info, f)
        
        print(f"Version info: {version_file}")
        print("\nNext steps:")
        print(f"  1. Test model: python scripts/run_detection.py --model {best_model}")
        print(f"  2. Create release: See MODEL_VERSIONING_GUIDE.md")
        print("=" * 70)
        
        return str(best_model)
        
    except Exception as e:
        print(f"\n❌ Error during fine-tuning: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        # Cleanup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description='Fine-tune CDW detection model')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to base model weights (best.pt)')
    parser.add_argument('--data', type=str, required=True,
                        help='Path to dataset YAML file')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of additional epochs (default: 10)')
    parser.add_argument('--batch', type=int, default=4,
                        help='Batch size (default: 4)')
    parser.add_argument('--imgsz', type=int, default=640,
                        help='Image size (default: 640)')
    parser.add_argument('--project', type=str, default='runs/cdw_detect',
                        help='Project directory')
    parser.add_argument('--name', type=str, default='finetune_v1.0.0',
                        help='Experiment name (use semantic versioning like finetune_v1.0.0)')
    parser.add_argument('--config', type=str,
                        help='Optional config file to load defaults from')
    
    args = parser.parse_args()
    
    # Verify inputs
    if not os.path.exists(args.model):
        print(f"❌ Error: Model not found: {args.model}")
        sys.exit(1)
    
    if not os.path.exists(args.data):
        print(f"❌ Error: Dataset YAML not found: {args.data}")
        sys.exit(1)
    
    # Load config if provided
    config = load_config(args.config) if args.config else {}
    
    # Fine-tune
    finetune_model(
        model_path=args.model,
        data_yaml=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        project=args.project,
        name=args.name,
        **config
    )


if __name__ == "__main__":
    main()
