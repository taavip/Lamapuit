#!/usr/bin/env python
"""
Train YOLO model for CDW detection.

Usage:
    python scripts/train_model.py --data data/dataset/dataset.yaml --epochs 50
"""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from cdw_detect.train import train


def main():
    parser = argparse.ArgumentParser(description='Train YOLO CDW model')
    parser.add_argument('--data', required=True, help='Path to dataset.yaml')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch', type=int, default=4, help='Batch size')
    parser.add_argument('--model', default='yolo11n-seg.pt', help='Base model')
    parser.add_argument('--device', default='cpu', help='Device (cpu or 0 for GPU)')
    parser.add_argument('--name', default='cdw_model', help='Run name')
    
    args = parser.parse_args()
    
    print(f"Training CDW Model")
    print(f"  Dataset: {args.data}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Device: {args.device}")
    print()
    
    best_model = train(
        dataset_yaml=args.data,
        model=args.model,
        epochs=args.epochs,
        batch=args.batch,
        device=args.device,
        name=args.name,
    )
    
    print(f"\nTraining complete!")
    print(f"Best model: {best_model}")


if __name__ == '__main__':
    main()
