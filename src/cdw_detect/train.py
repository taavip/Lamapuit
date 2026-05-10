"""
Training utilities for YOLO CDW detection.
"""

from pathlib import Path


def train(
    dataset_yaml: str,
    model: str = 'yolo11n-seg.pt',
    epochs: int = 50,
    batch: int = 4,
    imgsz: int = 640,
    patience: int = 15,
    project: str = 'runs/cdw_detect',
    name: str = 'train',
    device: str = 'cpu',
):
    """
    Train YOLO model for CDW segmentation.
    
    Args:
        dataset_yaml: Path to dataset.yaml
        model: Base model to start from
        epochs: Number of training epochs
        batch: Batch size
        imgsz: Image size
        patience: Early stopping patience
        project: Output project directory
        name: Run name
        device: Device to use ('cpu', '0', 'cuda')
    
    Returns:
        Path to best model weights
    """
    from ultralytics import YOLO
    import gc
    import torch
    
    # Clear memory before training
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    yolo = YOLO(model)
    
    # CPU-specific optimizations
    amp = False if device == 'cpu' else True
    
    results = yolo.train(
        data=dataset_yaml,
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        patience=patience,
        project=project,
        name=name,
        device=device,
        workers=0,  # Avoid multiprocessing issues on Windows
        cache='disk',
        exist_ok=True,
        verbose=True,
        amp=amp,  # Disable AMP on CPU (saves memory)
    )
    
    best_model = Path(project) / name / 'weights' / 'best.pt'
    print(f"Training complete. Best model: {best_model}")
    
    return best_model
