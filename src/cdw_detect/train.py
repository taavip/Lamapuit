"""
Training utilities for YOLO CDW detection.
"""

from pathlib import Path


def train(
    dataset_yaml: str,
    model: str = "yolo11n-seg.pt",
    epochs: int = 50,
    batch: int = 4,
    imgsz: int = 640,
    patience: int = 15,
    project: str = "runs/cdw_detect",
    name: str = "train",
    device: str = "cpu",
    copy_paste: float = 0.5,
    dropout: float = 0.1,
    cos_lr: bool = True,
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
        copy_paste: YOLO copy-paste augmentation probability (0-1).
            Copies CDW instances from positive tiles into other tiles at
            random positions/scales/orientations during training – the
            primary defence against overfitting on sparse labels.
        dropout: Feature-map dropout probability.  Small values (0.1) add
            regularisation without hurting convergence.
        cos_lr: Use cosine learning-rate schedule.  Smoothly decays LR to
            0 at the end of training, usually improving final accuracy.

    Returns:
        Path to best model weights

    Raises:
        FileNotFoundError: If dataset.yaml doesn't exist
        ValueError: If parameters are invalid
    """
    from ultralytics import YOLO
    import gc
    import torch
    import yaml

    # Validate inputs
    dataset_path = Path(dataset_yaml)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset config not found: {dataset_yaml}")

    # Validate dataset.yaml structure
    try:
        with open(dataset_path) as f:
            config = yaml.safe_load(f)
        if "train" not in config or "val" not in config or "names" not in config:
            raise ValueError("dataset.yaml must contain 'train', 'val', and 'names' keys")
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid dataset.yaml format: {e}")

    # Validate parameters
    if epochs < 1:
        raise ValueError(f"epochs must be >= 1, got {epochs}")
    if batch < 1:
        raise ValueError(f"batch must be >= 1, got {batch}")
    if imgsz < 32 or imgsz % 32 != 0:
        raise ValueError(f"imgsz must be multiple of 32 and >= 32, got {imgsz}")
    if patience < 1:
        raise ValueError(f"patience must be >= 1, got {patience}")
    if not (0.0 <= copy_paste <= 1.0):
        raise ValueError(f"copy_paste must be in [0, 1], got {copy_paste}")

    # Clear memory before training
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    try:
        yolo = YOLO(model)
    except Exception as e:
        raise ValueError(f"Failed to load model {model}: {e}")

    # CPU-specific optimizations
    amp = False if device == "cpu" else True
    workers = 0 if device == "cpu" else 8  # Use multiple workers for GPU

    results = yolo.train(
        data=dataset_yaml,
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        patience=patience,
        project=project,
        name=name,
        device=device,
        workers=workers,
        cache="disk",
        exist_ok=True,
        verbose=True,
        amp=amp,
        # --- Generalisation & regularisation ---
        copy_paste=copy_paste,  # paste CDW from other tiles → novel compositions
        dropout=dropout,  # feature-map dropout → less memorisation
        cos_lr=cos_lr,  # cosine LR decay → smoother convergence
        # --- Increased geometric diversity at training time ---
        degrees=45.0,  # random rotation ±45° (on top of our pre-augmented tiles)
        scale=0.5,  # random zoom ±50%
        mosaic=1.0,  # mosaic (default=1, combine 4 tiles per batch step)
        fliplr=0.5,
        flipud=0.25,
    )

    best_model = Path(project) / name / "weights" / "best.pt"
    print(f"Training complete. Best model: {best_model}")

    return best_model
