"""
YOLO-Seg Training Script for Timika-30K Dataset using YOLO26-Seg architecture.

Supports all model variants from nano (n) to extra-large (x):
  - yolo26n-seg.pt (Nano)
  - yolo26s-seg.pt (Small)
  - yolo26m-seg.pt (Medium) [Default]
  - yolo26l-seg.pt (Large)
  - yolo26x-seg.pt (Extra-Large)
"""

import argparse
from pathlib import Path
from typing import Union, Optional
from ultralytics import YOLO

from config import (
    DEFAULT_OUTPUT_DIR,
    YOLO26_SEG_VARIANTS,
    DEFAULT_IMAGE_SIZE,
    DEFAULT_BATCH_SIZE,
    DEFAULT_EPOCHS,
    DEFAULT_DEVICE,
    DEFAULT_MASK_RATIO,
)


def train_yolo_seg(
    model_name: str = "yolo26m-seg.pt",
    data_yaml: Path = DEFAULT_OUTPUT_DIR / "dataset.yaml",
    imgsz: int = DEFAULT_IMAGE_SIZE,
    epochs: int = DEFAULT_EPOCHS,
    batch: int = DEFAULT_BATCH_SIZE,
    device: Union[int, str] = DEFAULT_DEVICE,
    workers: int = 2,
    mask_ratio: int = DEFAULT_MASK_RATIO,
    overlap_mask: bool = True,
    cls_gain: float = 1.0,
    project: str = "runs/train",
    name: str = "timika_yolo26_seg",
):
    print("==================================================")
    print("    TIMIKA-30K YOLO26-SEG TRAINING LAUNCHER       ")
    print("==================================================")
    print(f"Model Base   : {model_name}")
    print(f"Dataset Spec : {data_yaml}")
    print(f"Image Size   : {imgsz}x{imgsz}")
    print(f"Epochs       : {epochs}")
    print(f"Batch Size   : {batch}")
    print(f"Class Loss   : {cls_gain}")
    print(f"Mask Ratio   : {mask_ratio}")
    print(f"Device       : {device}")
    print(f"Save Path    : {project}/{name}")
    print("==================================================\n")

    data_yaml = Path(data_yaml)
    if not data_yaml.exists():
        raise FileNotFoundError(
            f"Dataset configuration file not found at '{data_yaml}'. "
            "Please run 'python prepare_dataset.py' first!"
        )

    # Initialize YOLO Model
    print(f"Loading model weights: {model_name} ...")
    try:
        model = YOLO(model_name)
    except Exception as e:
        print(f"Note: Could not load exact weights file '{model_name}' directly ({e}).")
        fallback = "yolov8m-seg.pt" if "26" in model_name else "yolo11m-seg.pt"
        print(f"Falling back to base architecture weights: {fallback}")
        model = YOLO(fallback)

    # Execute Training
    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        workers=workers,
        project=project,
        name=name,
        exist_ok=True,
        pretrained=True,
        verbose=True,
        # Loss & Mask Quality Enhancements
        cls=cls_gain,
        mask_ratio=mask_ratio,
        overlap_mask=overlap_mask,
        # Augmentation hyperparameters optimized for Medical CXR
        degrees=10.0,       # Slight rotation
        scale=0.1,          # Slight scaling
        translate=0.05,     # Slight translation
        fliplr=0.5,         # Horizontal flip (bilateral chest symmetry)
        mosaic=0.5,         # Mosaic augmentation
    )

    print("\n==================================================")
    print("   TRAINING COMPLETED SUCCESSFULLY                ")
    print("==================================================")
    best_weights = Path(project) / name / "weights" / "best.pt"
    print(f"Best Weights Saved At: {best_weights.resolve()}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLO26-Seg Model on Timika-30K Dataset")
    
    parser.add_argument(
        "--variant",
        type=str,
        default="m",
        choices=["n", "s", "m", "l", "x"],
        help="YOLO26-Seg model variant size: n (nano), s (small), m (medium), l (large), x (extra-large). Default: m",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Explicit model checkpoint path or weights string (overrides --variant)",
    )
    parser.add_argument(
        "--data",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR / "dataset.yaml"),
        help="Path to dataset.yaml file",
    )
    parser.add_argument("--imgsz", type=int, default=DEFAULT_IMAGE_SIZE, help="Image size resolution (default: 512)")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="Number of training epochs (default: 50)")
    parser.add_argument("--batch", type=int, default=DEFAULT_BATCH_SIZE, help="Training batch size (default: 16)")
    parser.add_argument("--cls-gain", type=float, default=1.0, help="Class loss gain weight (default: 1.0)")
    parser.add_argument("--mask-ratio", type=int, default=DEFAULT_MASK_RATIO, help="Mask resolution ratio during training (default: 4)")
    parser.add_argument("--device", default=DEFAULT_DEVICE, help="GPU device ID or 'cpu' (default: 0)")
    parser.add_argument("--workers", type=int, default=2, help="Dataloader worker threads (default: 2)")
    parser.add_argument("--project", type=str, default="runs/train", help="Run save directory project name")
    parser.add_argument("--name", type=str, default="timika_yolo26_seg", help="Run experiment name")

    args = parser.parse_args()

    if args.model:
        selected_model = args.model
    else:
        selected_model = YOLO26_SEG_VARIANTS.get(args.variant.lower(), "yolo26m-seg.pt")

    train_yolo_seg(
        model_name=selected_model,
        data_yaml=Path(args.data),
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        cls_gain=args.cls_gain,
        mask_ratio=args.mask_ratio,
        project=args.project,
        name=args.name,
    )
