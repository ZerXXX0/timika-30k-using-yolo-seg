"""
Consecutive Multi-Variant Training Runner for YOLO26-Seg.

Trains all 5 YOLO26-Seg model variants consecutively (n -> s -> m -> l -> x):
  1. yolo26n-seg (Nano)
  2. yolo26s-seg (Small)
  3. yolo26m-seg (Medium)
  4. yolo26l-seg (Large)
  5. yolo26x-seg (Extra-Large)

Saves checkpoints for each model into distinct run folders and prints a summary
comparing all trained model checkpoints.
"""

from typing import List, Union
import time
import argparse
from pathlib import Path

from config import (
    DEFAULT_OUTPUT_DIR,
    YOLO26_SEG_VARIANTS,
    DEFAULT_IMAGE_SIZE,
    DEFAULT_BATCH_SIZE,
    DEFAULT_EPOCHS,
    DEFAULT_DEVICE,
    DEFAULT_MASK_RATIO,
)
from train import train_yolo_seg


def train_all_consecutively(
    variants: List[str] = ["n", "s", "m", "l", "x"],
    data_yaml: Path = DEFAULT_OUTPUT_DIR / "dataset.yaml",
    imgsz: int = DEFAULT_IMAGE_SIZE,
    epochs: int = DEFAULT_EPOCHS,
    batch: int = DEFAULT_BATCH_SIZE,
    cls_gain: float = 1.0,
    mask_ratio: int = DEFAULT_MASK_RATIO,
    device: Union[int, str] = DEFAULT_DEVICE,
    workers: int = 8,
    project: str = "runs/train",
):
    print("==================================================")
    print("  CONSECUTIVE YOLO26-SEG MULTI-VARIANT RUNNER    ")
    print("==================================================")
    variant_sequence = " -> ".join([v.upper() for v in variants])
    print(f"Variants Sequence : {variant_sequence}")
    print(f"Dataset Spec      : {data_yaml}")
    print(f"Epochs per Model  : {epochs}")
    print(f"Batch Size        : {batch}")
    print(f"Image Resolution  : {imgsz}x{imgsz}")
    print(f"Class Loss Gain   : {cls_gain}")
    print(f"Mask Ratio        : {mask_ratio}")
    print(f"Target Project    : {project}")
    print("==================================================\n")

    summary_results = []
    total_start_time = time.time()

    for idx, var in enumerate(variants, 1):
        var_lower = var.lower()
        model_weights = YOLO26_SEG_VARIANTS.get(var_lower, f"yolo26{var_lower}-seg.pt")
        run_name = f"timika_yolo26{var_lower}_seg"

        print(f"\n==================================================")
        print(f" [{idx}/{len(variants)}] LAUNCHING VARIANT '{var.upper()}' -> {model_weights}")
        print(f"==================================================")

        start_time = time.time()
        try:
            results = train_yolo_seg(
                model_name=model_weights,
                data_yaml=Path(data_yaml),
                imgsz=imgsz,
                epochs=epochs,
                batch=batch,
                device=device,
                workers=workers,
                cls_gain=cls_gain,
                mask_ratio=mask_ratio,
                project=project,
                name=run_name,
            )
            elapsed = time.time() - start_time
            best_ckpt = Path(project) / run_name / "weights" / "best.pt"
            status = "SUCCESS" if best_ckpt.exists() else "FINISHED"
        except Exception as e:
            elapsed = time.time() - start_time
            best_ckpt = "N/A"
            status = f"FAILED: {e}"
            print(f"--> Variant {var.upper()} training failed with error: {e}")

        summary_results.append({
            "variant": var.upper(),
            "model_weights": model_weights,
            "run_name": run_name,
            "status": status,
            "best_checkpoint": str(best_ckpt),
            "elapsed_seconds": round(elapsed, 2)
        })

    total_elapsed = time.time() - total_start_time

    print("\n==================================================================================")
    print("                      CONSECUTIVE TRAINING SUMMARY                                ")
    print("==================================================================================")
    print(f"{'Variant':<10} | {'Model Weights':<16} | {'Status':<10} | {'Time (s)':<10} | {'Best Checkpoint Path'}")
    print("-" * 90)
    for res in summary_results:
        print(f"{res['variant']:<10} | {res['model_weights']:<16} | {res['status']:<10} | {res['elapsed_seconds']:<10} | {res['best_checkpoint']}")
    print("-" * 90)
    print(f"Total Pipeline Execution Time: {round(total_elapsed / 60.0, 2)} minutes\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train All YOLO26-Seg Model Variants Consecutively (n to x)")

    parser.add_argument(
        "--variants",
        nargs="+",
        default=["n", "s", "m", "l", "x"],
        help="Sequence of variants to train (default: n s m l x)",
    )
    parser.add_argument(
        "--data",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR / "dataset.yaml"),
        help="Path to dataset.yaml file",
    )
    parser.add_argument("--imgsz", type=int, default=DEFAULT_IMAGE_SIZE, help="Image resolution size (default: 512)")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="Number of training epochs per variant (default: 50)")
    parser.add_argument("--batch", type=int, default=DEFAULT_BATCH_SIZE, help="Training batch size (default: 16)")
    parser.add_argument("--cls-gain", type=float, default=1.0, help="Class loss gain weight (default: 1.0)")
    parser.add_argument("--mask-ratio", type=int, default=DEFAULT_MASK_RATIO, help="Mask resolution ratio during training (default: 4)")
    parser.add_argument("--device", default=DEFAULT_DEVICE, help="GPU device ID or 'cpu' (default: 0)")
    parser.add_argument("--workers", type=int, default=8, help="Dataloader worker threads (default: 8)")
    parser.add_argument("--project", type=str, default="runs/train", help="Run save directory project name")

    args = parser.parse_args()

    train_all_consecutively(
        variants=args.variants,
        data_yaml=Path(args.data),
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        cls_gain=args.cls_gain,
        mask_ratio=args.mask_ratio,
        device=args.device,
        workers=args.workers,
        project=args.project,
    )
