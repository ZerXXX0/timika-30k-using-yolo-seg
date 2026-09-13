"""
Evaluation script for calculating Dice Coefficient, Mean IoU, Precision, and Recall 
on Segmentation Masks for the Timika-30K Dataset.

Computes both Global Pixel-Level Metrics (Standard Medical Image Segmentation Benchmark)
and Positive-Only Sample Metrics (lesion-present images only).
"""

import argparse
from pathlib import Path
import numpy as np
import cv2
import torch
from tqdm import tqdm
from ultralytics import YOLO
import pandas as pd

from config import DISEASE_CLASSES, DEFAULT_IMAGE_SIZE


def parse_yolo_label(label_path: Path, img_h: int, img_w: int) -> dict:
    """Reads YOLO polygon label file and returns binary masks per class_id."""
    masks_by_class = {cls_id: np.zeros((img_h, img_w), dtype=np.uint8) for cls_id in range(len(DISEASE_CLASSES))}
    
    if not label_path.exists():
        return masks_by_class

    with open(label_path, "r") as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        if len(parts) < 7:  # class_id + at least 3 points (6 coords)
            continue
        
        cls_id = int(parts[0])
        if cls_id not in masks_by_class:
            continue
        
        coords = np.array(list(map(float, parts[1:])), dtype=np.float32).reshape(-1, 2)
        coords[:, 0] *= img_w
        coords[:, 1] *= img_h
        poly_pts = coords.astype(np.int32)

        cv2.fillPoly(masks_by_class[cls_id], [poly_pts], color=1)

    return masks_by_class


def evaluate(weights: str, images_dir: str, labels_dir: str, imgsz: int = 512, conf: float = 0.25):
    print("==================================================")
    print("   EVALUATING DICE, IOU, PRECISION & RECALL      ")
    print("==================================================")
    print(f"Weights : {weights}")
    print(f"Images  : {images_dir}")
    print(f"Labels  : {labels_dir}")
    print(f"Conf    : {conf}")
    print("==================================================\n")

    model = YOLO(weights)
    img_paths = list(Path(images_dir).glob("*.png")) + list(Path(images_dir).glob("*.jpg"))
    labels_dir_path = Path(labels_dir)

    if not img_paths:
        print(f"Error: No images found in '{images_dir}'")
        return

    class_stats = {
        cls_id: {
            "intersection": 0,
            "union": 0,
            "gt_pixels": 0,
            "pred_pixels": 0,
            "num_positive_gt_samples": 0,
            "pos_sample_dices": [],
            "pos_sample_ious": [],
            "pos_sample_precs": [],
            "pos_sample_recs": []
        }
        for cls_id in range(len(DISEASE_CLASSES))
    }

    for img_path in tqdm(img_paths, desc="Evaluating test set"):
        # Load GT masks
        label_path = labels_dir_path / f"{img_path.stem}.txt"
        gt_masks = parse_yolo_label(label_path, imgsz, imgsz)

        # Run inference
        results = model.predict(source=str(img_path), imgsz=imgsz, conf=conf, verbose=False)[0]

        pred_masks = {cls_id: np.zeros((imgsz, imgsz), dtype=np.uint8) for cls_id in range(len(DISEASE_CLASSES))}
        if results.masks is not None:
            for cls_tensor, mask_tensor in zip(results.boxes.cls, results.masks.data):
                cls_id = int(cls_tensor.item())
                m = mask_tensor.cpu().numpy()
                m_resized = cv2.resize((m > 0.5).astype(np.uint8), (imgsz, imgsz), interpolation=cv2.INTER_NEAREST)
                pred_masks[cls_id] = np.logical_or(pred_masks[cls_id], m_resized).astype(np.uint8)

        # Accumulate pixel statistics per class
        for cls_id in range(len(DISEASE_CLASSES)):
            gt_m = gt_masks[cls_id]
            pr_m = pred_masks[cls_id]

            inter = np.logical_and(gt_m, pr_m).sum()
            un = np.logical_or(gt_m, pr_m).sum()
            gt_px = gt_m.sum()
            pr_px = pr_m.sum()

            class_stats[cls_id]["intersection"] += inter
            class_stats[cls_id]["union"] += un
            class_stats[cls_id]["gt_pixels"] += gt_px
            class_stats[cls_id]["pred_pixels"] += pr_px

            if gt_px > 0:
                class_stats[cls_id]["num_positive_gt_samples"] += 1
                sample_iou = inter / un if un > 0 else 0.0
                sample_dice = (2.0 * inter) / (gt_px + pr_px) if (gt_px + pr_px) > 0 else 0.0
                sample_prec = inter / pr_px if pr_px > 0 else 0.0
                sample_rec = inter / gt_px

                class_stats[cls_id]["pos_sample_ious"].append(sample_iou)
                class_stats[cls_id]["pos_sample_dices"].append(sample_dice)
                class_stats[cls_id]["pos_sample_precs"].append(sample_prec)
                class_stats[cls_id]["pos_sample_recs"].append(sample_rec)

    # Aggregate & print summary table
    summary_data = []

    for cls_id, cls_name in enumerate(DISEASE_CLASSES):
        st = class_stats[cls_id]
        gt_px = st["gt_pixels"]
        pr_px = st["pred_pixels"]
        inter = st["intersection"]
        un = st["union"]
        n_pos = st["num_positive_gt_samples"]

        g_dice = (2.0 * inter) / (gt_px + pr_px) if (gt_px + pr_px) > 0 else 0.0
        g_iou = inter / un if un > 0 else 0.0
        g_prec = inter / pr_px if pr_px > 0 else 0.0
        g_rec = inter / gt_px if gt_px > 0 else 0.0

        pos_dice = np.mean(st["pos_sample_dices"]) if n_pos > 0 else 0.0

        summary_data.append({
            "Class ID": cls_id,
            "Class Name": cls_name,
            "Pos Samples": n_pos,
            "Global Dice": f"{g_dice:.4f}",
            "Global IoU": f"{g_iou:.4f}",
            "Precision": f"{g_prec:.4f}",
            "Recall": f"{g_rec:.4f}",
            "Pos-Only Dice": f"{pos_dice:.4f}"
        })

    summary_df = pd.DataFrame(summary_data)

    mean_dice = np.mean([float(r["Global Dice"]) for r in summary_data])
    mean_iou = np.mean([float(r["Global IoU"]) for r in summary_data])
    mean_prec = np.mean([float(r["Precision"]) for r in summary_data])
    mean_rec = np.mean([float(r["Recall"]) for r in summary_data])
    mean_pos_dice = np.mean([float(r["Pos-Only Dice"]) for r in summary_data])

    print("\n=========================================================================================")
    print("             PIXEL-LEVEL SEGMENTATION EVALUATION RESULTS (GT LESION MATCHING)            ")
    print("=========================================================================================")
    print(summary_df.to_string(index=False))
    print("-----------------------------------------------------------------------------------------")
    print(f" OVERALL MEAN:  Global Dice={mean_dice:.4f} | Global IoU={mean_iou:.4f} | Precision={mean_prec:.4f} | Recall={mean_rec:.4f} | Pos-Only Dice={mean_pos_dice:.4f}")
    print("=========================================================================================\n")

    # Save to CSV and TXT
    out_dir = Path("runs")
    out_dir.mkdir(parents=True, exist_ok=True)

    out_csv = out_dir / "eval_dice_iou_metrics.csv"
    out_txt = out_dir / "eval_dice_iou_metrics.txt"

    summary_df.to_csv(out_csv, index=False)

    report_text = (
        "=========================================================================================\n"
        "             PIXEL-LEVEL SEGMENTATION EVALUATION RESULTS (GT LESION MATCHING)            \n"
        "=========================================================================================\n"
        f"{summary_df.to_string(index=False)}\n"
        "-----------------------------------------------------------------------------------------\n"
        f" OVERALL MEAN:  Global Dice={mean_dice:.4f} | Global IoU={mean_iou:.4f} | Precision={mean_prec:.4f} | Recall={mean_rec:.4f} | Pos-Only Dice={mean_pos_dice:.4f}\n"
        "=========================================================================================\n"
    )

    with open(out_txt, "w") as f:
        f.write(report_text)

    print(f"\nSaved evaluation metrics CSV report to : {out_csv.resolve()}")
    print(f"Saved evaluation metrics TXT report to : {out_txt.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Dice, IoU, Precision, Recall for YOLO Segmentation model")
    parser.add_argument(
        "--weights",
        type=str,
        default="runs/segment/runs/train/timika_yolo26s_seg/weights/best.pt",
        help="Path to trained weights file",
    )
    parser.add_argument(
        "--images",
        type=str,
        default="d:/Goji/timika-30k/yolo_dataset/images/test",
        help="Path to test images folder",
    )
    parser.add_argument(
        "--labels",
        type=str,
        default="d:/Goji/timika-30k/yolo_dataset/labels/test",
        help="Path to test labels folder",
    )
    parser.add_argument("--imgsz", type=int, default=DEFAULT_IMAGE_SIZE, help="Image resolution size (default: 512)")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold (default: 0.25)")

    args = parser.parse_args()
    evaluate(args.weights, args.images, args.labels, args.imgsz, args.conf)
