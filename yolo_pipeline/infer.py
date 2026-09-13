"""
Inference, Side-by-Side Visualization, and Metrics Evaluation Script 
for Timika-30K YOLO-Seg Pipeline.

Runs trained YOLO-Seg models on validation/test chest X-rays:
1. Performs prediction with class-specific confidence thresholds and optional morphological mask cleaning.
2. Generates Side-by-Side (Ground Truth vs Prediction) visual comparisons.
3. Calculates & exports pixel-level Dice Coefficient, IoU, Precision, and Recall metrics to .txt and .csv.
"""

import argparse
from pathlib import Path
from typing import Union, Dict, List, Optional
import json
import cv2
import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO
from tqdm import tqdm

from config import DEFAULT_OUTPUT_DIR, DEFAULT_IMAGE_SIZE, DISEASE_CLASSES, DISEASE_CLASS_TO_ID

# Vibrant color palette (BGR) for 7 disease classes
CLASS_COLORS = [
    (0, 0, 255),     # Atelectasis - Red
    (255, 0, 0),     # Cavitation - Blue
    (0, 255, 0),     # Infiltrate - Green
    (0, 255, 255),   # Lymphadenopathy - Yellow
    (255, 0, 255),   # Pleural Effusion - Magenta
    (255, 165, 0),   # Pneumothorax - Orange
    (128, 0, 128),   # TB Lesion - Purple
]

# Default per-class confidence thresholds (customizable for subtle vs distinct diseases)
DEFAULT_CLASS_CONF = {
    0: 0.15,  # Atelectasis
    1: 0.20,  # Cavitation
    2: 0.12,  # Infiltrate (diffuse, lower thresh)
    3: 0.20,  # Lymphadenopathy
    4: 0.25,  # Pleural Effusion (distinct)
    5: 0.20,  # Pneumothorax
    6: 0.15,  # TB Lesion
}


def apply_morphology(mask: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """Applies morphological closing to smooth binary segmentation masks and eliminate noise."""
    if kernel_size <= 1 or mask.sum() == 0:
        return mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    # Close small holes
    closed_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return closed_mask


def parse_yolo_label(label_path: Path, img_h: int, img_w: int) -> dict:
    """Reads YOLO polygon label file and returns binary masks per class_id."""
    masks_by_class = {cls_id: np.zeros((img_h, img_w), dtype=np.uint8) for cls_id in range(len(DISEASE_CLASSES))}
    
    if not label_path.exists():
        return masks_by_class

    with open(label_path, "r") as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        if len(parts) < 7:
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


def draw_gt_overlay(image: np.ndarray, label_path: Path) -> np.ndarray:
    """Draws ground truth polygon masks and bounding boxes on an image."""
    h, w = image.shape[:2]
    canvas = image.copy()
    overlay = image.copy()

    if not label_path.exists():
        cv2.putText(canvas, "No GT Label Found", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        return canvas

    with open(label_path, "r") as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        if len(parts) < 7:
            continue
        
        cls_id = int(parts[0])
        color = CLASS_COLORS[cls_id % len(CLASS_COLORS)]
        cls_name = DISEASE_CLASSES[cls_id] if cls_id < len(DISEASE_CLASSES) else f"Class {cls_id}"

        coords = np.array(list(map(float, parts[1:])), dtype=np.float32).reshape(-1, 2)
        coords[:, 0] *= w
        coords[:, 1] *= h
        pts = coords.astype(np.int32)

        cv2.fillPoly(overlay, [pts], color=color)

        x_min, y_min = np.min(pts[:, 0]), np.min(pts[:, 1])
        x_max, y_max = np.max(pts[:, 0]), np.max(pts[:, 1])
        cv2.polylines(canvas, [pts], isClosed=True, color=color, thickness=2)
        cv2.rectangle(canvas, (x_min, y_min), (x_max, y_max), color, 2)

        tag = f"GT: {cls_name}"
        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(canvas, (x_min, max(0, y_min - th - 6)), (x_min + tw + 4, max(th + 6, y_min)), color, -1)
        cv2.putText(canvas, tag, (x_min + 2, max(th + 2, y_min - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.addWeighted(overlay, 0.35, canvas, 0.65, 0, canvas)
    return canvas


def create_side_by_side(gt_img: np.ndarray, pred_img: np.ndarray) -> np.ndarray:
    """Concatenates Ground Truth and Prediction images horizontally with headers."""
    h1, w1 = gt_img.shape[:2]
    h2, w2 = pred_img.shape[:2]
    if h1 != h2 or w1 != w2:
        pred_img = cv2.resize(pred_img, (w1, h1))

    header_h = 40
    gt_header = np.zeros((header_h, w1, 3), dtype=np.uint8)
    pred_header = np.zeros((header_h, w1, 3), dtype=np.uint8)

    cv2.rectangle(gt_header, (0, 0), (w1, header_h), (50, 50, 50), -1)
    cv2.rectangle(pred_header, (0, 0), (w1, header_h), (40, 40, 80), -1)

    cv2.putText(gt_header, "GROUND TRUTH", (15, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(pred_header, "YOLO PREDICTION", (15, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2, cv2.LINE_AA)

    gt_panel = np.vstack([gt_header, gt_img])
    pred_panel = np.vstack([pred_header, pred_img])

    return np.hstack([gt_panel, pred_panel])


def run_inference(
    weights: str,
    source: str,
    labels: str = None,
    imgsz: int = DEFAULT_IMAGE_SIZE,
    conf: float = 0.25,
    class_conf_dict: Optional[Dict[int, float]] = None,
    morph_kernel: int = 3,
    device: Union[int, str] = 0,
    project: str = "runs/predict",
    name: str = "timika_yolo26_predictions",
):
    print("==================================================")
    print("  TIMIKA-30K YOLO-SEG INFERENCE & EVAL LAUNCHER   ")
    print("==================================================")
    print(f"Weights Path   : {weights}")
    print(f"Source Input   : {source}")
    print(f"Base Conf      : {conf}")
    print(f"Class Conf Dict: {class_conf_dict if class_conf_dict else 'None (using base conf)'}")
    print(f"Morph Kernel   : {morph_kernel}")
    print(f"Image Size     : {imgsz}x{imgsz}")
    print(f"Output Path    : {project}/{name}")
    print("==================================================\n")

    weights_path = Path(weights)
    if not weights_path.exists():
        raise FileNotFoundError(f"Model weights file not found at '{weights}'. Check path or train a model first.")

    model = YOLO(str(weights_path))

    out_dir = Path(project) / name
    sbs_dir = out_dir / "side_by_side"
    sbs_dir.mkdir(parents=True, exist_ok=True)

    # Determine labels directory for GT
    source_path = Path(source)
    if labels:
        labels_dir = Path(labels)
    else:
        if "images" in str(source_path):
            labels_dir = Path(str(source_path).replace("images", "labels"))
        else:
            labels_dir = source_path.parent / "labels" / source_path.name

    # Determine lowest confidence threshold to pass to model.predict
    min_conf = min(class_conf_dict.values()) if class_conf_dict else conf

    # 1. Run YOLO Predictions
    results = model.predict(
        source=source,
        imgsz=imgsz,
        conf=min_conf,
        device=device,
        project=project,
        name=name,
        save=True,
        save_txt=True,
        save_conf=True,
        show_labels=True,
        show_conf=True,
        boxes=True,
    )

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

    print("\n[INFO] Processing Side-by-Side views & calculating post-processed metrics...")
    for res in tqdm(results, desc="Processing images & evaluation"):
        img_path = Path(res.path)
        pred_rendered = res.plot()

        raw_img = cv2.imread(str(img_path))
        if raw_img is None:
            raw_img = pred_rendered.copy()

        gt_label_path = labels_dir / f"{img_path.stem}.txt"
        gt_rendered = draw_gt_overlay(raw_img, gt_label_path)

        # Create Side-by-Side view
        sbs_image = create_side_by_side(gt_rendered, pred_rendered)
        sbs_out_path = sbs_dir / f"{img_path.stem}_sbs.jpg"
        cv2.imwrite(str(sbs_out_path), sbs_image)

        # Parse Ground Truth masks
        gt_masks = parse_yolo_label(gt_label_path, imgsz, imgsz)

        # Parse Prediction masks with class-specific confidence & morphology
        pred_masks = {cls_id: np.zeros((imgsz, imgsz), dtype=np.uint8) for cls_id in range(len(DISEASE_CLASSES))}
        if res.masks is not None:
            for cls_tensor, conf_tensor, mask_tensor in zip(res.boxes.cls, res.boxes.conf, res.masks.data):
                cls_id = int(cls_tensor.item())
                c_val = float(conf_tensor.item())

                # Check class-specific confidence threshold
                target_conf = class_conf_dict.get(cls_id, conf) if class_conf_dict else conf
                if c_val < target_conf:
                    continue

                m = mask_tensor.cpu().numpy()
                m_binary = (m > 0.5).astype(np.uint8)
                m_resized = cv2.resize(m_binary, (imgsz, imgsz), interpolation=cv2.INTER_NEAREST)
                
                # Apply morphological post-processing
                m_clean = apply_morphology(m_resized, kernel_size=morph_kernel)

                pred_masks[cls_id] = np.logical_or(pred_masks[cls_id], m_clean).astype(np.uint8)

        # Accumulate pixel statistics
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

    # 2. Aggregate Metrics Report
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

    out_csv = out_dir / "eval_metrics.csv"
    out_txt = out_dir / "eval_metrics.txt"

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

    print("==================================================")
    print("   INFERENCE & EVALUATION COMPLETED              ")
    print("==================================================")
    print(f"Predictions saved to            : {out_dir}")
    print(f"Side-by-Side views saved to     : {sbs_dir}")
    print(f"Evaluation metrics CSV saved to : {out_csv}")
    print(f"Evaluation metrics TXT saved to : {out_txt}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run YOLO-Seg Inference & Metrics Evaluation on Chest X-Rays")
    
    parser.add_argument(
        "--weights",
        type=str,
        default="runs/segment/runs/train/timika_yolo26s_seg/weights/best.pt",
        help="Path to trained model weights checkpoint (e.g. best.pt)",
    )
    parser.add_argument(
        "--source",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR / "images" / "test"),
        help="Path to image file, folder, or list of images for prediction",
    )
    parser.add_argument(
        "--labels",
        type=str,
        default=None,
        help="Path to ground truth labels folder (default: auto-detected from source path)",
    )
    parser.add_argument("--imgsz", type=int, default=DEFAULT_IMAGE_SIZE, help="Image resolution size (default: 512)")
    parser.add_argument("--conf", type=float, default=0.25, help="Base confidence threshold (default: 0.25)")
    parser.add_argument(
        "--use-class-conf",
        action="store_true",
        help="Use optimized class-specific confidence thresholds",
    )
    parser.add_argument(
        "--morph-kernel",
        type=int,
        default=3,
        help="Kernel size for OpenCV morphological mask closing (0 to disable)",
    )
    parser.add_argument("--device", default=0, help="GPU device ID or 'cpu' (default: 0)")
    parser.add_argument("--project", type=str, default="runs/predict", help="Output directory project name")
    parser.add_argument("--name", type=str, default="timika_yolo26_predictions", help="Prediction run folder name")

    args = parser.parse_args()

    class_conf_dict = DEFAULT_CLASS_CONF if args.use_class_conf else None

    run_inference(
        weights=args.weights,
        source=args.source,
        labels=args.labels,
        imgsz=args.imgsz,
        conf=args.conf,
        class_conf_dict=class_conf_dict,
        morph_kernel=args.morph_kernel,
        device=args.device,
        project=args.project,
        name=args.name,
    )
