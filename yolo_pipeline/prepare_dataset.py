"""
Dataset Preparation Script for Timika-30K YOLO-Seg Pipeline.

Converts disease or organ segmentation masks into normalized YOLO polygon .txt labels
and organizes images and annotations into standard Ultralytics YOLO layout:

yolo_dataset/
├── dataset.yaml
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    ├── val/
    └── test/
"""

import os
import shutil
import argparse
import hashlib
from pathlib import Path
import pandas as pd
import numpy as np
import cv2
from typing import List, Tuple, Dict, Optional, Union
import yaml
from tqdm import tqdm

from config import (
    BASE_DATASET_DIR,
    DEFAULT_OUTPUT_DIR,
    PREPROCESSED_MODALITIES,
    DEFAULT_PREPROCESSING_MODE,
    DISEASE_CLASSES,
    DISEASE_CLASS_TO_ID,
    DISEASE_ID_TO_CLASS,
    ORGAN_CLASSES,
    ORGAN_CLASS_TO_ID,
)


def mask_to_yolo_polygons(mask: np.ndarray, class_id: int, min_contour_area: int = 15) -> List[str]:
    """
    Converts a binary or multiclass mask array into normalized YOLO segmentation polygon lines.
    Format: <class_id> <x1> <y1> <x2> <y2> ... <xn> <yn>
    """
    h, w = mask.shape[:2]
    if h == 0 or w == 0 or np.max(mask) == 0:
        return []

    _, thresh = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    yolo_lines = []
    for cnt in contours:
        if cv2.contourArea(cnt) < min_contour_area:
            continue

        epsilon = 0.003 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        coords = approx.reshape(-1, 2)

        if len(coords) < 3:
            continue

        norm_coords = []
        for x, y in coords:
            norm_x = max(0.0, min(1.0, round(float(x) / float(w), 6)))
            norm_y = max(0.0, min(1.0, round(float(y) / float(h), 6)))
            norm_coords.extend([norm_x, norm_y])

        line = f"{class_id} " + " ".join(map(str, norm_coords))
        yolo_lines.append(line)

    return yolo_lines


def create_composite_fusion_image(img_id: str, clahe_rel: str = "") -> Optional[np.ndarray]:
    """
    Fuses 3 complementary modalities into a 3-channel (RGB) image:
    Channel 0 (R): CLAHE Image (soft tissue contrast)
    Channel 1 (G): Bone Suppression Image (ribs/clavicle removed)
    Channel 2 (B): High-pass Laplacian Edge map (sharp boundary detail)
    """
    base_fname = os.path.basename(clahe_rel) if (clahe_rel and pd.notna(clahe_rel)) else f"{img_id}.png"

    clahe_path = BASE_DATASET_DIR / "preprocessed_clahe" / "images" / base_fname
    bone_path = BASE_DATASET_DIR / "preprocessed_bone_suppressed" / "images" / base_fname

    if not clahe_path.exists():
        clahe_path = BASE_DATASET_DIR / "preprocessed_clahe" / "images" / f"{img_id}.png"
    if not bone_path.exists():
        bone_path = BASE_DATASET_DIR / "preprocessed_bone_suppressed" / "images" / f"{img_id}.png"

    if not clahe_path.exists() or not bone_path.exists():
        return None

    clahe_img = cv2.imread(str(clahe_path), cv2.IMREAD_GRAYSCALE)
    bone_img = cv2.imread(str(bone_path), cv2.IMREAD_GRAYSCALE)

    if clahe_img is None or bone_img is None:
        return None

    if clahe_img.shape != bone_img.shape:
        bone_img = cv2.resize(bone_img, (clahe_img.shape[1], clahe_img.shape[0]))

    # Edge map using Laplacian
    laplacian = cv2.Laplacian(clahe_img, cv2.CV_8U, ksize=3)
    edge_map = cv2.convertScaleAbs(laplacian)

    composite = cv2.merge([clahe_img, bone_img, edge_map])
    return composite


def get_patient_split(patient_id: str, split_val: str, seed: int = 42) -> str:
    """Determines train/val/test split while ensuring zero patient leakage across splits."""
    if str(split_val).strip() == "test":
        return "test"
    
    pid_str = str(patient_id) if pd.notna(patient_id) else "unknown"
    hash_val = int(hashlib.md5(f"{pid_str}_{seed}".encode("utf-8")).hexdigest(), 16) % 100

    if str(split_val).strip() == "train":
        return "train" if hash_val < 85 else "val"

    if hash_val < 80:
        return "train"
    elif hash_val < 90:
        return "val"
    else:
        return "test"


def prepare_disease_yolo_dataset(
    mode: str = DEFAULT_PREPROCESSING_MODE,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    max_samples: Optional[int] = None,
):
    print(f"==================================================")
    print(f"   TIMIKA-30K YOLO-SEG DATASET PREPARATION       ")
    print(f"==================================================")
    print(f"Preprocessing Mode: {mode}")
    print(f"Target Output Dir : {output_dir}")

    is_composite = (mode == "composite_fusion")
    if not is_composite:
        img_source_dir = PREPROCESSED_MODALITIES.get(mode)
        if not img_source_dir or not img_source_dir.exists():
            raise FileNotFoundError(f"Preprocessing source folder not found: {img_source_dir}")

    # Create directory structure
    output_dir = Path(output_dir)
    images_dir = output_dir / "images"
    labels_dir = output_dir / "labels"

    for split in ["train", "val", "test"]:
        (images_dir / split).mkdir(parents=True, exist_ok=True)
        (labels_dir / split).mkdir(parents=True, exist_ok=True)

    # Load manifests
    clahe_manifest_path = BASE_DATASET_DIR / "preprocessed_clahe" / "manifest.csv"
    prep_dis_manifest_path = BASE_DATASET_DIR / "preprocessed_labels" / "disease" / "manifest.csv"

    df_meta = pd.read_csv(clahe_manifest_path)
    df_dis = pd.read_csv(prep_dis_manifest_path)

    print(f"Total metadata images: {len(df_meta)}")
    print(f"Total disease label records: {len(df_dis)}")

    positive_dis = df_dis[df_dis["positive"] == True].copy()
    print(f"Positive disease label records: {len(positive_dis)}")

    dis_by_image = {}
    for _, row in positive_dis.iterrows():
        img_id = row["id"]
        cls_name = row["class"]
        rel_label = row["label_relative_path"]
        if img_id not in dis_by_image:
            dis_by_image[img_id] = []
        dis_by_image[img_id].append((cls_name, rel_label))

    if max_samples and max_samples > 0:
        df_meta = df_meta.head(max_samples)
        print(f"--> Truncated dataset to max {max_samples} samples for quick testing.")

    split_counts = {"train": 0, "val": 0, "test": 0}
    positive_samples = 0
    negative_samples = 0

    print("\nProcessing images and extracting YOLO-Seg polygons...")
    for _, row in tqdm(df_meta.iterrows(), total=len(df_meta)):
        img_id = row["id"]
        patient_id = row.get("patient_id", img_id)
        orig_split = row.get("split", "-")
        split = get_patient_split(patient_id, orig_split)
        dst_img_path = images_dir / split / f"{img_id}.png"
        dst_lbl_path = labels_dir / split / f"{img_id}.txt"

        clahe_rel = row.get("clahe_relative_path", "")
        if is_composite:
            comp_img = create_composite_fusion_image(img_id, clahe_rel)
            if comp_img is None:
                continue
            cv2.imwrite(str(dst_img_path), comp_img)
        else:
            img_fname = f"{img_id}.png"
            src_img_path = img_source_dir / "images" / img_fname
            if not src_img_path.exists():
                base_fname = os.path.basename(clahe_rel) if pd.notna(clahe_rel) else img_fname
                src_img_path = img_source_dir / "images" / base_fname

            if not src_img_path.exists():
                continue
            shutil.copy2(src_img_path, dst_img_path)

        # Generate YOLO polygon lines
        yolo_lines = []
        if img_id in dis_by_image:
            positive_samples += 1
            for cls_name, rel_label_path in dis_by_image[img_id]:
                if cls_name not in DISEASE_CLASS_TO_ID:
                    continue
                cls_id = DISEASE_CLASS_TO_ID[cls_name]
                mask_file = BASE_DATASET_DIR / "preprocessed_labels" / "disease" / rel_label_path
                if mask_file.exists():
                    mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
                    if mask is not None:
                        lines = mask_to_yolo_polygons(mask, cls_id)
                        yolo_lines.extend(lines)
        else:
            negative_samples += 1

        with open(dst_lbl_path, "w", encoding="utf-8") as f:
            if yolo_lines:
                f.write("\n".join(yolo_lines) + "\n")

        split_counts[split] += 1

    # Write dataset.yaml
    yaml_content = {
        "path": str(output_dir.resolve()).replace("\\", "/"),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": DISEASE_ID_TO_CLASS,
    }

    yaml_file = output_dir / "dataset.yaml"
    with open(yaml_file, "w", encoding="utf-8") as f:
        yaml.dump(yaml_content, f, default_flow_style=False, sort_keys=False)

    print("\n==================================================")
    print("   PREPARATION SUMMARY                            ")
    print("==================================================")
    print(f"Output Path       : {output_dir}")
    print(f"Dataset YAML File : {yaml_file}")
    print(f"Split Breakdown   : Train={split_counts['train']}, Val={split_counts['val']}, Test={split_counts['test']}")
    print(f"Images with Masks : {positive_samples}")
    print(f"Negative Controls : {negative_samples}")
    print("Dataset preparation successfully completed!\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare Timika-30K YOLO-Seg Dataset")
    parser.add_argument(
        "--mode",
        type=str,
        default=DEFAULT_PREPROCESSING_MODE,
        choices=["bone_suppression_clahe", "clahe", "bone_suppressed", "composite_fusion"],
        help="Preprocessing image modality (default: bone_suppression_clahe)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Destination directory for YOLO dataset layout",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit total samples processed (useful for rapid dry-runs)",
    )

    args = parser.parse_args()
    prepare_disease_yolo_dataset(
        mode=args.mode,
        output_dir=Path(args.output_dir),
        max_samples=args.max_samples,
    )
