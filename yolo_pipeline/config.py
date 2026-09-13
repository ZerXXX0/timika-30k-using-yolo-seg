"""
Configuration settings for Timika-30K YOLO-Seg Pipeline.
Supports Scenario 1 (Baseline) and Scenario 2 (CV Optimizations).
"""

import os
from pathlib import Path

# Base Paths
BASE_DATASET_DIR = Path(r"d:\Goji\timika-30k\timika-30k")
YOLO_PIPELINE_DIR = Path(r"d:\Goji\timika-30k\yolo_pipeline")

# Scenario Dataset Directories
SCENARIO1_DATASET_DIR = Path(r"d:\Goji\timika-30k\yolo_dataset")
SCENARIO2_DATASET_DIR = Path(r"d:\Goji\timika-30k\yolo_dataset_scenario2")

DEFAULT_OUTPUT_DIR = SCENARIO1_DATASET_DIR

# Preprocessing Folders
PREPROCESSED_MODALITIES = {
    "bone_suppression_clahe": BASE_DATASET_DIR / "preprocessed_bone_suppression_clahe",
    "clahe": BASE_DATASET_DIR / "preprocessed_clahe",
    "bone_suppressed": BASE_DATASET_DIR / "preprocessed_bone_suppressed",
    "composite_fusion": BASE_DATASET_DIR / "preprocessed_composite_fusion",
}

DEFAULT_PREPROCESSING_MODE = "bone_suppression_clahe"

# Disease Classes (7 Classes)
DISEASE_CLASSES = [
    "atelectasis",
    "cavitation",
    "infiltrate",
    "lymphadenopathy",
    "pleural_effusion",
    "pneumothorax",
    "tb_lesion"
]

DISEASE_CLASS_TO_ID = {name: idx for idx, name in enumerate(DISEASE_CLASSES)}
DISEASE_ID_TO_CLASS = {idx: name for idx, name in enumerate(DISEASE_CLASSES)}

DEFAULT_CLASS_CONF_THRESHOLDS = {
    "atelectasis": 0.15,
    "cavitation": 0.20,
    "infiltrate": 0.12,
    "lymphadenopathy": 0.20,
    "pleural_effusion": 0.25,
    "pneumothorax": 0.20,
    "tb_lesion": 0.15
}

ORGAN_CLASSES = [
    "upper_left",
    "upper_right",
    "mid_left",
    "mid_right",
    "lower_left",
    "lower_right"
]

ORGAN_CLASS_TO_ID = {name: idx for idx, name in enumerate(ORGAN_CLASSES)}

YOLO26_SEG_VARIANTS = {
    "n": "yolo26n-seg.pt",
    "s": "yolo26s-seg.pt",
    "m": "yolo26m-seg.pt",
    "l": "yolo26l-seg.pt",
    "x": "yolo26x-seg.pt"
}

# Default training settings
DEFAULT_IMAGE_SIZE = 512
DEFAULT_HIGHRES_IMAGE_SIZE = 896
DEFAULT_BATCH_SIZE = 16
DEFAULT_EPOCHS = 50
DEFAULT_DEVICE = 0

DEFAULT_FOCAL_LOSS_GAMMA = 1.5
DEFAULT_MASK_RATIO = 4
