# Timika-30K YOLO-Seg Pipeline Documentation

Comprehensive technical documentation of the chest X-ray disease instance segmentation pipeline, architecture, mathematical formulations, and execution workflows for the **Timika-30K** dataset.

---

## 📑 Table of Contents
1. [Overview & Architecture](#1-overview--architecture)
2. [Dataset Preprocessing & Modality Fusion](#2-dataset-preprocessing--modality-fusion)
3. [Experiment Scenarios: Scenario 1 vs Scenario 2](#3-experiment-scenarios-scenario-1-vs-scenario-2)
4. [Training Dynamics & Loss Formulations](#4-training-dynamics--loss-formulations)
5. [Inference, Post-Processing & Visualization](#5-inference-post-processing--visualization)
6. [Evaluation Metrics & Benchmark Formulations](#6-evaluation-metrics--benchmark-formulations)
7. [CLI Reference & Quick Start](#7-cli-reference--quick-start)

---

## 1. Overview & Architecture

The pipeline processes chest X-rays (CXRs) to detect and segment **7 distinct pulmonary disease classes**:

| Class ID | Disease Name | Anatomical Characteristics |
|:---:|---|---|
| `0` | `atelectasis` | Partial or complete collapse of the lung tissue |
| `1` | `cavitation` | Thick-walled gas-filled abnormal space within lung parenchyma |
| `2` | `infiltrate` | Ill-defined density/opacity within lung tissues |
| `3` | `lymphadenopathy` | Enlargement of hilar or mediastinal lymph nodes |
| `4` | `pleural_effusion`| Fluid accumulation within the pleural space |
| `5` | `pneumothorax` | Air in the pleural cavity leading to lung collapse |
| `6` | `tb_lesion` | Tuberculous consolidations, nodules, or granulomas |

### Architectural Flowchart

```text
[ Raw DICOM / Preprocessed CXR ]
               │
               ▼
[ Preprocessing & 3-Channel Fusion ] ──► (CLAHE + Bone Suppression + Laplacian Edge)
               │
               ▼
[ YOLO Polygon Label Extraction ]   ──► (cv2.findContours -> approxPolyDP -> normalized coordinates)
               │
               ▼
[ Zero Patient-Leakage Split ]      ──► Train (22,534) | Val (1,640) | Test (6,320)
               │
               ▼
[ YOLO26-Seg High-Res Training ]    ──► (imgsz=896, cls_gain=1.0, mask_ratio=4)
               │
               ▼
[ Post-Processing & Thresholding ]  ──► (Class-specific conf + Morphological closing)
               │
               ▼
[ Evaluation & Side-by-Side Views ] ──► (Global Dice, IoU, Precision, Recall + SBS Images)
```

---

## 2. Dataset Preprocessing & Modality Fusion

Standard grayscale medical images duplicated across 3 channels fail to utilize the full representational capacity of convolutional backbones. To address this, the pipeline supports **3-Channel Composite Fusion**:

$$\text{Input Tensor } X \in \mathbb{R}^{H \times W \times 3} = [X_{\text{CLAHE}}, \, X_{\text{BoneSuppressed}}, \, X_{\text{Laplacian}}]$$

* **Channel 0 (Red) - Contrast-Limited Adaptive Histogram Equalization (CLAHE)**: Amplifies local contrast in soft lung parenchyma to make faint infiltrates visible.
* **Channel 1 (Green) - Bone-Suppressed CXR**: Soft tissue representation with rib and clavicle bone shadows removed, reducing false positives caused by bone overlap.
* **Channel 2 (Blue) - Laplacian High-Pass Edge Filter**: Computes the second spatial derivative $\Delta I = \frac{\partial^2 I}{\partial x^2} + \frac{\partial^2 I}{\partial y^2}$, providing sharp edge gradients along lesion borders and pleural margins.

---

## 3. Experiment Scenarios: Scenario 1 vs Scenario 2

To evaluate the impact of pure computer vision optimizations without confounding variables, the pipeline is divided into two separate, isolated scenarios:

| Pipeline Dimension | Scenario 1 (Baseline) | Scenario 2 (CV Optimized) |
|---|---|---|
| **Input Modality** | Single modality (`bone_suppression_clahe`) | 3-Channel Composite Fusion (`composite_fusion`) |
| **Input Resolution** | $512 \times 512$ | **$896 \times 896$** (3.06× pixel density) |
| **Mask Prototype Ratio** | Default (`mask_ratio=4`, lower prototype grid) | High-Resolution Prototype Grid |
| **Class Loss Weighting**| Standard (`cls=0.5`) | Enhanced (`cls=1.0`) |
| **Confidence Thresholding** | Fixed global threshold (`conf=0.25`) | **Class-Specific Thresholds** (`0.12` - `0.25`) |
| **Mask Post-Processing** | Raw thresholding | **OpenCV Morphological Closing** (`morph_kernel=3`) |
| **Dataset Location** | `d:\Goji\timika-30k\yolo_dataset\` | `d:\Goji\timika-30k\yolo_dataset_scenario2\` |
| **Results Location** | `results/scenario1_baseline/` | `results/scenario2_cv_optimized/` |

---

## 4. Training Dynamics & Loss Formulations

The multi-task objective function for YOLO segmentation combines bounding box coordinates, class probability, distribution focal loss, and mask prototype loss:

$$\mathcal{L}_{\text{total}} = \lambda_{\text{box}} \mathcal{L}_{\text{box}} + \lambda_{\text{cls}} \mathcal{L}_{\text{cls}} + \lambda_{\text{dfl}} \mathcal{L}_{\text{dfl}} + \lambda_{\text{mask}} \mathcal{L}_{\text{mask}}$$

### Handling Medical Class Imbalance:
1. **Elevated Classification Weight (`cls=1.0`)**: Penalizes misclassifications more heavily to counteract the extreme imbalance where >70% of images contain no lesions for specific classes.
2. **Overlap Mask Enforcement (`overlap_mask=True`)**: Preserves overlapping ground truth masks when multiple diseases (e.g., *infiltrate* + *pleural_effusion*) co-occur in the same lung zone.
3. **High Resolution Grid (`imgsz=896`)**: Preserves spatial feature maps at the P3, P4, and P5 pyramid layers, critical for small lesions (e.g. cavitations, lymphadenopathy) that vanish at 512px.

---

## 5. Inference, Post-Processing & Visualization

### Class-Specific Confidence Thresholding
Subtle, diffuse lesions require a lower activation threshold to avoid False Negatives (FN), while well-circumscribed lesions benefit from higher thresholds to suppress False Positives (FP):

```python
CLASS_CONF_THRESHOLDS = {
    "atelectasis": 0.15,
    "cavitation": 0.20,
    "infiltrate": 0.12,         # Diffuse opacity -> Lower threshold
    "lymphadenopathy": 0.20,
    "pleural_effusion": 0.25,   # Sharp fluid level -> Higher threshold
    "pneumothorax": 0.20,
    "tb_lesion": 0.15           # Fine granular nodule -> Lower threshold
}
```

### Morphological Mask Post-Processing
Raw predicted binary masks can exhibit jagged edges or pixel-level pinhole noise. We apply morphological closing using an elliptical structuring element:

$$\text{Mask}_{\text{clean}} = (\text{Mask} \oplus K) \ominus K$$

This closes internal holes and smooths the anatomical contours of the segmented lesions without altering overall lesion size.

### Side-by-Side Visualization Canvas
For every test image, `infer.py` renders:
- **Left Panel (Ground Truth)**: Ground truth polygon overlays with class-coded bounding boxes and banners.
- **Right Panel (YOLO Prediction)**: Predicted masks, bounding boxes, and model confidence scores.

---

## 6. Evaluation Metrics & Benchmark Formulations

### The "Empty Background" Metric Inflation Problem
In multi-class medical segmentation, any single X-ray typically exhibits at most 1 or 2 diseases. For the remaining 5+ disease classes, the ground-truth mask is **completely empty** ($GT = 0$).

If a naive metric gives a score of $1.0$ (100%) whenever $GT = 0$ and $\text{Pred} = 0$, the macro-average becomes **artificially inflated to >90%**, hiding the fact that actual disease lesions are being missed.

### Robust Formulations Used in this Pipeline:

1. **Global Dataset Pixel Overlap (Medical Benchmark Standard)**:
   Sums intersections and totals across the entire dataset first, completely eliminating empty mask inflation:
   $$\text{Global Dice} = \frac{2 \sum_i |GT_i \cap \text{Pred}_i|}{\sum_i |GT_i| + \sum_i |\text{Pred}_i|}$$
   $$\text{Global IoU} = \frac{\sum_i |GT_i \cap \text{Pred}_i|}{\sum_i |GT_i \cup \text{Pred}_i|}$$

2. **Positive-Only Sample Metrics (Lesion-Present Images Only)**:
   Computes metrics strictly on test images where the ground-truth disease is present ($|GT_i| > 0$):
   $$\text{Pos-Only Dice} = \frac{1}{N_{\text{pos}}} \sum_{i \in \text{pos}} \frac{2 |GT_i \cap \text{Pred}_i|}{|GT_i| + |\text{Pred}_i|}$$

---

## 7. CLI Reference & Quick Start

All scripts must be executed inside the **`qwen-vl_env`** conda environment.

### Scenario 1: Baseline Execution
```cmd
cd /d "d:\Goji\timika-30k\yolo_pipeline"

# Run Inference & Evaluation
python infer.py --weights runs/segment/runs/train/timika_yolo26s_seg/weights/best.pt --name scenario1_baseline_predictions
```

### Scenario 2: CV Optimized Execution
```cmd
cd /d "d:\Goji\timika-30k\yolo_pipeline"

# 1. Prepare 3-Channel Composite Fusion Dataset
python prepare_dataset.py --mode composite_fusion --output-dir d:\Goji\timika-30k\yolo_dataset_scenario2

# 2. Train High-Res Model (896x896)
python train.py --variant s --data d:\Goji\timika-30k\yolo_dataset_scenario2\dataset.yaml --imgsz 896 --batch 8 --workers 2 --cls-gain 1.0 --mask-ratio 4 --epochs 50 --name scenario2_cv_optimized_yolo26s

# 3. Run Inference with Post-Processing & Class Thresholds
python infer.py --weights runs/segment/runs/train/scenario2_cv_optimized_yolo26s/weights/best.pt --source d:\Goji\timika-30k\yolo_dataset_scenario2\images\test --imgsz 896 --use-class-conf --morph-kernel 3 --name scenario2_optimized_predictions
```

### Automated End-to-End Runner
To run Scenario 2 training and evaluation automatically in a single background process:
```cmd
cd /d "d:\Goji\timika-30k\yolo_pipeline"
python run_scenario2.py
```
