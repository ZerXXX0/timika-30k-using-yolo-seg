# Experimental Results & Evaluation Metrics

This folder contains training curves, loss histories, confusion matrices, and pixel-level test evaluation metrics for the Timika-30K chest X-ray segmentation experiments.

---

## 📁 Directory Structure

```text
results/
├── scenario1_baseline/
│   ├── results.csv                        # Per-epoch training & validation losses / metrics
│   ├── args.yaml                          # Training configuration arguments
│   └── labels.jpg                         # Dataset label distribution plot
│
└── scenario2_cv_optimized/
    ├── eval_metrics.txt                   # Complete pixel-level Dice, IoU, Precision, Recall report
    ├── eval_metrics.csv                   # Tabular evaluation metrics data
    ├── results.csv                        # 50-epoch loss & validation metrics history
    ├── results.png                        # Multi-panel training & validation loss curves
    ├── confusion_matrix.png               # Raw confusion matrix
    ├── confusion_matrix_normalized.png    # Normalized confusion matrix
    ├── BoxF1_curve.png                    # Bounding Box F1-Confidence curve
    ├── BoxPR_curve.png                    # Bounding Box Precision-Recall curve
    ├── MaskF1_curve.png                   # Segmentation Mask F1-Confidence curve
    ├── MaskPR_curve.png                   # Segmentation Mask Precision-Recall curve
    ├── labels.jpg                         # Dataset label distribution
    ├── args.yaml                          # Model hyperparameters (896x896, cls=1.0, mask_ratio=4)
    └── sample_predictions/                # Top representative Ground Truth vs Prediction visualizations
```

---

## 📊 Scenario 2 Evaluation Summary (6,320 Test Images)

| Class ID | Disease Name | Pos Samples | Global Dice | Global IoU | Precision | Recall | Pos-Only Dice |
|---|---|---|---|---|---|---|---|
| `0` | atelectasis | 60 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| `1` | cavitation | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| `2` | **infiltrate** | 482 | **0.3098** | **0.1833** | **0.4128** | **0.2479** | 0.1722 |
| `3` | lymphadenopathy | 2 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| `4` | pleural_effusion | 270 | 0.0156 | 0.0078 | 0.3752 | 0.0079 | 0.0020 |
| `5` | **pneumothorax** | 37 | **0.1105** | **0.0585** | 0.0641 | **0.3990** | **0.3105** |
| `6` | **tb_lesion** | 81 | **0.2195** | **0.1233** | **0.1329** | **0.6294** | **0.4760** |
| **-** | **OVERALL MEAN** | **936** | **0.0936** | **0.0533** | **0.1407** | **0.1835** | **0.1372** |
