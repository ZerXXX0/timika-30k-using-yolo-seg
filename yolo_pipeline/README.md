# Timika-30K YOLO26-Seg Training Pipeline

Complete, modular Python pipeline for training **YOLO Instance Segmentation (YOLO26-Seg)** models on the Timika-30K chest X-ray dataset.

Default preprocessing modality: **`preprocessed_bone_suppression_clahe`** (Bone Suppression + CLAHE contrast enhancement).

---

## 📁 Directory Structure

```text
d:\Goji\timika-30k\yolo_pipeline\
├── config.py             # Global constants, default paths, class definitions
├── prepare_dataset.py    # Converts mask PNGs to YOLO polygon .txt & exports splits
├── train.py              # Ultralytics YOLO26-Seg model training launcher
├── infer.py              # Visual prediction & overlay script for test images
└── README.md             # Pipeline documentation & CLI examples
```

---

## ⚙️ 1. Prepare Dataset

Convert disease segmentation masks into standard Ultralytics YOLO-Seg format (`dataset.yaml` + `images/` & `labels/` folders):

```bash
# Prepare dataset using Bone Suppression + CLAHE preprocessed images (default)
python prepare_dataset.py

# Optional: Run a quick dry-run on 100 samples
python prepare_dataset.py --max-samples 100
```

### Output Layout Generated (`d:\Goji\timika-30k\yolo_dataset\`):
```text
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
```

---

## 🚀 2. Model Training (YOLO26-Seg)

### Train All Variants Consecutively (`n` -> `s` -> `m` -> `l` -> `x`):
To train **all 5 variants sequentially** in one command:

```bash
python train_all_variants.py --epochs 50 --batch 16
```

This runner automatically trains:
1. `yolo26n-seg.pt` (Nano) -> `runs/train/timika_yolo26n_seg/`
2. `yolo26s-seg.pt` (Small) -> `runs/train/timika_yolo26s_seg/`
3. `yolo26m-seg.pt` (Medium) -> `runs/train/timika_yolo26m_seg/`
4. `yolo26l-seg.pt` (Large) -> `runs/train/timika_yolo26l_seg/`
5. `yolo26x-seg.pt` (Extra-Large) -> `runs/train/timika_yolo26x_seg/`

And outputs a final side-by-side performance summary table comparing execution times and saved checkpoints.

### Train a Single Specific Variant:
```bash
# Train Nano variant (yolo26n-seg.pt)
python train.py --variant n --epochs 50 --batch 32

# Train Medium variant (yolo26m-seg.pt)
python train.py --variant m --epochs 50 --batch 16

# Train Extra-Large variant (yolo26x-seg.pt)
python train.py --variant x --epochs 50 --batch 8
```

---

## 🔍 3. Inference & Prediction

Run trained weights on test chest X-rays to visualize predicted segmentation masks and bounding boxes:

```bash
# Run inference on test split images using best saved weights
python infer.py --weights runs/train/timika_yolo26_seg/weights/best.pt --source d:/Goji/timika-30k/yolo_dataset/images/test
```

Predicted images with mask overlays will be saved to `runs/predict/timika_yolo26_predictions/`.

---

## 🏷️ Disease Class Indexing (7 Classes)

| Class ID | Disease Name |
|---|---|
| `0` | atelectasis |
| `1` | cavitation |
| `2` | infiltrate |
| `3` | lymphadenopathy |
| `4` | pleural_effusion |
| `5` | pneumothorax |
| `6` | tb_lesion |
