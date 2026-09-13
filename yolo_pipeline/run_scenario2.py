"""
Automated Pipeline Runner for Scenario 2 (CV Optimizations).
Executes High-Resolution Training with Enhanced Class Loss and Mask Prototype Ratio,
followed by Side-by-Side Visualization and Pixel-Level Metrics Evaluation.
"""

import sys
import subprocess
from pathlib import Path

print("==========================================================")
print("       TIMIKA-30K SCENARIO 2 AUTOMATED PIPELINE         ")
print("==========================================================")

# Step 1: Train Scenario 2 Model
train_cmd = [
    sys.executable, "train.py",
    "--variant", "s",
    "--data", r"d:\Goji\timika-30k\yolo_dataset_scenario2\dataset.yaml",
    "--imgsz", "896",
    "--batch", "8",
    "--workers", "2",
    "--cls-gain", "1.0",
    "--mask-ratio", "4",
    "--epochs", "50",
    "--name", "scenario2_cv_optimized_yolo26s"
]

print(f"\n[Step 1/2] Launching Training: {' '.join(train_cmd)}\n")
res = subprocess.run(train_cmd)

if res.returncode != 0:
    print(f"\n[ERROR] Training exited with code {res.returncode}. Aborting inference.")
    sys.exit(res.returncode)

# Step 2: Locate Trained Weights
candidate_weights = list(Path("runs").rglob("scenario2_cv_optimized_yolo26s/weights/best.pt"))
if candidate_weights:
    weights_path = candidate_weights[0]
else:
    weights_path = Path("runs/train/scenario2_cv_optimized_yolo26s/weights/best.pt")

print(f"\n[Step 2/2] Training completed successfully. Located best checkpoint: {weights_path}")

# Step 3: Run Inference, Side-by-Side Generation, and Metric Evaluation
infer_cmd = [
    sys.executable, "infer.py",
    "--weights", str(weights_path),
    "--source", r"d:\Goji\timika-30k\yolo_dataset_scenario2\images\test",
    "--imgsz", "896",
    "--use-class-conf",
    "--morph-kernel", "3",
    "--name", "scenario2_optimized_predictions"
]

print(f"Launching Inference & Evaluation: {' '.join(infer_cmd)}\n")
subprocess.run(infer_cmd)

print("\n==========================================================")
print("     SCENARIO 2 PIPELINE FULLY COMPLETED!                ")
print("==========================================================")
