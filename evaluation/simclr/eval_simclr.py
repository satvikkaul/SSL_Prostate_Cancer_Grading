"""
Evaluate SimCLR Classifier

Generates comprehensive evaluation metrics for the SimCLR-based classifier.

Usage:
    python evaluate_simclr.py

Output:
    - Confusion matrix: ./output/simclr/confusion_matrix.png
    - Classification report: printed to console
    - Metrics saved: ./output/simclr/evaluation_metrics.txt
"""

import os
import sys
# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, cohen_kappa_score

from data.generator import DataGenerator, create_tf_dataset

# Configuration
DATASET_DIR = './dataset'
TEST_FILE = 'Test.csv'
MODEL_PATH = './output/simclr/best_simclr_classifier.keras'
OUTPUT_DIR = './output/simclr'
IMG_SIZE = (128, 128)
BATCH_SIZE = 16
CLASS_NAMES = ['NC', 'G3', 'G5', 'G4']

print("=" * 70)
print("SimCLR Classifier Evaluation")
print("=" * 70)

# Load test data
print(f"\n[1/5] Loading test data...")
test_df = pd.read_csv(os.path.join(DATASET_DIR, TEST_FILE))
test_df[CLASS_NAMES] = test_df[CLASS_NAMES].astype('float32')
print(f"✓ Test samples: {len(test_df)}")

# Setup generator
print(f"\n[2/5] Setting up data generator...")
test_generator = DataGenerator(
    data_frame=test_df,
    y=IMG_SIZE[0], x=IMG_SIZE[1], target_channels=3,
    y_cols=CLASS_NAMES,
    batch_size=BATCH_SIZE,
    path_to_img=os.path.join(DATASET_DIR, 'images'),
    shuffle=False,
    data_augmentation=False,
    mode='custom'
)
test_dataset = create_tf_dataset(test_generator)
print(f"✓ Generator ready")

# Load model
print(f"\n[3/5] Loading model...")
if not os.path.exists(MODEL_PATH):
    print(f"ERROR: Model not found at {MODEL_PATH}")
    print("Please run fine_tune_simclr.py first!")
    exit()

model = tf.keras.models.load_model(MODEL_PATH, compile=False)
print(f"✓ Model loaded")

# Generate predictions
print(f"\n[4/5] Generating predictions...")
predictions = model.predict(test_dataset, verbose=1)
y_pred = np.argmax(predictions, axis=1)
y_true = np.argmax(test_df[CLASS_NAMES].values, axis=1)
print(f"✓ Predictions complete")

# Calculate metrics
print(f"\n[5/5] Calculating metrics...")

# Classification report
print("\n" + "=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)
report = classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=3)
print(report)

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('SimCLR Classifier - Confusion Matrix')
cm_path = os.path.join(OUTPUT_DIR, 'confusion_matrix.png')
plt.savefig(cm_path, dpi=150, bbox_inches='tight')
print(f"✓ Confusion matrix saved: {cm_path}")

# AUC-ROC
print("\n" + "=" * 70)
print("AUC-ROC SCORES")
print("=" * 70)
auc_scores = {}
for i, class_name in enumerate(CLASS_NAMES):
    if len(np.unique(y_true == i)) > 1:  # Check if class exists in test set
        auc = roc_auc_score((y_true == i).astype(int), predictions[:, i])
        auc_scores[class_name] = auc
        print(f"{class_name}: {auc:.3f}")
    else:
        print(f"{class_name}: N/A (not in test set)")

# Cohen's Kappa
kappa = cohen_kappa_score(y_true, y_pred, weights='quadratic')
print(f"\nCohen's Kappa (Quadratic): {kappa:.3f}")

# ROC Curves
plt.figure(figsize=(10, 8))
for i, class_name in enumerate(CLASS_NAMES):
    if class_name in auc_scores:
        fpr, tpr, _ = roc_curve((y_true == i).astype(int), predictions[:, i])
        plt.plot(fpr, tpr, label=f'{class_name} (AUC={auc_scores[class_name]:.2f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves - SimCLR Classifier')
plt.legend()
plt.grid(True, alpha=0.3)
roc_path = os.path.join(OUTPUT_DIR, 'roc_curves.png')
plt.savefig(roc_path, dpi=150, bbox_inches='tight')
print(f"✓ ROC curves saved: {roc_path}")

# Save metrics to file
metrics_path = os.path.join(OUTPUT_DIR, 'evaluation_metrics.txt')
with open(metrics_path, 'w') as f:
    f.write("SimCLR Classifier - Evaluation Metrics\n")
    f.write("=" * 70 + "\n\n")
    f.write("CLASSIFICATION REPORT\n")
    f.write("=" * 70 + "\n")
    f.write(report + "\n")
    f.write("\nAUC-ROC SCORES\n")
    f.write("=" * 70 + "\n")
    for class_name, auc in auc_scores.items():
        f.write(f"{class_name}: {auc:.3f}\n")
    f.write(f"\nCohen's Kappa: {kappa:.3f}\n")
    f.write(f"\nConfusion Matrix:\n{cm}\n")

print(f"✓ Metrics saved: {metrics_path}")

print("\n" + "=" * 70)
print("Evaluation Complete!")
print("=" * 70)
