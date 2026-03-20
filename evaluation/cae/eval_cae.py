import pandas as pd
import numpy as np
import sys
import os
# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
import tensorflow as tf
from data.generator import DataGenerator, create_tf_dataset
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, cohen_kappa_score
import matplotlib.pyplot as plt
import os

# --- CONFIGURATION ---
DATASET_DIR = './dataset'
TEST_FILE = 'Test.csv'
MODEL_PATH = './output/cae/best_cae_classifier.keras'
LEGACY_MODEL_PATH = './output/best_model_stage1.keras'
OUTPUT_DIR = './output/cae'
IMG_SIZE = (128, 128)  # MUST MATCH TRAINING! Model expects 128x128
BATCH_SIZE = 16
CLASS_NAMES = ['NC', 'G3', 'G5', 'G4']  # MUST match CSV column order!
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. Load Test Data
print(f"Loading test data from {TEST_FILE}...")
test_df = pd.read_csv(os.path.join(DATASET_DIR, TEST_FILE))
# Ensure columns are strings
test_df[CLASS_NAMES] = test_df[CLASS_NAMES].astype('float32')

# 2. Setup Generator (MUST MATCH TRAIN SETTINGS)
print("Setting up custom DataGenerator...")
test_generator = DataGenerator(
    data_frame=test_df,
    y=IMG_SIZE[0], x=IMG_SIZE[1], target_channels=3,
    y_cols=CLASS_NAMES,
    batch_size=BATCH_SIZE,
    path_to_img=os.path.join(DATASET_DIR, 'images'),
    shuffle=False,  # Vital for matching predictions to true labels
    data_augmentation=False,
    mode='custom'
)

# Wrap in tf.data pipeline (optional for inference, but ensures consistency)
test_dataset = create_tf_dataset(test_generator)

# 3. Load the Best Model
if not os.path.exists(MODEL_PATH):
    if os.path.exists(LEGACY_MODEL_PATH):
        MODEL_PATH = LEGACY_MODEL_PATH
    else:
        print(f"ERROR: Model not found at {MODEL_PATH} or {LEGACY_MODEL_PATH}")
        exit()

print(f"Loading model: {MODEL_PATH}")
# compile=False is CRITICAL here because we used a custom FocalLoss.
# We don't need to train, only predict, so we skip compiling the loss function.
model = tf.keras.models.load_model(MODEL_PATH, compile=False)

# 4. Generate Predictions
print("Running predictions (this may take a moment)...")
predictions = model.predict(test_dataset, verbose=1)

# Convert probabilities to class indices (0, 1, 2, 3)
y_pred = np.argmax(predictions, axis=1)
y_true = np.argmax(test_df[CLASS_NAMES].values, axis=1)

# 5. Generate Report
print("\n" + "="*40)
print("FINAL CLASSIFICATION REPORT")
print("="*40)
report = classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=3)
print(report)

# 6. Generate Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
plt.imshow(cm, cmap='Blues')
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, str(cm[i, j]), ha='center', va='center', color='black')
plt.xticks(range(len(CLASS_NAMES)), CLASS_NAMES)
plt.yticks(range(len(CLASS_NAMES)), CLASS_NAMES)
plt.colorbar()
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix: Self-Supervised Prostate Grading')
save_path = os.path.join(OUTPUT_DIR, 'confusion_matrix.png')
plt.savefig(save_path)
print(f"Confusion Matrix saved to {save_path}")

print("\n" + "=" * 70)
print("AUC-ROC SCORES")
print("=" * 70)
auc_scores = {}
for i, class_name in enumerate(CLASS_NAMES):
    if len(np.unique(y_true == i)) > 1:
        auc = roc_auc_score((y_true == i).astype(int), predictions[:, i])
        auc_scores[class_name] = auc
        print(f"{class_name}: {auc:.3f}")
    else:
        print(f"{class_name}: N/A (not in test set)")

kappa = cohen_kappa_score(y_true, y_pred, weights='quadratic')
print(f"\nCohen's Kappa (Quadratic): {kappa:.3f}")

plt.figure(figsize=(10, 8))
for i, class_name in enumerate(CLASS_NAMES):
    if class_name in auc_scores:
        fpr, tpr, _ = roc_curve((y_true == i).astype(int), predictions[:, i])
        plt.plot(fpr, tpr, label=f'{class_name} (AUC={auc_scores[class_name]:.2f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves - CAE Classifier')
plt.legend()
plt.grid(True, alpha=0.3)
roc_path = os.path.join(OUTPUT_DIR, 'roc_curves.png')
plt.savefig(roc_path, dpi=150, bbox_inches='tight')
print(f"ROC curves saved to {roc_path}")

metrics_path = os.path.join(OUTPUT_DIR, 'evaluation_metrics.txt')
with open(metrics_path, 'w') as f:
    f.write("CAE Classifier - Evaluation Metrics\n")
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

print(f"Metrics saved to {metrics_path}")
print("Evaluation Complete.")
