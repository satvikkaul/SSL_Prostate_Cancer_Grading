import pandas as pd
import numpy as np
import sys
import os
# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
import tensorflow as tf
from data.generator import DataGenerator, create_tf_dataset
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- CONFIGURATION ---
DATASET_DIR = './dataset'
TEST_FILE = 'Test.csv'
# IMPORTANT: Use Stage 1 model (frozen encoder + conv features)
MODEL_PATH = './output/best_model_stage1.keras'  # Stage 1 best checkpoint
IMG_SIZE = (128, 128)  # MUST MATCH TRAINING! Model expects 128x128
BATCH_SIZE = 16
CLASS_NAMES = ['NC', 'G3', 'G5', 'G4']  # MUST match CSV column order!

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
    print(f"ERROR: Model not found at {MODEL_PATH}")
    print("Please rename your best model file to match this path.")
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
print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))

# 6. Generate Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix: Self-Supervised Prostate Grading')
save_path = './output/final_confusion_matrix.png'
plt.savefig(save_path)
print(f"Confusion Matrix saved to {save_path}")
print("Evaluation Complete.")