import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- CONFIGURATION ---
DATASET_DIR = './dataset'
TEST_FILE = 'Test.csv'
# IMPORTANT: We use the STABLE frozen model, not the unstable new one
MODEL_PATH = './output/final_classifier.keras' 
IMG_SIZE = (512, 512)
BATCH_SIZE = 16
CLASS_NAMES = ['NC', 'G3', 'G4', 'G5']

# 1. Load Test Data
print(f"Loading test data from {TEST_FILE}...")
test_df = pd.read_csv(os.path.join(DATASET_DIR, TEST_FILE))
# Ensure columns are strings
test_df[CLASS_NAMES] = test_df[CLASS_NAMES].astype('float32')

# 2. Setup Generator
test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_dataframe(
    dataframe=test_df,
    directory=os.path.join(DATASET_DIR, 'images'), # Adjust path if needed
    x_col="image_name",
    y_col=CLASS_NAMES,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="raw",
    shuffle=False # vital for confusion matrix!
)

# 3. Load the Best Model
if not os.path.exists(MODEL_PATH):
    print(f"ERROR: Model not found at {MODEL_PATH}")
    print("Please rename your best model file to match this path.")
    exit()

print(f"Loading model: {MODEL_PATH}")
model = tf.keras.models.load_model(MODEL_PATH)

# 4. Generate Predictions
print("Running predictions (this may take a moment)...")
predictions = model.predict(test_generator, verbose=1)

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