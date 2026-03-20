"""
Evaluate the MoCo fine-tuned classifier on the held-out SICAP test split.
"""

import argparse
import os
import sys

os.environ.setdefault("MPLCONFIGDIR", os.path.join("/tmp", "mplconfig"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report, cohen_kappa_score, confusion_matrix, roc_auc_score, roc_curve

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from data.generator import DataGenerator, create_tf_dataset


DATASET_DIR = "./dataset"
TEST_FILE = "Test.csv"
DEFAULT_MODEL_CANDIDATES = [
    "./output/moco/best_moco_overall.keras",
    "./output/moco/best_moco_fine_tuned.keras",
    "./output/moco/best_moco_classifier.keras",
]
OUTPUT_DIR = "./output/moco"
IMG_SIZE = (128, 128)
BATCH_SIZE = 16
CLASS_NAMES = ["NC", "G3", "G5", "G4"]


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a MoCo classifier checkpoint")
    parser.add_argument("--model_path", type=str, default=None, help="Explicit model path to evaluate")
    return parser.parse_args()


def resolve_model_path(explicit_path=None):
    if explicit_path:
        return explicit_path
    for candidate in DEFAULT_MODEL_CANDIDATES:
        if os.path.exists(candidate):
            return candidate
    return DEFAULT_MODEL_CANDIDATES[0]


def main():
    args = parse_args()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model_path = resolve_model_path(args.model_path)

    print("=" * 70)
    print("MoCo Classifier Evaluation")
    print("=" * 70)

    test_df = pd.read_csv(os.path.join(DATASET_DIR, TEST_FILE))
    test_df[CLASS_NAMES] = test_df[CLASS_NAMES].astype("float32")
    print(f"Test samples: {len(test_df)}")

    test_generator = DataGenerator(
        data_frame=test_df,
        y=IMG_SIZE[0],
        x=IMG_SIZE[1],
        target_channels=3,
        y_cols=CLASS_NAMES,
        batch_size=BATCH_SIZE,
        path_to_img=os.path.join(DATASET_DIR, "images"),
        shuffle=False,
        data_augmentation=False,
        mode="custom",
    )
    test_dataset = create_tf_dataset(test_generator)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"MoCo classifier not found at {model_path}")

    print(f"Using model: {model_path}")
    model = tf.keras.models.load_model(model_path, compile=False)
    predictions = model.predict(test_dataset, verbose=1)
    y_pred = np.argmax(predictions, axis=1)
    y_true = np.argmax(test_df[CLASS_NAMES].values, axis=1)

    print("\n" + "=" * 70)
    print("CLASSIFICATION REPORT")
    print("=" * 70)
    report = classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=3, zero_division=0)
    print(report)

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, cmap="Greens")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
    plt.xticks(range(len(CLASS_NAMES)), CLASS_NAMES)
    plt.yticks(range(len(CLASS_NAMES)), CLASS_NAMES)
    plt.colorbar()
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("MoCo Classifier - Confusion Matrix")
    cm_path = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=150, bbox_inches="tight")

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

    kappa = cohen_kappa_score(y_true, y_pred, weights="quadratic")
    print(f"\nCohen's Kappa (Quadratic): {kappa:.3f}")

    plt.figure(figsize=(10, 8))
    for i, class_name in enumerate(CLASS_NAMES):
        if class_name in auc_scores:
            fpr, tpr, _ = roc_curve((y_true == i).astype(int), predictions[:, i])
            plt.plot(fpr, tpr, label=f"{class_name} (AUC={auc_scores[class_name]:.2f})")
    plt.plot([0, 1], [0, 1], "k--", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves - MoCo Classifier")
    plt.legend()
    plt.grid(True, alpha=0.3)
    roc_path = os.path.join(OUTPUT_DIR, "roc_curves.png")
    plt.savefig(roc_path, dpi=150, bbox_inches="tight")

    metrics_path = os.path.join(OUTPUT_DIR, "evaluation_metrics.txt")
    with open(metrics_path, "w") as f:
        f.write("MoCo Classifier - Evaluation Metrics\n")
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

    print(f"Saved confusion matrix: {cm_path}")
    print(f"Saved ROC curves: {roc_path}")
    print(f"Saved metrics: {metrics_path}")


if __name__ == "__main__":
    main()
