"""
t-SNE visualization for the MoCo v2 encoder.

Loads either:
  - encoder_q .weights.h5
  - a full-state tf.train checkpoint from training/moco/pretrain_moco.py
"""

import argparse
import os
import sys
from collections import Counter

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.manifold import TSNE

matplotlib.use("Agg")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from models.moco_model import build_encoder
from training.moco.pretrain_moco import _resolve_resume_path

OUTPUT_DIR = "./output/results/moco"
MANIFEST_PATH = "./dataset/Pretrain_Manifest.csv"
SICAP_LABEL_CSVS = [
    "./dataset/TrainSplit.csv",
    "./dataset/Val.csv",
    "./dataset/Train.csv",
    "./dataset/Test.csv",
]


def parse_args():
    parser = argparse.ArgumentParser(description="MoCo v2 t-SNE visualization")
    parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint path")
    parser.add_argument("--n_samples", type=int, default=1000, help="Max number of samples")
    parser.add_argument("--batch_size", type=int, default=32, help="Inference batch size")
    parser.add_argument("--perplexity", type=int, default=30, help="t-SNE perplexity")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output PNG path (default: output/results/moco/tsne_pilot.png)",
    )
    return parser.parse_args()


def load_image(path, size=(128, 128)):
    try:
        img = cv2.imread(path)
        if img is None:
            return None
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (size[1], size[0]))
        return img.astype(np.float32) / 255.0
    except Exception:
        return None


def extract_features(encoder, image_paths, batch_size=32):
    features = []
    valid_paths = []

    n_images = len(image_paths)
    for i in range(0, n_images, batch_size):
        batch_paths = image_paths[i : i + batch_size]
        imgs = [load_image(p) for p in batch_paths]
        valid = [(img, path) for img, path in zip(imgs, batch_paths) if img is not None]
        if not valid:
            continue

        batch_imgs, batch_paths_valid = zip(*valid)
        batch_tensor = tf.constant(np.array(batch_imgs), dtype=tf.float32)
        embeddings = encoder(batch_tensor, training=False)
        features.append(embeddings.numpy())
        valid_paths.extend(batch_paths_valid)

        if (i // batch_size + 1) % 5 == 0:
            print(f"  Extracted features: {len(valid_paths)}/{n_images}")

    if not features:
        return np.array([]), []
    return np.vstack(features), valid_paths


def build_grade_map():
    grade_cols = ["NC", "G3", "G4", "G5"]
    grade_map = {}

    for csv_path in SICAP_LABEL_CSVS:
        if not os.path.exists(csv_path):
            continue
        df = pd.read_csv(csv_path)
        if "image_name" not in df.columns:
            continue

        for _, row in df.iterrows():
            image_name = row["image_name"]
            for col in grade_cols:
                if col in row and row[col] == 1:
                    grade_map[image_name] = col
                    break
            if image_name not in grade_map:
                grade_map[image_name] = "NC"

    return grade_map


def build_color_labels(valid_paths):
    grade_map = build_grade_map()
    label_colors = {
        "NC": "#2196F3",
        "G3": "#4CAF50",
        "G4": "#FF9800",
        "G5": "#F44336",
        "PANDA": "#9C27B0",
        "UNKNOWN": "#9E9E9E",
    }

    labels = []
    colors = []
    for path in valid_paths:
        fname = os.path.basename(path)
        if fname.startswith("panda_"):
            label = "PANDA"
        else:
            label = grade_map.get(fname, "UNKNOWN")
        labels.append(label)
        colors.append(label_colors[label])

    from matplotlib.patches import Patch

    legend_handles = [
        Patch(color=label_colors[label], label=label)
        for label in sorted(set(labels))
        if label in label_colors
    ]
    return labels, colors, legend_handles


def load_encoder_from_checkpoint(checkpoint_path):
    checkpoint_path = _resolve_resume_path(checkpoint_path)
    if not checkpoint_path:
        raise FileNotFoundError("Could not resolve checkpoint path for t-SNE loading.")

    encoder = build_encoder(input_shape=(128, 128, 3))
    if checkpoint_path.endswith(".weights.h5"):
        encoder.load_weights(checkpoint_path)
        encoder.trainable = False
        return encoder

    checkpoint = tf.train.Checkpoint(encoder_q=encoder)
    checkpoint.restore(checkpoint_path).expect_partial()
    encoder.trainable = False
    return encoder


def main():
    args = parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = args.output or os.path.join(OUTPUT_DIR, "tsne_pilot.png")

    print(f"Loading encoder from: {args.checkpoint}")
    encoder = load_encoder_from_checkpoint(args.checkpoint)
    print(f"  Encoder loaded. Output dim: {encoder.output_shape}")

    manifest = pd.read_csv(MANIFEST_PATH)
    manifest["image_path"] = manifest["image_path"].apply(
        lambda p: os.path.join(".", "dataset", p.replace("\\", "/").split("/dataset/")[-1])
    )
    print(f"Manifest loaded: {len(manifest)} total images")

    n_samples = min(args.n_samples, len(manifest))
    sampled = manifest.sample(n=n_samples, random_state=42)
    image_paths = sampled["image_path"].tolist()
    print(f"Using {n_samples} samples for t-SNE")

    print("Extracting features...")
    features, valid_paths = extract_features(encoder, image_paths, batch_size=args.batch_size)
    if len(features) == 0:
        print("ERROR: No features extracted. Check image paths in manifest.")
        return

    print(f"Feature matrix shape: {features.shape}")
    print(f"Running t-SNE (perplexity={args.perplexity})...")
    embeddings_2d = TSNE(
        n_components=2,
        perplexity=args.perplexity,
        learning_rate="auto",
        init="pca",
        random_state=42,
    ).fit_transform(features)
    print("t-SNE complete.")

    labels, colors, legend_handles = build_color_labels(valid_paths)

    fig, ax = plt.subplots(figsize=(12, 10))
    ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=colors, alpha=0.6, s=12, linewidths=0)
    ax.set_title(
        f"MoCo v2 t-SNE - {len(features)} patches\nCheckpoint: {os.path.basename(args.checkpoint)}",
        fontsize=14,
        pad=15,
    )
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.legend(handles=legend_handles, loc="upper right", fontsize=10, framealpha=0.8)
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nt-SNE plot saved: {output_path}")

    dist = Counter(labels)
    print("\nLabel distribution:")
    for label, count in sorted(dist.items()):
        print(f"  {label:7s}: {count:5d} ({100 * count / len(labels):.1f}%)")


if __name__ == "__main__":
    main()
