"""
t-SNE Visualization for MoCo v2 Encoder
=========================================
Loads a pre-trained encoder_q, extracts embeddings from the Pretrain_Manifest,
and generates a t-SNE scatter plot to visualize feature quality.

Usage:
    # After at least 1 checkpoint saved:
    python evaluation/moco/tsne_moco.py --checkpoint output/models/moco/encoder_q_epoch010.weights.h5
    
    # Limit samples for speed:
    python evaluation/moco/tsne_moco.py --checkpoint <path> --n_samples 500
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server/script use
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import tensorflow as tf
from sklearn.manifold import TSNE
import cv2

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from models.moco_model import build_encoder

OUTPUT_DIR = './output/results/moco'
MANIFEST_PATH = './dataset/Pretrain_Manifest.csv'
SICAP_TRAIN_CSV = './dataset/Train.csv'
SICAP_IMG_DIR = './dataset/images'


def parse_args():
    parser = argparse.ArgumentParser(description='MoCo v2 t-SNE Visualization')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to encoder_q .weights.h5 checkpoint file')
    parser.add_argument('--n_samples', type=int, default=1000,
                        help='Max number of samples to visualize (default: 1000)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Inference batch size (default: 32)')
    parser.add_argument('--perplexity', type=int, default=30,
                        help='t-SNE perplexity (default: 30)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output PNG path (default: output/results/moco/tsne_pilot.png)')
    return parser.parse_args()


def load_image(path, size=(128, 128)):
    """Load and normalize a single image. Returns None on failure."""
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
    """Run inference on all image_paths and return feature array."""
    features = []
    valid_paths = []
    
    n = len(image_paths)
    for i in range(0, n, batch_size):
        batch_paths = image_paths[i:i + batch_size]
        imgs = [load_image(p) for p in batch_paths]
        
        # Filter out failed loads
        valid = [(img, path) for img, path in zip(imgs, batch_paths) if img is not None]
        if not valid:
            continue
        
        batch_imgs, batch_paths_valid = zip(*valid)
        batch_tensor = tf.constant(np.array(batch_imgs), dtype=tf.float32)
        
        embeddings = encoder(batch_tensor, training=False)
        features.append(embeddings.numpy())
        valid_paths.extend(batch_paths_valid)
        
        if (i // batch_size + 1) % 5 == 0:
            print(f"  Extracted features: {len(valid_paths)}/{n}")
    
    if not features:
        return np.array([]), []
    
    return np.vstack(features), valid_paths


def build_color_labels(valid_paths, sicap_train_csv, sicap_img_dir):
    """
    Assign color labels:
      - SICAPv2 patches → colored by Gleason grade (NC/G3/G4/G5)
      - PANDA patches → separate color 'panda'
    
    Returns (labels, colormap, legend_handles)
    """
    # Build a lookup: image_name → grade for SICAPv2
    grade_map = {}
    grade_cols = ['NC', 'G3', 'G4', 'G5']
    
    if os.path.exists(sicap_train_csv):
        df = pd.read_csv(sicap_train_csv)
        for _, row in df.iterrows():
            name = row['image_name']
            for col in grade_cols:
                if col in row and row[col] == 1:
                    grade_map[name] = col
                    break
            if name not in grade_map:
                grade_map[name] = 'NC'
    
    label_colors = {
        'NC':   '#2196F3',  # Blue
        'G3':   '#4CAF50',  # Green
        'G4':   '#FF9800',  # Orange
        'G5':   '#F44336',  # Red
        'PANDA':'#9C27B0',  # Purple
    }
    
    labels = []
    colors = []
    
    for path in valid_paths:
        fname = os.path.basename(path)
        if fname.startswith('panda_'):
            labels.append('PANDA')
            colors.append(label_colors['PANDA'])
        elif fname in grade_map:
            grade = grade_map[fname]
            labels.append(grade)
            colors.append(label_colors[grade])
        else:
            labels.append('NC')
            colors.append(label_colors['NC'])
    
    # Build legend patches
    from matplotlib.patches import Patch
    unique_labels = sorted(set(labels))
    legend_handles = [
        Patch(color=label_colors[l], label=l) for l in unique_labels
        if l in label_colors
    ]
    
    return labels, colors, legend_handles


def main():
    args = parse_args()
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = args.output or os.path.join(OUTPUT_DIR, 'tsne_pilot.png')
    
    # ── Load encoder ─────────────────────────────────────────────────────────
    print(f"Loading encoder from: {args.checkpoint}")
    encoder = build_encoder(input_shape=(128, 128, 3))
    encoder.load_weights(args.checkpoint)
    encoder.trainable = False
    print(f"  Encoder loaded. Output dim: {encoder.output_shape}")
    
    # ── Load manifest ─────────────────────────────────────────────────────────
    manifest = pd.read_csv(MANIFEST_PATH)
    # Fix absolute paths for cross-OS compatibility (e.g Windows to Colab Linux)
    manifest['image_path'] = manifest['image_path'].apply(
        lambda p: os.path.join('.', 'dataset', p.replace('\\', '/').split('/dataset/')[-1])
    )
    print(f"Manifest loaded: {len(manifest)} total images")
    
    # Sample up to n_samples
    n = min(args.n_samples, len(manifest))
    sampled = manifest.sample(n=n, random_state=42)
    image_paths = sampled['image_path'].tolist()
    print(f"Using {n} samples for t-SNE")
    
    # ── Extract features ──────────────────────────────────────────────────────
    print("Extracting features...")
    features, valid_paths = extract_features(encoder, image_paths, batch_size=args.batch_size)
    
    if len(features) == 0:
        print("ERROR: No features extracted. Check image paths in manifest.")
        return
    
    print(f"Feature matrix shape: {features.shape}")
    
    # ── t-SNE ─────────────────────────────────────────────────────────────────
    print(f"Running t-SNE (perplexity={args.perplexity})...")
    tsne = TSNE(n_components=2, perplexity=args.perplexity, 
                learning_rate='auto', init='pca',
                random_state=42)
    embeddings_2d = tsne.fit_transform(features)
    print("t-SNE complete.")
    
    # ── Color labels ──────────────────────────────────────────────────────────
    labels, colors, legend_handles = build_color_labels(
        valid_paths, SICAP_TRAIN_CSV, SICAP_IMG_DIR
    )
    
    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.scatter(
        embeddings_2d[:, 0], embeddings_2d[:, 1],
        c=colors, alpha=0.6, s=12, linewidths=0
    )
    ax.set_title(f'MoCo v2 t-SNE — {len(features)} patches\n'
                 f'Checkpoint: {os.path.basename(args.checkpoint)}',
                 fontsize=14, pad=15)
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    ax.legend(handles=legend_handles, loc='upper right',
              fontsize=10, framealpha=0.8)
    ax.grid(True, alpha=0.2)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ t-SNE plot saved: {output_path}")
    
    # Print distribution summary
    from collections import Counter
    dist = Counter(labels)
    print("\nLabel distribution:")
    for k, v in sorted(dist.items()):
        print(f"  {k:6s}: {v:5d} ({100*v/len(labels):.1f}%)")


if __name__ == '__main__':
    main()
