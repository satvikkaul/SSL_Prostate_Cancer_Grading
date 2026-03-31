"""
Fine-tuning SimCLR Encoder for Classification

Loads the pretrained SimCLR encoder and trains a classifier for Gleason grading.
Uses the same two-stage approach as fine_tune.py but with SimCLR weights.

Usage:
    python fine_tune_simclr.py

Output:
    - Trained classifier: ./output/simclr/simclr_classifier.keras
    - Results plot: ./output/simclr/classification_results.png
"""

import argparse
import glob
import os
import sys
# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import class_weight

from models.cae_model import ConvVarAutoencoder
from data.generator import DataGenerator, create_tf_dataset
from tensorflow.keras.layers import Dense, Dropout, GlobalMaxPooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune SimCLR encoder for classification")
    parser.add_argument(
        '--epochs_stage1',
        type=int,
        default=50,
        help='Stage 1 epochs (frozen encoder)',
    )
    parser.add_argument(
        '--epochs_stage2',
        type=int,
        default=0,
        help='Stage 2 epochs (unfrozen encoder)',
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=8,
        help='Batch size',
    )
    parser.add_argument(
        '--encoder_weights',
        type=str,
        default='./output/simclr/encoder_weights.h5',
        help='Path to SimCLR encoder weights',
    )
    parser.add_argument(
        '--weights',
        dest='encoder_weights',
        type=str,
        help='Alias for --encoder_weights',
    )
    return parser.parse_args()


args = parse_args()


def resolve_encoder_weights_path(requested_path):
    candidates = []

    if requested_path:
        if requested_path.endswith('.h5') and not requested_path.endswith('.weights.h5'):
            candidates.append(requested_path[:-3] + '.weights.h5')
        candidates.append(requested_path)

    candidates.extend([
        './output/simclr/encoder_weights.weights.h5',
        './output/simclr/encoder_weights.h5',
    ])

    seen = set()
    deduped_candidates = []
    for candidate in candidates:
        if candidate and candidate not in seen:
            deduped_candidates.append(candidate)
            seen.add(candidate)

    for candidate in deduped_candidates:
        if candidate and os.path.exists(candidate):
            return candidate

    wildcard_candidates = sorted(glob.glob('./output/simclr/encoder_weights*.h5'))
    if wildcard_candidates:
        weights_suffix = [p for p in wildcard_candidates if p.endswith('.weights.h5')]
        return weights_suffix[0] if weights_suffix else wildcard_candidates[0]

    return requested_path


def print_device_configuration():
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    gpus = tf.config.list_physical_devices('GPU')
    if visible_devices:
        print(f"CUDA_VISIBLE_DEVICES preset to: {visible_devices}")
    else:
        print("CUDA_VISIBLE_DEVICES not set; using TensorFlow default device selection.")
    print(f"Detected GPUs: {len(gpus)}")

# ============================================================================
# CONFIGURATION
# ============================================================================

BATCH_SIZE = args.batch_size
EPOCHS_STAGE_1 = args.epochs_stage1      # Train head only
EPOCHS_STAGE_2 = args.epochs_stage2       # Fine-tune encoder (set to 0 to skip)
LR_STAGE_1 = 0.00001
LR_STAGE_2 = 5e-5
IMG_DIM = (128, 128, 3)

# Paths
TRAIN_CSV = "./dataset/TrainSplit.csv"
VAL_CSV = "./dataset/Val.csv"
IMG_DIR = "./dataset/images/"
ENCODER_WEIGHTS = resolve_encoder_weights_path(args.encoder_weights)  # SimCLR pretrained weights

# Output
OUTPUT_DIR = "./output/simclr"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 70)
print("SimCLR Fine-Tuning for Gleason Grading")
print("=" * 70)
print(f"Encoder Weights: {ENCODER_WEIGHTS}")
print(f"Batch Size: {BATCH_SIZE}")
print(f"Stage 1 Epochs: {EPOCHS_STAGE_1}")
print(f"Stage 2 Epochs: {EPOCHS_STAGE_2}")
print_device_configuration()
print("=" * 70)

# ============================================================================
# DATA PREPARATION
# ============================================================================

print("\n[1/4] Loading Data...")
df_train = pd.read_csv(TRAIN_CSV)
df_val = pd.read_csv(VAL_CSV)
df_train['image_name'] = df_train['image_name'].astype(str)
df_val['image_name'] = df_val['image_name'].astype(str)
class_columns = ['NC', 'G3', 'G5', 'G4']
print(f"✓ Training samples: {len(df_train)}")
print(f"✓ Validation samples: {len(df_val)}")

# Create data generators
train_generator = DataGenerator(
    data_frame=df_train,
    y=IMG_DIM[0], x=IMG_DIM[1], target_channels=IMG_DIM[2],
    y_cols=class_columns,
    batch_size=BATCH_SIZE,
    path_to_img=IMG_DIR,
    shuffle=True,
    data_augmentation=True,
    mode='custom'
)

val_generator = DataGenerator(
    data_frame=df_val,
    y=IMG_DIM[0], x=IMG_DIM[1], target_channels=IMG_DIM[2],
    y_cols=class_columns,
    batch_size=BATCH_SIZE,
    path_to_img=IMG_DIR,
    shuffle=False,
    data_augmentation=False,
    mode='custom'
)

# Convert to tf.data
train_dataset = create_tf_dataset(train_generator)
val_dataset = create_tf_dataset(val_generator)

print(f"✓ Data generators ready")

# ============================================================================
# MODEL CREATION
# ============================================================================

print("\n[2/4] Building Classifier...")

# Architecture (must match SimCLR pretraining)
encoder_conv_filters = [16, 32, 64, 128, 256]
encoder_conv_kernel_size = [3, 3, 3, 3, 3]
encoder_conv_strides = [2, 2, 2, 2, 2]
bottle_conv_filters = [128, 64, 128]
bottle_conv_kernel_size = [3, 3, 3]
bottle_conv_strides = [1, 1, 1]
decoder_conv_t_filters = [128, 64, 32, 16, 3]
decoder_conv_t_kernel_size = [3, 3, 3, 3, 3]
decoder_conv_t_strides = [2, 2, 2, 2, 2]
bottle_dim = (16, 16, 128)
z_dim = 256

# Create encoder
my_VAE = ConvVarAutoencoder(
    IMG_DIM, encoder_conv_filters, encoder_conv_kernel_size, encoder_conv_strides,
    bottle_dim, bottle_conv_filters, bottle_conv_kernel_size, bottle_conv_strides,
    decoder_conv_t_filters, decoder_conv_t_kernel_size, decoder_conv_t_strides, z_dim
)
my_VAE.build(use_batch_norm=True, use_dropout=True)

# Load SimCLR pretrained weights
try:
    my_VAE.encoder.load_weights(ENCODER_WEIGHTS)
    print(f"✓ Loaded SimCLR encoder weights")
except Exception as e:
    print(f"ERROR: Could not load weights from {ENCODER_WEIGHTS}")
    print(f"Error: {e}")
    print("\nPlease run simclr_pretrain.py first!")
    exit()

encoder = my_VAE.encoder

# Add classification head
bottleneck_output = encoder.get_layer('dropout_7').output
x = GlobalMaxPooling2D()(bottleneck_output)
x = Dense(200, activation='relu', name='dense_200')(x)
predictions = Dense(4, activation='softmax', name='classification_head')(x)

classifier = Model(inputs=encoder.input, outputs=predictions)
print(f"✓ Classifier built: {classifier.input.shape} → {classifier.output.shape}")

# ============================================================================
# CLASS WEIGHTS
# ============================================================================

print("\n[3/4] Calculating Class Weights...")
train_labels = np.argmax(df_train[class_columns].values, axis=1)
unique_classes = np.unique(train_labels)
class_weights_array = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=unique_classes,
    y=train_labels
)
class_weights_dict = {i: w for i, w in zip(unique_classes, class_weights_array)}
# Phase 3: Cap weights at 3.0 to prevent training collapse, combine with focal loss
class_weights_dict = {k: min(v, 3.0) for k, v in class_weights_dict.items()}
print(f"Class weights (capped at 3.0): {class_weights_dict}")

# Focal loss
def focal_loss(alpha=0.5, gamma=2.0):
    def loss_fn(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        ce = -y_true * tf.math.log(y_pred)
        focal_weight = tf.pow(1.0 - y_pred, gamma)
        focal_loss = alpha * focal_weight * ce
        return tf.reduce_mean(tf.reduce_sum(focal_loss, axis=-1))
    return loss_fn

# ============================================================================
# TRAINING
# ============================================================================

print("\n[4/4] Training Classifier...")
print("=" * 70)

# Stage 1: Train head only (frozen encoder)
print("\n--- Stage 1: Training Classification Head ---")
for layer in encoder.layers:
    layer.trainable = False

trainable_count = sum([tf.size(w).numpy() for w in classifier.trainable_weights])
print(f"Trainable parameters: {trainable_count:,}")

classifier.compile(
    optimizer=SGD(learning_rate=LR_STAGE_1, momentum=0.9, clipnorm=1.0),
    loss=focal_loss(alpha=0.5, gamma=2.0),
    metrics=['accuracy']
)

history_stage1 = classifier.fit(
    train_dataset,
    epochs=EPOCHS_STAGE_1,
    validation_data=val_dataset,
    class_weight=class_weights_dict,  # Phase 3: combined with focal loss
    callbacks=[
        ModelCheckpoint(
            os.path.join(OUTPUT_DIR, 'best_simclr_classifier.keras'),
            save_best_only=True,
            monitor='val_loss',
            verbose=1
        )
    ],
    verbose=2
)

# Stage 2: Fine-tune encoder (optional)
if EPOCHS_STAGE_2 > 0:
    print("\n--- Stage 2: Fine-Tuning Encoder ---")
    for layer in encoder.layers:
        if "batch_normalization" not in layer.name:
            layer.trainable = True
    
    trainable_count = sum([tf.size(w).numpy() for w in classifier.trainable_weights])
    print(f"Trainable parameters: {trainable_count:,}")
    
    classifier.compile(
        optimizer=Adam(learning_rate=LR_STAGE_2, clipnorm=1.0),
        loss=focal_loss(alpha=0.5, gamma=1.5),
        metrics=['accuracy']
    )
    
    history_stage2 = classifier.fit(
        train_dataset,
        epochs=EPOCHS_STAGE_2,
        validation_data=val_dataset,
        class_weight=class_weights_dict,  # Phase 3: combined with focal loss
        callbacks=[
            ModelCheckpoint(
                os.path.join(OUTPUT_DIR, 'best_simclr_fine_tuned.keras'),
                save_best_only=True,
                monitor='val_loss'
            ),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)
        ],
        verbose=2
    )

# ============================================================================
# SAVE RESULTS
# ============================================================================

print("\n" + "=" * 70)
print("Training Complete!")
print("=" * 70)

# Save final model
classifier.save(os.path.join(OUTPUT_DIR, 'simclr_classifier_final.keras'))
print(f"✓ Saved model: {OUTPUT_DIR}/simclr_classifier_final.keras")

# Combine histories
if EPOCHS_STAGE_2 > 0:
    acc = history_stage1.history['accuracy'] + history_stage2.history['accuracy']
    val_acc = history_stage1.history['val_accuracy'] + history_stage2.history['val_accuracy']
    loss = history_stage1.history['loss'] + history_stage2.history['loss']
    val_loss = history_stage1.history['val_loss'] + history_stage2.history['val_loss']
else:
    acc = history_stage1.history['accuracy']
    val_acc = history_stage1.history['val_accuracy']
    loss = history_stage1.history['loss']
    val_loss = history_stage1.history['val_loss']

# Plot results
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(acc, label='Train Acc')
plt.plot(val_acc, label='Val Acc')
if EPOCHS_STAGE_2 > 0:
    plt.axvline(x=EPOCHS_STAGE_1, color='k', linestyle='--', label='Unfreeze')
plt.title('SimCLR Classifier Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(loss, label='Train Loss')
plt.plot(val_loss, label='Val Loss')
if EPOCHS_STAGE_2 > 0:
    plt.axvline(x=EPOCHS_STAGE_1, color='k', linestyle='--', label='Unfreeze')
plt.title('SimCLR Classifier Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plot_path = os.path.join(OUTPUT_DIR, 'classification_results.png')
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
print(f"✓ Saved plot: {plot_path}")

# Summary
print("\nFinal Results:")
print(f"  Train Accuracy: {acc[-1]:.4f}")
print(f"  Val Accuracy: {val_acc[-1]:.4f}")
print(f"  Train Loss: {loss[-1]:.4f}")
print(f"  Val Loss: {val_loss[-1]:.4f}")
print("\nNext Step: Run evaluate_simclr.py to generate metrics")
print("=" * 70)
