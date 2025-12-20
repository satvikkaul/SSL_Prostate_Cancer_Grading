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

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import class_weight

from variational_autoencoder import ConvVarAutoencoder
from my_data_generator import DataGenerator, create_tf_dataset
from tensorflow.keras.layers import Dense, Dropout, GlobalMaxPooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

# ============================================================================
# CONFIGURATION
# ============================================================================

BATCH_SIZE = 8
EPOCHS_STAGE_1 = 50      # Train head only
EPOCHS_STAGE_2 = 0       # Fine-tune encoder (set to 0 to skip)
LR_STAGE_1 = 0.00001
LR_STAGE_2 = 5e-5
IMG_DIM = (128, 128, 3)

# Paths
TRAIN_CSV = "./dataset/Train.csv"
TEST_CSV = "./dataset/Test.csv"
IMG_DIR = "./dataset/images/"
ENCODER_WEIGHTS = './output/simclr/encoder_weights.h5'  # SimCLR pretrained weights

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
print("=" * 70)

# ============================================================================
# DATA PREPARATION
# ============================================================================

print("\n[1/4] Loading Data...")
df_train = pd.read_csv(TRAIN_CSV)
df_test = pd.read_csv(TEST_CSV)
df_train['image_name'] = df_train['image_name'].astype(str)
df_test['image_name'] = df_test['image_name'].astype(str)
class_columns = ['NC', 'G3', 'G5', 'G4']
print(f"✓ Training samples: {len(df_train)}")
print(f"✓ Test samples: {len(df_test)}")

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

test_generator = DataGenerator(
    data_frame=df_test,
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
test_dataset = create_tf_dataset(test_generator)

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
print(f"Class Weights: {class_weights_dict}")
print("Note: Using focal loss instead for better stability")

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
    validation_data=test_dataset,
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
        validation_data=test_dataset,
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
