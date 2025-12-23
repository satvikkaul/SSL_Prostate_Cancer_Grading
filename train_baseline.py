"""
Baseline Training (No SSL Pretraining)

Trains a classifier from scratch with random initialization (no SSL pretraining).
This serves as the baseline to demonstrate the benefit of SSL pretraining.

Usage:
    python train_baseline.py

Output:
    - Trained model: ./output/baseline/baseline_classifier.keras
    - Results plot: ./output/baseline/training_curves.png
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
EPOCHS = 50
LEARNING_RATE = 0.00001
IMG_DIM = (128, 128, 3)

# Paths
TRAIN_CSV = "./dataset/Train.csv"
TEST_CSV = "./dataset/Test.csv"
IMG_DIR = "./dataset/images/"

# Output
OUTPUT_DIR = "./output/baseline"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 70)
print("Baseline Training (No SSL Pretraining)")
print("=" * 70)
print(f"Batch Size: {BATCH_SIZE}")
print(f"Epochs: {EPOCHS}")
print(f"Learning Rate: {LEARNING_RATE}")
print("Note: Encoder initialized with RANDOM weights (no pretraining)")
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
# MODEL CREATION (RANDOM INITIALIZATION)
# ============================================================================

print("\n[2/4] Building Classifier with Random Initialization...")

# Architecture (same as SimCLR/Autoencoder for fair comparison)
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

# Create encoder with RANDOM weights (no pretraining)
my_VAE = ConvVarAutoencoder(
    IMG_DIM, encoder_conv_filters, encoder_conv_kernel_size, encoder_conv_strides,
    bottle_dim, bottle_conv_filters, bottle_conv_kernel_size, bottle_conv_strides,
    decoder_conv_t_filters, decoder_conv_t_kernel_size, decoder_conv_t_strides, z_dim
)
my_VAE.build(use_batch_norm=True, use_dropout=True)
encoder = my_VAE.encoder

print(f"✓ Encoder created with RANDOM initialization (no SSL pretraining)")

# Add classification head
bottleneck_output = encoder.get_layer('dropout_7').output
x = GlobalMaxPooling2D()(bottleneck_output)
x = Dense(200, activation='relu', name='dense_200')(x)
predictions = Dense(4, activation='softmax', name='classification_head')(x)

classifier = Model(inputs=encoder.input, outputs=predictions)
print(f"✓ Classifier built: {classifier.input.shape} → {classifier.output.shape}")

# Count parameters
trainable_count = sum([tf.size(w).numpy() for w in classifier.trainable_weights])
print(f"✓ Trainable parameters: {trainable_count:,}")

# ============================================================================
# CLASS WEIGHTS & LOSS
# ============================================================================

print("\n[3/4] Setting up Training...")
train_labels = np.argmax(df_train[class_columns].values, axis=1)
unique_classes = np.unique(train_labels)
class_weights_array = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=unique_classes,
    y=train_labels
)
class_weights_dict = {i: w for i, w in zip(unique_classes, class_weights_array)}
print(f"Class Weights: {class_weights_dict}")
print("Note: Using focal loss for consistency with SSL models")

# Focal loss (same as fine_tune.py and fine_tune_simclr.py)
def focal_loss(alpha=0.5, gamma=2.0):
    def loss_fn(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        ce = -y_true * tf.math.log(y_pred)
        focal_weight = tf.pow(1.0 - y_pred, gamma)
        focal_loss = alpha * focal_weight * ce
        return tf.reduce_mean(tf.reduce_sum(focal_loss, axis=-1))
    return loss_fn

# Compile model
classifier.compile(
    optimizer=SGD(learning_rate=LEARNING_RATE, momentum=0.9, clipnorm=1.0),
    loss=focal_loss(alpha=0.5, gamma=2.0),
    metrics=['accuracy']
)
print(f"✓ Optimizer: SGD with momentum=0.9, clipnorm=1.0")
print(f"✓ Loss: Focal Loss (alpha=0.5, gamma=2.0)")

# ============================================================================
# TRAINING
# ============================================================================

print("\n[4/4] Training Baseline Classifier...")
print("=" * 70)

history = classifier.fit(
    train_dataset,
    epochs=EPOCHS,
    validation_data=test_dataset,
    callbacks=[
        ModelCheckpoint(
            os.path.join(OUTPUT_DIR, 'best_baseline_classifier.keras'),
            save_best_only=True,
            monitor='val_loss',
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
    ],
    verbose=2
)

print("\n" + "=" * 70)
print("Training Complete!")
print("=" * 70)

# ============================================================================
# SAVE RESULTS
# ============================================================================

# Save final model
classifier.save(os.path.join(OUTPUT_DIR, 'baseline_classifier_final.keras'))
print(f"✓ Saved model: {OUTPUT_DIR}/baseline_classifier_final.keras")

# Plot results
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title('Baseline Classifier Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Baseline Classifier Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plot_path = os.path.join(OUTPUT_DIR, 'training_curves.png')
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
print(f"✓ Saved plot: {plot_path}")

# Save training history
history_df = pd.DataFrame(history.history)
history_df.to_csv(os.path.join(OUTPUT_DIR, 'training_history.csv'), index=False)
print(f"✓ Saved training history")

# Summary
print("\nFinal Results:")
print(f"  Train Accuracy: {history.history['accuracy'][-1]:.4f}")
print(f"  Val Accuracy: {history.history['val_accuracy'][-1]:.4f}")
print(f"  Train Loss: {history.history['loss'][-1]:.4f}")
print(f"  Val Loss: {history.history['val_loss'][-1]:.4f}")
print("\nNext Step: Run evaluate_baseline.py to generate metrics")
print("=" * 70)
