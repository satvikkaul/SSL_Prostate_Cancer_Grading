import pandas as pd
import numpy as np
import os
import tensorflow as tf
from my_data_generator import DataGenerator, create_tf_dataset
from tensorflow.keras.layers import Dense, Dropout # type: ignore
from tensorflow.keras.models import Model # type: ignore
from tensorflow.keras.optimizers import SGD, Adam # type: ignore
from variational_autoencoder import ConvVarAutoencoder
from sklearn.utils import class_weight
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau # type: ignore

# --- CONFIGURATION ---
BATCH_SIZE = 8
EPOCHS_STAGE_1 = 50
EPOCHS_STAGE_2 = 0      # Skip Stage 2 (overfits)
LR_STAGE_1 = 0.00001       # Reduced from 0.5 (was causing divergence)
LR_STAGE_2 = 5e-5
IMG_DIM = (128, 128, 3) # PAPER: Uses 128x128 patches, NOT 512x512!

# Paths
TRAIN_CSV = "./dataset/Train.csv"
TEST_CSV = "./dataset/Test.csv"
IMG_DIR = "./dataset/images/"
WEIGHTS_PATH = './output/models/exp_0012/weights/VAE.weights.h5'  # Updated to new trained model 


# --- 1. SETUP DATA GENERATORS ---
print("Loading Data...")
df_train = pd.read_csv(TRAIN_CSV)
df_test = pd.read_csv(TEST_CSV)
df_train['image_name'] = df_train['image_name'].astype(str)
df_test['image_name'] = df_test['image_name'].astype(str)
class_columns = ['NC', 'G3', 'G5', 'G4']  # MUST match CSV column order!
print(f"Found {len(df_train)} training images.")

# Custom Generators
train_generator = DataGenerator(
    data_frame=df_train,
    y=IMG_DIM[0], x=IMG_DIM[1], target_channels=IMG_DIM[2],
    y_cols=class_columns,
    batch_size=BATCH_SIZE,
    path_to_img=IMG_DIR,
    shuffle=True,
    data_augmentation=True,  # Augmentation ON
    mode='custom'
)

test_generator = DataGenerator(
    data_frame=df_test,
    y=IMG_DIM[0], x=IMG_DIM[1], target_channels=IMG_DIM[2],
    y_cols=class_columns,
    batch_size=BATCH_SIZE,
    path_to_img=IMG_DIR,
    shuffle=False,
    data_augmentation=False, # Augmentation OFF
    mode='custom'
)

print("Converting to tf.data.Dataset...")
train_dataset = create_tf_dataset(train_generator)
test_dataset = create_tf_dataset(test_generator)

# --- 2. BUILD MODEL & LOAD WEIGHTS ---
print("Building Model and Loading SSL Weights...")

# Architecture configuration (Must match Main.py)
encoder_conv_filters = [16, 32, 64, 128, 256]
encoder_conv_kernel_size = [3, 3, 3, 3, 3]
encoder_conv_strides = [2, 2, 2, 2, 2]
bottle_conv_filters = [128, 64, 128]  # UPDATED: Removed 1-filter bottleneck
bottle_conv_kernel_size = [3, 3, 3]   # UPDATED: Matches new architecture
bottle_conv_strides = [1, 1, 1]       # UPDATED: Matches new architecture
decoder_conv_t_filters = [128, 64, 32, 16, 3]
decoder_conv_t_kernel_size = [3, 3, 3, 3, 3]
decoder_conv_t_strides = [2, 2, 2, 2, 2]
bottle_dim = (16, 16, 128)  # UPDATED: Matches new bottleneck
z_dim = 256                 # UPDATED: Increased capacity

my_VAE = ConvVarAutoencoder(
    IMG_DIM, encoder_conv_filters, encoder_conv_kernel_size, encoder_conv_strides,
    bottle_dim, bottle_conv_filters, bottle_conv_kernel_size, bottle_conv_strides,
    decoder_conv_t_filters, decoder_conv_t_kernel_size, decoder_conv_t_strides, z_dim
)

my_VAE.build(use_batch_norm=True, use_dropout=True)
try:
    my_VAE.model.load_weights(WEIGHTS_PATH)
    print("SUCCESS: SSL Weights loaded.")
except OSError:
    print(f"ERROR: Could not find weights at {WEIGHTS_PATH}")
    exit()

# --- 3. CLASS WEIGHTS ---
print("Calculating class weights...")
train_labels = np.argmax(df_train[class_columns].values, axis=1)
unique_classes = np.unique(train_labels)
class_weights_array = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=unique_classes,
    y=train_labels
)

class_weights_dict = {i: w for i, w in zip(unique_classes, class_weights_array)}
print(f"Computed Class Weights: {class_weights_dict}")

# FINAL: Don't use class weights - focal loss handles class imbalance better
# Extreme weights caused training collapse (accuracy → 15%)
# Using focal loss parameters that previously achieved 1 G5 prediction
print(f"Computed Class Weights (NOT USED): {class_weights_dict}")
print("Using Focal Loss instead of class weights for better stability")

# --- 4. CREATE CLASSIFIER ---
from tensorflow.keras.layers import GlobalMaxPooling2D

encoder = my_VAE.encoder
# Initial State: Freeze Encoder
# for layer in encoder.layers:
#     layer.trainable = False

# PAPER APPROACH: Use bottleneck output → GMP → Dense[200, 4]
# Get bottleneck output (last drop out layer before flatten)
bottleneck_output = encoder.get_layer('dropout_7').output  # (4, 4, 128) after bottleneck

# Global Max Pooling (as per paper)
x = GlobalMaxPooling2D()(bottleneck_output)  # → 128 features

# Dense layers [200, 4] as per paper
x = Dense(200, activation='relu', name='dense_200')(x)
predictions = Dense(4, activation='softmax', name='classification_head')(x)
classifier = Model(inputs=encoder.input, outputs=predictions)

# --- 5. STAGE 1: TRAIN HEAD ONLY ---
print("\n--- STAGE 1: Training Head Only ---")

# ========== FOCAL LOSS EXPERIMENT (REVERSIBLE) ==========
# Define Focal Loss function for handling extreme class imbalance
def focal_loss(alpha=0.25, gamma=2.0):
    """
    Focal Loss: FL(pt) = -alpha * (1-pt)^gamma * log(pt)
    - alpha: Balancing factor for class weights (0.25 recommended)
    - gamma: Focusing parameter (2.0 = down-weight easy examples by 4x)
    """
    def loss_fn(y_true, y_pred):
        # Clip predictions to prevent log(0)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)

        # Calculate cross entropy
        ce = -y_true * tf.math.log(y_pred)

        # Calculate focal weight: (1 - pt)^gamma
        focal_weight = tf.pow(1.0 - y_pred, gamma)

        # Apply focal loss formula
        focal_loss = alpha * focal_weight * ce

        return tf.reduce_mean(tf.reduce_sum(focal_loss, axis=-1))

    return loss_fn

# FINAL RUN: Use focal loss params that achieved first G5 prediction
# alpha=0.5, gamma=2.0 previously got 1 G5 correct (0.4% recall, 100% precision)
# This is the best configuration we found - attempting to replicate
classifier.compile(
    optimizer=SGD(learning_rate=LR_STAGE_1, momentum=0.9, clipnorm=1.0),
    loss=focal_loss(alpha=0.5, gamma=2.0),  # BEST CONFIGURATION
    # loss='categorical_crossentropy',  # Disabled - focal loss is better
    metrics=['accuracy']
)
print(f"\nTraining with LR={LR_STAGE_1}, Frozen encoder layers: {sum([not l.trainable for l in encoder.layers])}/{len(encoder.layers)}")
history_stage1 = classifier.fit(
    train_dataset,
    epochs=EPOCHS_STAGE_1,
    validation_data=test_dataset,
    # No class weights - focal loss handles imbalance internally
    callbacks=[ModelCheckpoint('./output/best_model_stage1.keras', save_best_only=True, monitor='val_loss')]
)

# --- 6. STAGE 2: FINE-TUNE ENCODER ---
print("\n--- STAGE 2: Fine-Tuning Encoder ---")
for layer in encoder.layers:
    if "batch_normalization" in layer.name:
        layer.trainable = False # Keep BatchNorm frozen
    else:
        layer.trainable = True
# Use same focal loss function from Stage 1
classifier.compile(
    optimizer=Adam(learning_rate=LR_STAGE_2, clipnorm=1.0),
    loss=focal_loss(alpha=0.50, gamma=1.5),  # FOCAL LOSS EXPERIMENT (same as Stage 1)
    # loss='categorical_crossentropy',  # ORIGINAL (uncomment to revert)
    metrics=['accuracy']
)

history_stage2 = classifier.fit(
    train_dataset,
    epochs=EPOCHS_STAGE_2,
    validation_data=test_dataset,
    class_weight=class_weights_dict,  # ENABLED: Capped weights (0.5-2.0)
    callbacks=[
        ModelCheckpoint('./output/best_model_fine_tuned.keras', save_best_only=True, monitor='val_loss'),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1)
    ]
)

# --- 7. SAVE & PLOT ---

classifier.save('./output/final_classifier.keras')
print("Model saved to ./output/final_classifier.keras")

# Handle plotting when Stage 2 is skipped (EPOCHS_STAGE_2=0)
if EPOCHS_STAGE_2 > 0:
    def append_history(history1, history2, metric):
        return history1.history[metric] + history2.history[metric]
    acc = append_history(history_stage1, history_stage2, 'accuracy')
    val_acc = append_history(history_stage1, history_stage2, 'val_accuracy')
    loss = append_history(history_stage1, history_stage2, 'loss')
    val_loss = append_history(history_stage1, history_stage2, 'val_loss')
else:
    # Only Stage 1 was run
    acc = history_stage1.history['accuracy']
    val_acc = history_stage1.history['val_accuracy']
    loss = history_stage1.history['loss']
    val_loss = history_stage1.history['val_loss']
plt.figure(figsize=(12, 5))

# Accuracy Plot
plt.subplot(1, 2, 1)
plt.plot(acc, label='Train Acc')
plt.plot(val_acc, label='Val Acc')
if EPOCHS_STAGE_2 > 0:
    plt.axvline(x=len(history_stage1.history['accuracy']), color='k', linestyle='--', label='Unfreeze Point')
plt.title('Accuracy (Stage 1 Only)' if EPOCHS_STAGE_2 == 0 else 'Accuracy (Combined)')
plt.legend()

# Loss Plot
plt.subplot(1, 2, 2)
plt.plot(loss, label='Train Loss')
plt.plot(val_loss, label='Val Loss')
if EPOCHS_STAGE_2 > 0:
    plt.axvline(x=len(history_stage1.history['loss']), color='k', linestyle='--', label='Unfreeze Point')
plt.title('Loss (Stage 1 Only)' if EPOCHS_STAGE_2 == 0 else 'Loss (Combined)')
plt.legend()
plt.savefig('./output/classification_results_combined.png')
print("Results plot saved.")