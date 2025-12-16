import pandas as pd
import numpy as np
import os
import tensorflow as tf
from my_data_generator import DataGenerator
from tensorflow.keras.layers import Dense, Dropout, Flatten, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD, Adam
from variational_autoencoder import ConvVarAutoencoder
from sklearn.utils import class_weight
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ModelCheckpoint

# --- CONFIGURATION ---
BATCH_SIZE = 32 
EPOCHS_STAGE_1 = 10     # Warm-up epochs
EPOCHS_STAGE_2 = 20     # Fine-tuning epochs
LR_STAGE_1 = 1e-3       # Higher LR for new head
LR_STAGE_2 = 1e-5       # Very low LR for fine-tuning
IMG_DIM = (512, 512, 3)

# Paths
TRAIN_CSV = "./dataset/Train.csv"
TEST_CSV = "./dataset/Test.csv"
IMG_DIR = "./dataset/images/"
WEIGHTS_PATH = './output/models/exp_0007/weights/VAE.weights.h5' 

# --- 1. SETUP DATA GENERATORS ---
print("Loading Data...")
df_train = pd.read_csv(TRAIN_CSV)
df_test = pd.read_csv(TEST_CSV)
df_train['image_name'] = df_train['image_name'].astype(str)
df_test['image_name'] = df_test['image_name'].astype(str)
class_columns = ['NC', 'G3', 'G4', 'G5']
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

# --- 2. BUILD MODEL & LOAD WEIGHTS ---
print("Building Model and Loading SSL Weights...")

# Architecture configuration (Must match Main.py)
encoder_conv_filters = [16, 32, 64, 128, 256]
encoder_conv_kernel_size = [3, 3, 3, 3, 3]
encoder_conv_strides = [2, 2, 2, 2, 2]
bottle_conv_filters = [64, 32, 1, 256]
bottle_conv_kernel_size = [3, 3, 3, 3]
bottle_conv_strides = [1, 1, 1, 1]
decoder_conv_t_filters = [128, 64, 32, 16, 3]
decoder_conv_t_kernel_size = [3, 3, 3, 3, 3]
decoder_conv_t_strides = [2, 2, 2, 2, 2]
bottle_dim = (32, 32, 256)
z_dim = 200

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
print(f"Class Weights: {class_weights_dict}")
# --- 4. CREATE CLASSIFIER ---
encoder = my_VAE.encoder
# Initial State: Freeze Encoder
for layer in encoder.layers:
    layer.trainable = False
x = encoder.output
x = Dropout(0.5)(x)
predictions = Dense(4, activation='softmax', name='classification_head')(x)
classifier = Model(inputs=encoder.input, outputs=predictions)

# --- 5. STAGE 1: TRAIN HEAD ONLY ---
print("\n--- STAGE 1: Training Head Only ---")
classifier.compile(optimizer=Adam(learning_rate=LR_STAGE_1), 
                   loss='categorical_crossentropy', 
                   metrics=['accuracy'])

history_stage1 = classifier.fit(
    train_generator,
    epochs=EPOCHS_STAGE_1,
    validation_data=test_generator,
    class_weight=class_weights_dict,
    callbacks=[ModelCheckpoint('./output/best_model_stage1.keras', save_best_only=True, monitor='val_loss')]
)

# --- 6. STAGE 2: FINE-TUNE ENCODER ---
print("\n--- STAGE 2: Fine-Tuning Encoder ---")
for layer in encoder.layers:
    if "batch_normalization" in layer.name:
        layer.trainable = False # Keep BatchNorm frozen
    else:
        layer.trainable = True
classifier.compile(optimizer=Adam(learning_rate=LR_STAGE_2), 
                   loss='categorical_crossentropy', 
                   metrics=['accuracy'])

history_stage2 = classifier.fit(
    train_generator,
    epochs=EPOCHS_STAGE_2,
    validation_data=test_generator,
    class_weight=class_weights_dict,
    callbacks=[ModelCheckpoint('./output/best_model_fine_tuned.keras', save_best_only=True, monitor='val_loss')]
)

# --- 7. SAVE & PLOT ---

classifier.save('./output/final_classifier.keras')
print("Model saved to ./output/final_classifier.keras")

def append_history(history1, history2, metric):
    return history1.history[metric] + history2.history[metric]

acc = append_history(history_stage1, history_stage2, 'accuracy')
val_acc = append_history(history_stage1, history_stage2, 'val_accuracy')
loss = append_history(history_stage1, history_stage2, 'loss')
val_loss = append_history(history_stage1, history_stage2, 'val_loss')
plt.figure(figsize=(12, 5))

# Accuracy Plot
plt.subplot(1, 2, 1)
plt.plot(acc, label='Train Acc')
plt.plot(val_acc, label='Val Acc')
plt.axvline(x=len(history_stage1.history['accuracy']), color='k', linestyle='--', label='Unfreeze Point')
plt.title('Accuracy')
plt.legend()

# Loss Plot
plt.subplot(1, 2, 2)
plt.plot(loss, label='Train Loss')
plt.plot(val_loss, label='Val Loss')
plt.axvline(x=len(history_stage1.history['loss']), color='k', linestyle='--', label='Unfreeze Point')
plt.title('Loss')
plt.legend()
plt.savefig('./output/classification_results_combined.png')
print("Results plot saved.")