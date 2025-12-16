import pandas as pd
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Dropout, Flatten, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD, Adam
from variational_autoencoder import ConvVarAutoencoder

# --- CONFIGURATION ---
BATCH_SIZE = 16 
EPOCHS = 20     # Train longer for classification
LR = 0.00005      # Lower learning rate for fine-tuning
IMG_DIM = (512, 512, 3)

# Paths (Match your structure)
TRAIN_CSV = "./dataset/Train.csv"
TEST_CSV = "./dataset/Test.csv"
IMG_DIR = "./dataset/images/"
WEIGHTS_PATH = './output/models/exp_0007/weights/VAE.weights.h5' 

# --- 1. SETUP DATA GENERATORS ---
print("Loading Data...")
df_train = pd.read_csv(TRAIN_CSV)
df_test = pd.read_csv(TEST_CSV)

# Ensure filenames match what flow_from_dataframe expects
# (Sometimes filenames in CSV need full paths or extensions)
df_train['image_name'] = df_train['image_name'].astype(str)
df_test['image_name'] = df_test['image_name'].astype(str)

# Define the columns that represent our classes
class_columns = ['NC', 'G3', 'G4', 'G5']

# Create Data Generators (Standard Keras)
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

print(f"Found {len(df_train)} training images.")
train_generator = train_datagen.flow_from_dataframe(
    dataframe=df_train,
    directory=IMG_DIR,
    x_col="image_name",
    y_col=class_columns,
    target_size=(IMG_DIM[0], IMG_DIM[1]),
    batch_size=BATCH_SIZE,
    class_mode="raw", # Use 'raw' because we have multiple columns for one-hot
    shuffle=True
)

test_generator = test_datagen.flow_from_dataframe(
    dataframe=df_test,
    directory=IMG_DIR,
    x_col="image_name",
    y_col=class_columns,
    target_size=(IMG_DIM[0], IMG_DIM[1]),
    batch_size=BATCH_SIZE,
    class_mode="raw",
    shuffle=False
)

# --- 2. BUILD MODEL & LOAD WEIGHTS ---
print("Building Model and Loading SSL Weights...")

# Initialize the architecture (Must match Main.py exactly)
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

# Build and Load
my_VAE.build(use_batch_norm=True, use_dropout=True)
try:
    my_VAE.model.load_weights(WEIGHTS_PATH)
    print("SUCCESS: SSL Weights loaded.")
except OSError:
    print(f"ERROR: Could not find weights at {WEIGHTS_PATH}")
    print("Please update the WEIGHTS_PATH variable in the script.")
    exit()

# --- 3. CREATE CLASSIFIER (FINE-TUNING) ---
# Extract just the encoder part (input -> latent space)
encoder = my_VAE.encoder

# === DEBUG: LIST ALL LAYERS ===
print("Index | Layer Name")
print("-" * 30)
for i, layer in enumerate(encoder.layers):
    print(f"{i:4d}  | {layer.name}")
print("-" * 30)
# ==============================
print(f"Total layers in encoder: {len(encoder.layers)}")

# We define the CUTOFF point based on the layer list.
# Index 30 corresponds to 'bottle_conv2', the start of the deep layers.
CUTOFF_INDEX = 30 

print(f"Strategy: Freezing layers 0-{CUTOFF_INDEX-1}. Unfreezing layers {CUTOFF_INDEX}+ (Deep Blocks)...")

for i, layer in enumerate(encoder.layers):
    if i < CUTOFF_INDEX:
        # Layers 0 to 29 (Early vision) -> LOCKED
        layer.trainable = False
    else:
        # Layers 30 to 39 (Deep Bottleneck + Output) -> UNLOCKED
        layer.trainable = True
        print(f"   -> Unlocked: [{i}] {layer.name}")

# ==========================================

# Add classification head
x = encoder.output
# The encoder output is already a Dense vector (z_dim=200)
# We just need to map 200 features -> 4 classes
x = Dropout(0.5)(x) # Regularization
predictions = Dense(4, activation='softmax', name='classification_head')(x)

# Create final model
classifier = Model(inputs=encoder.input, outputs=predictions)

# Compile (Paper uses SGD, but Adam is often easier for quick results)
classifier.compile(optimizer=Adam(learning_rate=LR), 
                   loss='categorical_crossentropy', 
                   metrics=['accuracy'])

classifier.summary()

# --- 4. TRAIN ---
print("Starting Fine-Tuning...")
history = classifier.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=test_generator
)

# --- 5. SAVE RESULTS ---
classifier.save('./output/final_classifier.keras')
print("Model saved to ./output/final_classifier.keras")

# Plot accuracy
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title('Accuracy')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss')
plt.savefig('./output/classification_results.png')
print("Results plot saved.")