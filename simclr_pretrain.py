"""
SimCLR Pretraining Script

Trains the SimCLR model using contrastive learning on unlabeled histopathology images.
This script is GPU-optimized and includes:
- Progress tracking
- Automatic checkpointing
- Learning rate scheduling
- Visualization of embeddings

Usage:
    python simclr_pretrain.py

Output:
    - Pretrained encoder: ./output/simclr/encoder_weights.h5
    - Full model: ./output/simclr/simclr_model.h5
    - Training logs: ./output/simclr/training_log.csv
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Change to your GPU ID

import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from variational_autoencoder import ConvVarAutoencoder
from simclr_model import SimCLRModel, SimCLRTrainer
from simclr_augmentations import SimCLRAugmentation, create_simclr_augmentation_pair
from my_data_generator import DataGenerator

# ============================================================================
# CONFIGURATION
# ============================================================================

# Training hyperparameters
BATCH_SIZE = 64         # Reduce to 32 if OOM (Out of Memory)
EPOCHS = 30             # Standard: 100-200, Quick: 20-30, Demo: 10
LEARNING_RATE = 0.001   # SimCLR typically uses 0.001-0.003
TEMPERATURE = 0.5       # Temperature for NT-Xent loss (0.5 is standard)
WEIGHT_DECAY = 1e-6     # L2 regularization

# Architecture
IMG_SIZE = 128
PROJECTION_DIM = 128    # Embedding dimension
PROJECTION_HIDDEN = 256 # Hidden layer in projection head

# Data paths
TRAIN_CSV = "./dataset/Train.csv"
TEST_CSV = "./dataset/Test.csv"
IMG_DIR = "./dataset/images/"

# Output paths
OUTPUT_DIR = "./output/simclr"
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
LOG_FILE = os.path.join(OUTPUT_DIR, "training_log.csv")

# Create output directories
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

print("=" * 70)
print("SimCLR Contrastive Learning - Pretraining")
print("=" * 70)
print(f"Batch Size: {BATCH_SIZE}")
print(f"Epochs: {EPOCHS}")
print(f"Learning Rate: {LEARNING_RATE}")
print(f"Temperature: {TEMPERATURE}")
print(f"Projection Dim: {PROJECTION_DIM}")
print("=" * 70)

# ============================================================================
# DATA PREPARATION
# ============================================================================

print("\n[1/5] Loading Data...")
df_train = pd.read_csv(TRAIN_CSV)
df_test = pd.read_csv(TEST_CSV)
print(f"✓ Training samples: {len(df_train)}")
print(f"✓ Validation samples: {len(df_test)}")

# Create custom data generator for contrastive learning
class SimCLRDataGenerator(DataGenerator):
    """
    Modified DataGenerator that returns two augmented views of each image.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.augmenter = SimCLRAugmentation(img_size=self.y)
    
    def __getitem__(self, index):
        """Returns (view1, view2) pairs instead of (image, label)"""
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        
        view1_batch, view2_batch = [], []
        
        for idx in indexes:
            # Get base image (already normalized by get_sample)
            image, _ = self.get_sample(idx)
            
            # Create two augmented views
            view1 = self.augmenter(image)
            view2 = self.augmenter(image)
            
            view1_batch.append(view1)
            view2_batch.append(view2)
        
        return np.array(view1_batch), np.array(view2_batch)

# Initialize data generators
train_generator = SimCLRDataGenerator(
    data_frame=df_train,
    y=IMG_SIZE, x=IMG_SIZE, target_channels=3,
    batch_size=BATCH_SIZE,
    path_to_img=IMG_DIR,
    shuffle=True,
    data_augmentation=False,  # We handle augmentation in __getitem__
    vae_mode=False,
    mode='custom'
)

val_generator = SimCLRDataGenerator(
    data_frame=df_test,
    y=IMG_SIZE, x=IMG_SIZE, target_channels=3,
    batch_size=BATCH_SIZE,
    path_to_img=IMG_DIR,
    shuffle=False,
    data_augmentation=False,
    vae_mode=False,
    mode='custom'
)

print(f"✓ Training batches: {len(train_generator)}")
print(f"✓ Validation batches: {len(val_generator)}")

# ============================================================================
# MODEL CREATION
# ============================================================================

print("\n[2/5] Building SimCLR Model...")

# Build encoder (reuse existing architecture)
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

# Create VAE to get encoder architecture
my_VAE = ConvVarAutoencoder(
    (IMG_SIZE, IMG_SIZE, 3),
    encoder_conv_filters, encoder_conv_kernel_size, encoder_conv_strides,
    bottle_dim, bottle_conv_filters, bottle_conv_kernel_size, bottle_conv_strides,
    decoder_conv_t_filters, decoder_conv_t_kernel_size, decoder_conv_t_strides,
    z_dim
)
my_VAE.build(use_batch_norm=True, use_dropout=True)
encoder = my_VAE.encoder

print(f"✓ Encoder architecture loaded")
print(f"  Input: {IMG_SIZE}×{IMG_SIZE}×3")
print(f"  Output: {z_dim} features")

# Create SimCLR model
simclr_model = SimCLRModel(
    encoder=encoder,
    projection_hidden_dim=PROJECTION_HIDDEN,
    projection_output_dim=PROJECTION_DIM
)

# Test forward pass
test_input = tf.random.normal((2, IMG_SIZE, IMG_SIZE, 3))
test_output = simclr_model(test_input, training=False)
print(f"✓ SimCLR model: {test_input.shape} → {test_output.shape}")

# ============================================================================
# OPTIMIZER & TRAINER
# ============================================================================

print("\n[3/5] Setting up Training...")

# Optimizer with weight decay (AdamW)
optimizer = tf.keras.optimizers.Adam(
    learning_rate=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)

# Learning rate schedule (cosine decay)
lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=LEARNING_RATE,
    decay_steps=EPOCHS * len(train_generator),
    alpha=0.1  # Minimum LR = 0.1 * initial_LR
)
optimizer.learning_rate = lr_schedule

# Create trainer
trainer = SimCLRTrainer(
    model=simclr_model,
    optimizer=optimizer,
    temperature=TEMPERATURE
)

print(f"✓ Optimizer: Adam with cosine decay")
print(f"✓ Temperature: {TEMPERATURE}")

# ============================================================================
# TRAINING LOOP
# ============================================================================

print("\n[4/5] Training SimCLR...")
print("=" * 70)

# Training history
history = {
    'epoch': [],
    'train_loss': [],
    'val_loss': [],
    'learning_rate': []
}

best_val_loss = float('inf')

for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch+1}/{EPOCHS}")
    trainer.reset_metrics()
    
    # Training
    for batch_idx in range(len(train_generator)):
        view1, view2 = train_generator[batch_idx]
        loss = trainer.train_step(view1, view2)
        
        if batch_idx % 10 == 0:
            print(f"  Batch {batch_idx}/{len(train_generator)} - Loss: {loss.numpy():.4f}", end='\r')
    
    train_loss = trainer.train_loss_metric.result().numpy()
    
    # Validation
    for batch_idx in range(len(val_generator)):
        view1, view2 = val_generator[batch_idx]
        trainer.val_step(view1, view2)
    
    val_loss = trainer.val_loss_metric.result().numpy()
    current_lr = optimizer.learning_rate(epoch * len(train_generator)).numpy()
    
    # Log results
    print(f"  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | LR: {current_lr:.6f}")
    
    history['epoch'].append(epoch + 1)
    history['train_loss'].append(float(train_loss))
    history['val_loss'].append(float(val_loss))
    history['learning_rate'].append(float(current_lr))
    
    # Save checkpoint if best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        checkpoint_path = os.path.join(CHECKPOINT_DIR, f"best_model_epoch_{epoch+1}.h5")
        simclr_model.save_weights(checkpoint_path)
        print(f"  ✓ Saved checkpoint (best val_loss: {val_loss:.4f})")
    
    # Save periodic checkpoint
    if (epoch + 1) % 5 == 0:
        checkpoint_path = os.path.join(CHECKPOINT_DIR, f"checkpoint_epoch_{epoch+1}.h5")
        simclr_model.save_weights(checkpoint_path)

print("\n" + "=" * 70)
print("Training Complete!")
print("=" * 70)

# ============================================================================
# SAVE RESULTS
# ============================================================================

print("\n[5/5] Saving Results...")

# Save final model
final_model_path = os.path.join(OUTPUT_DIR, "simclr_final_model.h5")
simclr_model.save_weights(final_model_path)
print(f"✓ Saved final model: {final_model_path}")

# Save encoder separately (this is what we'll use for fine-tuning)
encoder_path = os.path.join(OUTPUT_DIR, "encoder_weights.h5")
encoder.save_weights(encoder_path)
print(f"✓ Saved encoder weights: {encoder_path}")

# Save training log
log_df = pd.DataFrame(history)
log_df.to_csv(LOG_FILE, index=False)
print(f"✓ Saved training log: {LOG_FILE}")

# Plot training curves
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history['epoch'], history['train_loss'], label='Train Loss')
plt.plot(history['epoch'], history['val_loss'], label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('SimCLR Training Loss')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history['epoch'], history['learning_rate'])
plt.xlabel('Epoch')
plt.ylabel('Learning Rate')
plt.title('Learning Rate Schedule')
plt.grid(True)

plot_path = os.path.join(OUTPUT_DIR, "training_curves.png")
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
print(f"✓ Saved training curves: {plot_path}")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "=" * 70)
print("TRAINING SUMMARY")
print("=" * 70)
print(f"Final Train Loss: {history['train_loss'][-1]:.4f}")
print(f"Final Val Loss: {history['val_loss'][-1]:.4f}")
print(f"Best Val Loss: {best_val_loss:.4f}")
print(f"\nSaved Files:")
print(f"  - Encoder: {encoder_path}")
print(f"  - Full Model: {final_model_path}")
print(f"  - Training Log: {LOG_FILE}")
print(f"  - Training Curves: {plot_path}")
print("=" * 70)
print("\nNext Step: Run fine_tune_simclr.py to train classifier")
print("=" * 70)
