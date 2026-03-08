"""
MoCo v2 Pre-training Script
============================
Usage:
    # Full run
    python training/moco/pretrain_moco.py

    # Pilot run (10 epochs, quick test)
    python training/moco/pretrain_moco.py --epochs 10 --batch_size 16

Hardware target: RTX 3060 Laptop (6GB VRAM)
Config: ResNet50, K=4096, AMP=ON, batch=16 (fallback to 8 on OOM)
"""

import os
import sys
import argparse
import time
import numpy as np
import pandas as pd
import tensorflow as tf

# Ensure root is in path when running as a script
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from models.moco_model import MoCoV2Model, get_moco_augmenter
from data.generator import MoCoDataGenerator

# ─── Mixed Precision (AMP) ───────────────────────────────────────────────────
tf.keras.mixed_precision.set_global_policy('mixed_float16')

# ─── Paths ───────────────────────────────────────────────────────────────────
MANIFEST_PATH = './dataset/Pretrain_Manifest.csv'
CHECKPOINT_DIR = './output/models/moco'
LOG_DIR = './output/results/moco/logs'


def parse_args():
    parser = argparse.ArgumentParser(description='MoCo v2 Pre-training')
    parser.add_argument('--epochs', type=int, default=200,
                        help='Number of training epochs (default: 200)')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size (default: 16, fallback to 8 on OOM)')
    parser.add_argument('--queue_size', type=int, default=4096,
                        help='Negative queue size K (default: 4096)')
    parser.add_argument('--momentum', type=float, default=0.999,
                        help='Momentum coefficient m (default: 0.999)')
    parser.add_argument('--temperature', type=float, default=0.2,
                        help='InfoNCE temperature tau (default: 0.2)')
    parser.add_argument('--lr', type=float, default=0.03,
                        help='Base learning rate (default: 0.03)')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='SGD weight decay (default: 1e-4)')
    parser.add_argument('--save_freq', type=int, default=10,
                        help='Save checkpoint every N epochs (default: 10)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    return parser.parse_args()


def build_optimizer(lr, weight_decay):
    """SGD with momentum (Keras 3 handles AMP scaling via global policy)."""
    return tf.keras.optimizers.SGD(
        learning_rate=lr, momentum=0.9, weight_decay=weight_decay
    )


def cosine_lr_schedule(base_lr, epoch, total_epochs, warmup_epochs=5):
    """Cosine annealing LR schedule with linear warmup."""
    if epoch < warmup_epochs:
        return base_lr * (epoch + 1) / warmup_epochs
    progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
    return base_lr * 0.5 * (1.0 + np.cos(np.pi * progress))


def train_step(model, optimizer, augmenter_q, augmenter_k, images):
    """
    Single MoCo v2 training step.
    AMP is handled automatically by the mixed_float16 global policy;
    no manual loss scaling needed in Keras 3.
    """
    with tf.GradientTape() as tape:
        # Augment on GPU
        xq = augmenter_q(images, training=True)
        xk = augmenter_k(images, training=True)
        
        # Query forward pass (trainable)
        q = model.encoder_q(xq, training=True)
        
        # Key forward pass (momentum encoder — no gradient)
        k = model.encoder_k(xk, training=False)
        k = tf.stop_gradient(k)
        
        # Compute InfoNCE loss (casts to float32 internally)
        loss = model.info_nce_loss(q, k)
        # Cast back to compute-dtype for AMP compatibility
        loss = tf.cast(loss, tf.float32)
    
    # Backprop only through encoder_q
    grads = tape.gradient(loss, model.encoder_q.trainable_variables)
    
    # Gradient clipping for stability (ResNet50 from random init)
    grads, _ = tf.clip_by_global_norm(grads, clip_norm=1.0)
    
    optimizer.apply_gradients(zip(grads, model.encoder_q.trainable_variables))
    
    # Update queue: enqueue current keys, dequeue oldest
    model.queue.dequeue_and_enqueue(k)
    
    # Momentum update: θ_k ← m·θ_k + (1−m)·θ_q
    model.momentum_update()
    
    return loss


def run_training(args, batch_size):
    """Main training loop. Raises ResourceExhaustedError on OOM."""
    # ── Load manifest ────────────────────────────────────────────────────────
    if not os.path.exists(MANIFEST_PATH):
        raise FileNotFoundError(
            f"Pretrain_Manifest.csv not found at {MANIFEST_PATH}.\n"
            "Run: python data/setup.py"
        )
    
    manifest = pd.read_csv(MANIFEST_PATH)
    # Fix absolute paths for cross-OS compatibility (e.g Windows to Colab Linux)
    manifest['image_path'] = manifest['image_path'].apply(
        lambda p: os.path.join('.', 'dataset', p.replace('\\', '/').split('/dataset/')[-1])
    )
    print(f"Loaded manifest: {len(manifest)} images")
    
    # ── Data generator ───────────────────────────────────────────────────────
    generator = MoCoDataGenerator(
        manifest, image_size=(128, 128), batch_size=batch_size, shuffle=True
    )
    print(f"Generator: {len(generator)} batches/epoch @ batch_size={batch_size}")
    
    # ── Model ────────────────────────────────────────────────────────────────
    model = MoCoV2Model(
        input_shape=(128, 128, 3),
        queue_size=args.queue_size,
        momentum=args.momentum,
        temperature=args.temperature
    )
    print(f"MoCo v2 model built. Encoder params: {model.encoder_q.count_params():,}")
    
    # ── Augmentation ─────────────────────────────────────────────────────────
    # Create two separate instances of the augmenters for independent randomness
    augmenter_q = get_moco_augmenter((128, 128))
    augmenter_k = get_moco_augmenter((128, 128))
    
    # ── Optimizer ────────────────────────────────────────────────────────────
    optimizer = build_optimizer(args.lr, args.weight_decay)
    
    # ── Resume from checkpoint ───────────────────────────────────────────────
    start_epoch = 0
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    
    if args.resume and os.path.exists(args.resume):
        model.encoder_q.load_weights(args.resume)
        print(f"Resumed encoder_q from: {args.resume}")
    
    # ── Training Loop ────────────────────────────────────────────────────────
    log_file = os.path.join(LOG_DIR, 'pretrain_log.csv')
    log_rows = []
    
    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()
        
        # Update learning rate
        lr = cosine_lr_schedule(args.lr, epoch, args.epochs)
        optimizer.learning_rate.assign(lr)
        
        epoch_losses = []
        
        step = 0
        for images_t in generator:
            loss = train_step(model, optimizer, augmenter_q, augmenter_k, images_t)
            epoch_losses.append(float(loss))
            
            if step % 10 == 0:
                print(f"  Epoch {epoch+1}/{args.epochs} | Step {step}/{len(generator)} "
                      f"| Loss: {float(loss):.4f} | LR: {lr:.6f}")
            step += 1
        
        generator.on_epoch_end()
        
        avg_loss = np.mean(epoch_losses)
        epoch_time = time.time() - epoch_start
        
        print(f"Epoch {epoch+1}/{args.epochs} | Avg Loss: {avg_loss:.4f} "
              f"| Time: {epoch_time:.1f}s | LR: {lr:.6f}")
        
        log_rows.append({
            'epoch': epoch + 1,
            'avg_loss': avg_loss,
            'lr': lr,
            'time_s': epoch_time
        })
        
        # Save checkpoint
        if (epoch + 1) % args.save_freq == 0 or epoch == args.epochs - 1:
            ckpt_path = os.path.join(CHECKPOINT_DIR, f'encoder_q_epoch{epoch+1:03d}.weights.h5')
            model.encoder_q.save_weights(ckpt_path)
            print(f"  ✓ Checkpoint saved: {ckpt_path}")
    
    # Save training log
    log_df = pd.DataFrame(log_rows)
    log_df.to_csv(log_file, index=False)
    print(f"\nTraining log saved to: {log_file}")
    
    return model


def main():
    args = parse_args()
    
    print("=" * 60)
    print("MoCo v2 Pre-training | RTX 3060 6GB | AMP=ON")
    print(f"  Epochs:      {args.epochs}")
    print(f"  Batch size:  {args.batch_size}")
    print(f"  Queue K:     {args.queue_size}")
    print(f"  Momentum m:  {args.momentum}")
    print(f"  Temperature: {args.temperature}")
    print(f"  Base LR:     {args.lr}")
    print("=" * 60)
    
    # OOM auto-fallback
    batch_size = args.batch_size
    try:
        model = run_training(args, batch_size)
    except tf.errors.ResourceExhaustedError as e:
        print(f"\n⚠ OOM with batch_size={batch_size}. Retrying with batch_size={batch_size//2}...")
        tf.keras.backend.clear_session()
        batch_size = batch_size // 2
        model = run_training(args, batch_size)
    
    print("\n✓ Pre-training complete!")
    return model


if __name__ == '__main__':
    main()
