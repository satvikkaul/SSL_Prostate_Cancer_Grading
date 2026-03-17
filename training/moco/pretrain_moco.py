"""
MoCo v2 pre-training script.

Usage:
    python training/moco/pretrain_moco.py
    python training/moco/pretrain_moco.py --epochs 10 --batch_size 16
"""

import argparse
import os
import re
import sys
import time

import numpy as np
import pandas as pd
import tensorflow as tf

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from data.generator import MoCoDataGenerator
from models.moco_model import MoCoV2Model, get_moco_augmenter

MANIFEST_PATH = "./dataset/Pretrain_Manifest.csv"
CHECKPOINT_DIR = "./output/models/moco"
STATE_DIR = os.path.join(CHECKPOINT_DIR, "state")
LOG_DIR = "./output/results/moco/logs"


def configure_precision_policy():
    """
    Configure mixed precision based on accelerator availability.

    Override with SSL_MIXED_PRECISION:
      - auto (default)
      - mixed_float16 / true / 1 / on
      - float32 / false / 0 / off
    """
    override = os.environ.get("SSL_MIXED_PRECISION", "auto").strip().lower()
    gpus = tf.config.list_physical_devices("GPU")

    if override in {"mixed_float16", "true", "1", "on"}:
        policy = "mixed_float16"
    elif override in {"float32", "false", "0", "off"}:
        policy = "float32"
    else:
        policy = "mixed_float16" if gpus else "float32"

    tf.keras.mixed_precision.set_global_policy(policy)
    return policy, len(gpus)


def parse_args():
    parser = argparse.ArgumentParser(description="MoCo v2 pre-training")
    parser.add_argument("--epochs", type=int, default=200, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--queue_size", type=int, default=4096, help="Negative queue size K")
    parser.add_argument("--momentum", type=float, default=0.999, help="Momentum coefficient m")
    parser.add_argument("--temperature", type=float, default=0.2, help="InfoNCE temperature tau")
    parser.add_argument("--lr", type=float, default=0.03, help="Base learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="SGD weight decay")
    parser.add_argument("--save_freq", type=int, default=10, help="Save checkpoint every N epochs")
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to a full-state checkpoint or legacy encoder_q .weights.h5 file",
    )
    return parser.parse_args()


def build_optimizer(lr, weight_decay):
    return tf.keras.optimizers.SGD(learning_rate=lr, momentum=0.9, weight_decay=weight_decay)


def cosine_lr_schedule(base_lr, epoch, total_epochs, warmup_epochs=5):
    if epoch < warmup_epochs:
        return base_lr * (epoch + 1) / warmup_epochs
    progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
    return base_lr * 0.5 * (1.0 + np.cos(np.pi * progress))


def _normalize_manifest_paths(manifest_df):
    manifest_df["image_path"] = manifest_df["image_path"].apply(
        lambda p: os.path.join(".", "dataset", p.replace("\\", "/").split("/dataset/")[-1])
    )
    return manifest_df


def _build_optimizer_slots(optimizer, variables):
    if hasattr(optimizer, "build"):
        optimizer.build(variables)


def _make_checkpoint(model, optimizer):
    return tf.train.Checkpoint(
        encoder_q=model.encoder_q,
        encoder_k=model.encoder_k,
        optimizer=optimizer,
        queue=model.queue.get_queue(),
        queue_ptr=model.queue.get_ptr(),
    )


def _extract_epoch_from_path(path):
    basename = os.path.basename(path.rstrip("\\/"))
    match = re.search(r"epoch(\d+)", basename)
    if match:
        return int(match.group(1))
    match = re.search(r"-(\d+)$", basename)
    if match:
        return int(match.group(1))
    return 0


def _resolve_resume_path(resume_path):
    if os.path.isdir(resume_path):
        latest = tf.train.latest_checkpoint(resume_path)
        if latest:
            return latest
    return resume_path


def restore_training_state(model, optimizer, resume_path):
    resume_path = _resolve_resume_path(resume_path)
    checkpoint_exists = resume_path and (
        tf.io.gfile.exists(resume_path)
        or tf.io.gfile.exists(resume_path + ".index")
    )
    if not checkpoint_exists:
        raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")

    start_epoch = _extract_epoch_from_path(resume_path)

    if resume_path.endswith(".weights.h5"):
        model.encoder_q.load_weights(resume_path)
        model.sync_key_encoder()
        print(f"Resumed legacy encoder_q weights from: {resume_path}")
        print("Warning: queue and optimizer state were not restored from legacy weights.")
        return start_epoch

    _build_optimizer_slots(optimizer, model.encoder_q.trainable_variables)
    checkpoint = _make_checkpoint(model, optimizer)
    checkpoint.restore(resume_path)
    print(f"Resumed full MoCo state from: {resume_path}")
    return start_epoch


def save_training_state(model, optimizer, manager, epoch):
    state_path = manager.save(checkpoint_number=epoch + 1)
    encoder_path = os.path.join(CHECKPOINT_DIR, f"encoder_q_epoch{epoch + 1:03d}.weights.h5")
    model.encoder_q.save_weights(encoder_path)
    print(f"  Saved state checkpoint: {state_path}")
    print(f"  Saved encoder_q weights: {encoder_path}")


def train_step(model, optimizer, augmenter_q, augmenter_k, images):
    with tf.GradientTape() as tape:
        xq = augmenter_q(images, training=True)
        xk = augmenter_k(images, training=True)

        q = model.encoder_q(xq, training=True)
        k = model.encoder_k(xk, training=False)
        k = tf.stop_gradient(k)

        loss = tf.cast(model.info_nce_loss(q, k), tf.float32)

    grads = tape.gradient(loss, model.encoder_q.trainable_variables)
    grads, _ = tf.clip_by_global_norm(grads, clip_norm=1.0)
    optimizer.apply_gradients(zip(grads, model.encoder_q.trainable_variables))

    model.queue.dequeue_and_enqueue(k)
    model.momentum_update()
    return loss


def run_training(args, batch_size):
    if not os.path.exists(MANIFEST_PATH):
        raise FileNotFoundError(
            f"Pretrain_Manifest.csv not found at {MANIFEST_PATH}.\nRun: python data/setup.py"
        )

    manifest = _normalize_manifest_paths(pd.read_csv(MANIFEST_PATH))
    print(f"Loaded manifest: {len(manifest)} images")

    generator = MoCoDataGenerator(manifest, image_size=(128, 128), batch_size=batch_size, shuffle=True)
    print(f"Generator: {len(generator)} batches/epoch @ batch_size={batch_size}")

    model = MoCoV2Model(
        input_shape=(128, 128, 3),
        queue_size=args.queue_size,
        momentum=args.momentum,
        temperature=args.temperature,
    )
    print(f"MoCo v2 model built. Encoder params: {model.encoder_q.count_params():,}")

    augmenter_q = get_moco_augmenter((128, 128))
    augmenter_k = get_moco_augmenter((128, 128))
    optimizer = build_optimizer(args.lr, args.weight_decay)

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(STATE_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    start_epoch = 0
    checkpoint = _make_checkpoint(model, optimizer)
    manager = tf.train.CheckpointManager(
        checkpoint,
        directory=STATE_DIR,
        max_to_keep=5,
        checkpoint_name="moco_state",
    )

    log_file = os.path.join(LOG_DIR, "pretrain_log.csv")
    log_rows = []

    if args.resume:
        start_epoch = restore_training_state(model, optimizer, args.resume)
        if os.path.exists(log_file):
            log_rows = pd.read_csv(log_file).to_dict("records")
        print(f"Training will resume from epoch {start_epoch + 1}.")

    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()
        lr = cosine_lr_schedule(args.lr, epoch, args.epochs)
        optimizer.learning_rate.assign(lr)

        epoch_losses = []
        for step, images_t in enumerate(generator):
            loss = train_step(model, optimizer, augmenter_q, augmenter_k, images_t)
            epoch_losses.append(float(loss))

            if step % 10 == 0:
                print(
                    f"  Epoch {epoch + 1}/{args.epochs} | Step {step}/{len(generator)} "
                    f"| Loss: {float(loss):.4f} | LR: {lr:.6f}"
                )

        generator.on_epoch_end()

        avg_loss = float(np.mean(epoch_losses))
        epoch_time = time.time() - epoch_start
        print(
            f"Epoch {epoch + 1}/{args.epochs} | Avg Loss: {avg_loss:.4f} "
            f"| Time: {epoch_time:.1f}s | LR: {lr:.6f}"
        )

        log_rows.append(
            {
                "epoch": epoch + 1,
                "avg_loss": avg_loss,
                "lr": lr,
                "time_s": epoch_time,
            }
        )

        if (epoch + 1) % args.save_freq == 0 or epoch == args.epochs - 1:
            save_training_state(model, optimizer, manager, epoch)

        pd.DataFrame(log_rows).to_csv(log_file, index=False)

    print(f"\nTraining log saved to: {log_file}")
    return model


def main():
    args = parse_args()
    precision_policy, gpu_count = configure_precision_policy()

    print("=" * 60)
    print("MoCo v2 Pre-training")
    print(f"  Epochs:      {args.epochs}")
    print(f"  Batch size:  {args.batch_size}")
    print(f"  Queue K:     {args.queue_size}")
    print(f"  Momentum m:  {args.momentum}")
    print(f"  Temperature: {args.temperature}")
    print(f"  Base LR:     {args.lr}")
    print(f"  GPUs:        {gpu_count}")
    print(f"  Precision:   {precision_policy}")
    if args.resume:
        print(f"  Resume:      {args.resume}")
    print("=" * 60)

    batch_size = args.batch_size
    try:
        model = run_training(args, batch_size)
    except tf.errors.ResourceExhaustedError:
        print(f"\nOOM with batch_size={batch_size}. Retrying with batch_size={batch_size // 2}...")
        tf.keras.backend.clear_session()
        batch_size = batch_size // 2
        model = run_training(args, batch_size)

    print("\nPre-training complete.")
    return model


if __name__ == "__main__":
    main()
