"""
Fine-tune a pretrained MoCo v2 encoder for Gleason grading.

Usage:
    ./venv/bin/python training/moco/finetune_moco.py
    ./venv/bin/python training/moco/finetune_moco.py --checkpoint output/models/moco/encoder_q_epoch010.weights.h5
"""

import argparse
import glob
import os
import sys
import json

os.environ.setdefault("MPLCONFIGDIR", os.path.join("/tmp", "mplconfig"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.utils import class_weight
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, SGD

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from data.generator import DataGenerator, create_tf_dataset
from models.moco_model import build_encoder
from training.moco.pretrain_moco import _resolve_resume_path


BATCH_SIZE = 64        # Increased for A100 GPU (was 8)
EPOCHS_STAGE_1 = 50    # Stage 1: head-only training (was 30)
EPOCHS_STAGE_2 = 20    # Stage 2: full encoder unfreeze (was 10)
LR_STAGE_1 = 1e-4
LR_STAGE_2 = 5e-5
IMG_DIM = (128, 128, 3)
CLASS_COLUMNS = ["NC", "G3", "G5", "G4"]
TRAIN_CSV = "./dataset/TrainSplit.csv"
VAL_CSV = "./dataset/Val.csv"
IMG_DIR = "./dataset/images"
OUTPUT_DIR = "./output/moco"
DEFAULT_CHECKPOINT_GLOB = "./output/models/moco/encoder_q_epoch*.weights.h5"
BEST_STAGE1_PATH = os.path.join(OUTPUT_DIR, "best_moco_classifier.keras")
BEST_STAGE2_PATH = os.path.join(OUTPUT_DIR, "best_moco_fine_tuned.keras")
BEST_OVERALL_PATH = os.path.join(OUTPUT_DIR, "best_moco_overall.keras")
HEAD_LAYER_NAMES = {"moco_cls_dropout", "moco_cls_dense", "moco_classification_head"}


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune MoCo encoder for classification")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to encoder_q .weights.h5 or full-state MoCo checkpoint",
    )
    return parser.parse_args()


def focal_loss(alpha=0.5, gamma=2.0):
    def loss_fn(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        ce = -y_true * tf.math.log(y_pred)
        focal_weight = tf.pow(1.0 - y_pred, gamma)
        return tf.reduce_mean(tf.reduce_sum(alpha * focal_weight * ce, axis=-1))

    return loss_fn


def latest_moco_encoder_weights():
    candidates = sorted(glob.glob(DEFAULT_CHECKPOINT_GLOB))
    return candidates[-1] if candidates else None


def load_moco_encoder_weights(encoder, checkpoint_path):
    checkpoint_path = _resolve_resume_path(checkpoint_path)
    if not checkpoint_path:
        raise FileNotFoundError("No MoCo checkpoint path resolved.")

    if checkpoint_path.endswith(".weights.h5"):
        encoder.load_weights(checkpoint_path)
        return checkpoint_path

    checkpoint = tf.train.Checkpoint(encoder_q=encoder)
    checkpoint.restore(checkpoint_path).expect_partial()
    return checkpoint_path


def print_device_configuration():
    gpus = tf.config.list_physical_devices("GPU")
    print(f"Detected GPUs: {len(gpus)}")
    if not gpus:
        tf.config.run_functions_eagerly(True)


def build_classifier(encoder):
    feature_output = encoder.get_layer("proj_relu").output
    x = Dropout(0.2, name="moco_cls_dropout")(feature_output)
    x = Dense(256, activation="relu", name="moco_cls_dense")(x)
    predictions = Dense(len(CLASS_COLUMNS), activation="softmax", name="moco_classification_head")(x)
    return Model(inputs=encoder.input, outputs=predictions, name="moco_classifier")


def set_stage1_trainability(classifier):
    for layer in classifier.layers:
        layer.trainable = layer.name in HEAD_LAYER_NAMES


def set_stage2_trainability(classifier):
    for layer in classifier.layers:
        if layer.name in HEAD_LAYER_NAMES:
            layer.trainable = True
        elif "bn" in layer.name.lower():
            layer.trainable = False
        else:
            layer.trainable = True


def main():
    args = parse_args()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    checkpoint_path = args.checkpoint or latest_moco_encoder_weights()
    if not checkpoint_path:
        raise FileNotFoundError(
            "No MoCo checkpoint found. Run training/moco/pretrain_moco.py first or pass --checkpoint."
        )

    print("=" * 70)
    print("MoCo Fine-Tuning for Gleason Grading")
    print("=" * 70)
    print(f"Checkpoint: {checkpoint_path}")
    print_device_configuration()
    print("=" * 70)

    df_train = pd.read_csv(TRAIN_CSV)
    df_val = pd.read_csv(VAL_CSV)
    df_train["image_name"] = df_train["image_name"].astype(str)
    df_val["image_name"] = df_val["image_name"].astype(str)
    print(f"Training samples: {len(df_train)}")
    print(f"Validation samples: {len(df_val)}")

    train_generator = DataGenerator(
        data_frame=df_train,
        y=IMG_DIM[0],
        x=IMG_DIM[1],
        target_channels=IMG_DIM[2],
        y_cols=CLASS_COLUMNS,
        batch_size=BATCH_SIZE,
        path_to_img=IMG_DIR,
        shuffle=True,
        data_augmentation=True,
        mode="custom",
    )
    val_generator = DataGenerator(
        data_frame=df_val,
        y=IMG_DIM[0],
        x=IMG_DIM[1],
        target_channels=IMG_DIM[2],
        y_cols=CLASS_COLUMNS,
        batch_size=BATCH_SIZE,
        path_to_img=IMG_DIR,
        shuffle=False,
        data_augmentation=False,
        mode="custom",
    )

    train_dataset = create_tf_dataset(train_generator)
    val_dataset = create_tf_dataset(val_generator)

    encoder = build_encoder(input_shape=IMG_DIM, proj_dim=128, hidden_dim=2048)
    resolved_checkpoint = load_moco_encoder_weights(encoder, checkpoint_path)
    print(f"Loaded MoCo weights from: {resolved_checkpoint}")

    classifier = build_classifier(encoder)

    train_labels = np.argmax(df_train[CLASS_COLUMNS].values, axis=1)
    unique_classes = np.unique(train_labels)
    class_weights_array = class_weight.compute_class_weight(
        class_weight="balanced",
        classes=unique_classes,
        y=train_labels,
    )
    class_weights_dict = {i: w for i, w in zip(unique_classes, class_weights_array)}
    print(f"Class weights (reference only): {class_weights_dict}")

    print("\n--- Stage 1: Train Classification Head ---")
    set_stage1_trainability(classifier)

    classifier.compile(
        optimizer=SGD(learning_rate=LR_STAGE_1, momentum=0.9, clipnorm=1.0),
        loss=focal_loss(alpha=0.5, gamma=2.0),
        metrics=["accuracy"],
    )

    history_stage1 = classifier.fit(
        train_dataset,
        epochs=EPOCHS_STAGE_1,
        validation_data=val_dataset,
        callbacks=[
            ModelCheckpoint(
                BEST_STAGE1_PATH,
                save_best_only=True,
                monitor="val_loss",
                verbose=1,
            )
        ],
        verbose=2,
    )

    history_stage2 = None
    stage1_best_val = min(history_stage1.history["val_loss"])
    stage2_best_val = None
    if EPOCHS_STAGE_2 > 0:
        print("\n--- Stage 2: Fine-Tune Encoder ---")
        if not os.path.exists(BEST_STAGE1_PATH):
            raise FileNotFoundError(f"Best Stage 1 checkpoint not found at {BEST_STAGE1_PATH}")

        print(f"Reloading best Stage 1 checkpoint: {BEST_STAGE1_PATH}")
        classifier = tf.keras.models.load_model(BEST_STAGE1_PATH, compile=False)
        set_stage2_trainability(classifier)

        classifier.compile(
            optimizer=Adam(learning_rate=LR_STAGE_2, clipnorm=1.0),
            loss=focal_loss(alpha=0.5, gamma=1.5),
            metrics=["accuracy"],
        )

        history_stage2 = classifier.fit(
            train_dataset,
            epochs=EPOCHS_STAGE_2,
            validation_data=val_dataset,
            callbacks=[
                ModelCheckpoint(
                    BEST_STAGE2_PATH,
                    save_best_only=True,
                    monitor="val_loss",
                    verbose=1,
                ),
                ReduceLROnPlateau(
                    monitor="val_loss",
                    factor=0.5,
                    patience=3,
                    min_lr=1e-6,
                    verbose=1,
                ),
            ],
            verbose=2,
        )
        stage2_best_val = min(history_stage2.history["val_loss"])

    classifier.save(os.path.join(OUTPUT_DIR, "moco_classifier_final.keras"))

    best_stage = "stage1"
    best_model_path = BEST_STAGE1_PATH
    best_val_loss = stage1_best_val
    if stage2_best_val is not None and stage2_best_val < stage1_best_val:
        best_stage = "stage2"
        best_model_path = BEST_STAGE2_PATH
        best_val_loss = stage2_best_val

    best_model = tf.keras.models.load_model(best_model_path, compile=False)
    best_model.save(BEST_OVERALL_PATH)

    summary = {
        "pretrained_checkpoint": resolved_checkpoint,
        "best_stage": best_stage,
        "best_model_path": best_model_path,
        "best_overall_export": BEST_OVERALL_PATH,
        "stage1_best_val_loss": float(stage1_best_val),
        "stage2_best_val_loss": float(stage2_best_val) if stage2_best_val is not None else None,
        "stage1_epochs": EPOCHS_STAGE_1,
        "stage2_epochs": EPOCHS_STAGE_2,
    }
    with open(os.path.join(OUTPUT_DIR, "selection_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    if history_stage2 is None:
        acc = history_stage1.history["accuracy"]
        val_acc = history_stage1.history["val_accuracy"]
        loss = history_stage1.history["loss"]
        val_loss = history_stage1.history["val_loss"]
    else:
        acc = history_stage1.history["accuracy"] + history_stage2.history["accuracy"]
        val_acc = history_stage1.history["val_accuracy"] + history_stage2.history["val_accuracy"]
        loss = history_stage1.history["loss"] + history_stage2.history["loss"]
        val_loss = history_stage1.history["val_loss"] + history_stage2.history["val_loss"]

    history_df = pd.DataFrame(
        {
            "accuracy": acc,
            "val_accuracy": val_acc,
            "loss": loss,
            "val_loss": val_loss,
        }
    )
    history_df.to_csv(os.path.join(OUTPUT_DIR, "training_history.csv"), index=False)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(acc, label="Train Acc")
    plt.plot(val_acc, label="Val Acc")
    if history_stage2 is not None:
        plt.axvline(x=EPOCHS_STAGE_1, color="k", linestyle="--", label="Unfreeze")
    plt.title("MoCo Classifier Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(loss, label="Train Loss")
    plt.plot(val_loss, label="Val Loss")
    if history_stage2 is not None:
        plt.axvline(x=EPOCHS_STAGE_1, color="k", linestyle="--", label="Unfreeze")
    plt.title("MoCo Classifier Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(OUTPUT_DIR, "classification_results.png"), dpi=150, bbox_inches="tight")

    print("\nTraining complete.")
    print(f"Saved classifier to: {OUTPUT_DIR}/moco_classifier_final.keras")
    print(f"Saved Stage 1 best checkpoint to: {BEST_STAGE1_PATH}")
    if history_stage2 is not None:
        print(f"Saved Stage 2 best checkpoint to: {BEST_STAGE2_PATH}")
    print(f"Saved best overall checkpoint to: {BEST_OVERALL_PATH} ({best_stage}, val_loss={best_val_loss:.4f})")


if __name__ == "__main__":
    main()
