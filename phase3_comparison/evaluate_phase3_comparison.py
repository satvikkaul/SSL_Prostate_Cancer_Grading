"""
Evaluate Phase 3 fine-tuned checkpoints without overwriting Phase 2 artifacts.

This script mirrors the robust Phase 2 notebook evaluation logic:
- preserves training class order ['NC', 'G3', 'G5', 'G4']
- uses a Keras 3 compatibility loader for .keras archives
- reads labels directly from Test.csv to avoid generator cycling issues

Usage example:
    python phase3_comparison/evaluate_phase3_comparison.py \
        --runs-dir /content/drive/MyDrive/Prostate_SSL/runs \
        --cae-run cae_phase3_20260330_123456 \
        --simclr-run simclr_phase3_20260330_124500 \
        --moco-run moco_phase3_20260330_125000
"""

import argparse
import json
import os
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report, cohen_kappa_score, confusion_matrix, roc_auc_score
from tensorflow import keras

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.generator import DataGenerator


CLASS_NAMES = ["NC", "G3", "G5", "G4"]
IMG_DIM = (128, 128, 3)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Phase 3 fine-tuned checkpoints")
    parser.add_argument("--runs-dir", required=True, help="Drive runs directory containing baseline/phase3 folders")
    parser.add_argument(
        "--dataset-root",
        default="./dataset",
        help="Local dataset root containing Test.csv and images/",
    )
    parser.add_argument(
        "--baseline-run",
        default="baseline_20260329_110621",
        help="Baseline run folder name under runs-dir",
    )
    parser.add_argument("--cae-run", required=True, help="Phase 3 CAE run folder name under runs-dir")
    parser.add_argument("--simclr-run", required=True, help="Phase 3 SimCLR run folder name under runs-dir")
    parser.add_argument("--moco-run", required=True, help="Phase 3 MoCo run folder name under runs-dir")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for Phase 3 comparison outputs. Defaults to ./phase3_comparison",
    )
    return parser.parse_args()


def load_model_compat(model_path):
    model_path = str(model_path)

    def _strip(obj):
        if isinstance(obj, dict):
            return {k: _strip(v) for k, v in obj.items() if k != "quantization_config"}
        if isinstance(obj, list):
            return [_strip(i) for i in obj]
        return obj

    with zipfile.ZipFile(model_path, "r") as src:
        names = src.namelist()
        cfg = json.loads(src.read("config.json").decode("utf-8"))
        patched_cfg = json.dumps(_strip(cfg))

        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".keras")
        os.close(tmp_fd)
        try:
            with zipfile.ZipFile(tmp_path, "w", zipfile.ZIP_DEFLATED) as dst:
                for item in names:
                    dst.writestr(item, patched_cfg if item == "config.json" else src.read(item))
            return keras.models.load_model(tmp_path, compile=False)
        finally:
            os.unlink(tmp_path)


def load_test_generator(dataset_root):
    test_csv = dataset_root / "Test.csv"
    images_dir = dataset_root / "images"
    if not test_csv.exists():
        raise FileNotFoundError(f"Test.csv not found at {test_csv}")

    df = pd.read_csv(test_csv)
    df["image_name"] = df["image_name"].astype(str)
    return DataGenerator(
        data_frame=df,
        y=IMG_DIM[0],
        x=IMG_DIM[1],
        target_channels=IMG_DIM[2],
        y_cols=CLASS_NAMES,
        batch_size=32,
        path_to_img=str(images_dir),
        shuffle=False,
        data_augmentation=False,
        mode="custom",
    )


def evaluate_checkpoint(model_path, model_name, stage_name, dataset_root):
    model_path = Path(model_path)
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        return None

    print("\n" + "=" * 70)
    print(f"Evaluating: {model_name} - {stage_name}")
    print(f"Model: {model_path}")
    print("=" * 70)

    model = load_model_compat(model_path)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    test_df = pd.read_csv(dataset_root / "Test.csv")
    y_true = np.argmax(test_df[CLASS_NAMES].values.astype(np.float32), axis=1)
    n_samples = len(y_true)

    test_gen_pred = load_test_generator(dataset_root)
    y_proba = model.predict(test_gen_pred, verbose=1)[:n_samples]
    y_pred = np.argmax(y_proba, axis=1)

    report = classification_report(y_true, y_pred, target_names=CLASS_NAMES, output_dict=True, zero_division=0)
    kappa = cohen_kappa_score(y_true, y_pred)

    auc_scores = {}
    try:
        for i, cls in enumerate(CLASS_NAMES):
            auc_scores[cls] = roc_auc_score((y_true == i).astype(int), y_proba[:, i])
    except Exception as exc:
        print(f"Warning: AUC skipped: {exc}")
        auc_scores = {c: 0.0 for c in CLASS_NAMES}

    cm = confusion_matrix(y_true, y_pred)
    result = {
        "model": model_name,
        "stage": stage_name,
        "accuracy": report["accuracy"],
        "macro_f1": report["macro avg"]["f1-score"],
        "weighted_f1": report["weighted avg"]["f1-score"],
        "kappa": kappa,
        "n_samples": n_samples,
        "confusion_matrix": cm.tolist(),
        "per_class": {},
    }

    for cls in CLASS_NAMES:
        if cls in report:
            result["per_class"][cls] = {
                "precision": report[cls]["precision"],
                "recall": report[cls]["recall"],
                "f1": report[cls]["f1-score"],
                "support": report[cls]["support"],
                "auc": auc_scores.get(cls, 0.0),
            }

    print(f"✅ Accuracy: {result['accuracy']:.1%} | Macro F1: {result['macro_f1']:.3f} | Kappa: {result['kappa']:.3f}")
    return result


def build_comparison_rows(results):
    by_model = {}
    for result in results:
        by_model.setdefault(result["model"], {})[result["stage"]] = result

    rows = []
    for model_name in ["CAE-SSL", "SimCLR-SSL", "MoCo-SSL", "Baseline (No SSL)"]:
        if model_name not in by_model:
            continue

        stages = by_model[model_name]
        if "Stage 1 (Frozen)" in stages and "Stage 2 (End-to-End)" in stages:
            stage1 = stages["Stage 1 (Frozen)"]
            stage2 = stages["Stage 2 (End-to-End)"]
            rows.append(
                {
                    "Model": model_name,
                    "Stage 1 Acc": f"{stage1['accuracy']:.1%}",
                    "Stage 2 Acc": f"{stage2['accuracy']:.1%}",
                    "Acc Δ": f"{(stage2['accuracy'] - stage1['accuracy']) * 100:+.1f}%",
                    "Stage 1 F1": f"{stage1['macro_f1']:.3f}",
                    "Stage 2 F1": f"{stage2['macro_f1']:.3f}",
                    "F1 Δ": f"{stage2['macro_f1'] - stage1['macro_f1']:+.3f}",
                    "Stage 1 κ": f"{stage1['kappa']:.3f}",
                    "Stage 2 κ": f"{stage2['kappa']:.3f}",
                    "κ Δ": f"{stage2['kappa'] - stage1['kappa']:+.3f}",
                    "Better Stage": "🟢 Stage 2" if stage2["kappa"] > stage1["kappa"] else "🔴 Stage 1",
                    "n_samples": stage2["n_samples"],
                }
            )
        elif "Single Stage" in stages:
            stage = stages["Single Stage"]
            rows.append(
                {
                    "Model": model_name,
                    "Stage 1 Acc": "N/A",
                    "Stage 2 Acc": f"{stage['accuracy']:.1%}",
                    "Acc Δ": "N/A",
                    "Stage 1 F1": "N/A",
                    "Stage 2 F1": f"{stage['macro_f1']:.3f}",
                    "F1 Δ": "N/A",
                    "Stage 1 κ": "N/A",
                    "Stage 2 κ": f"{stage['kappa']:.3f}",
                    "κ Δ": "N/A",
                    "Better Stage": "N/A (No SSL)",
                    "n_samples": stage["n_samples"],
                }
            )
        elif "Stage 2 (End-to-End) [Selected]" in stages:
            stage2 = stages["Stage 2 (End-to-End) [Selected]"]
            rows.append(
                {
                    "Model": model_name,
                    "Stage 1 Acc": "See val_loss*",
                    "Stage 2 Acc": f"{stage2['accuracy']:.1%}",
                    "Acc Δ": "N/A",
                    "Stage 1 F1": "See val_loss*",
                    "Stage 2 F1": f"{stage2['macro_f1']:.3f}",
                    "F1 Δ": "N/A",
                    "Stage 1 κ": "See val_loss*",
                    "Stage 2 κ": f"{stage2['kappa']:.3f}",
                    "κ Δ": "N/A",
                    "Better Stage": "🟢 Stage 2 (selected)",
                    "n_samples": stage2["n_samples"],
                }
            )

    return pd.DataFrame(rows)


def main():
    args = parse_args()
    runs_dir = Path(args.runs_dir)
    dataset_root = Path(args.dataset_root)
    output_dir = Path(args.output_dir) if args.output_dir else runs_dir / "phase3_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Runs dir: {runs_dir}")
    print(f"Dataset root: {dataset_root}")
    print(f"Output dir: {output_dir}")

    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found at {dataset_root}")

    results = []

    checkpoints = [
        (runs_dir / args.baseline_run / "baseline" / "best_baseline_classifier.keras", "Baseline (No SSL)", "Single Stage"),
        (runs_dir / args.cae_run / "best_model_stage1.keras", "CAE-SSL", "Stage 1 (Frozen)"),
        (runs_dir / args.cae_run / "best_model_fine_tuned.keras", "CAE-SSL", "Stage 2 (End-to-End)"),
        (runs_dir / args.simclr_run / "simclr" / "best_simclr_classifier.keras", "SimCLR-SSL", "Stage 1 (Frozen)"),
        (runs_dir / args.simclr_run / "simclr" / "best_simclr_fine_tuned.keras", "SimCLR-SSL", "Stage 2 (End-to-End)"),
        (runs_dir / args.moco_run / "moco" / "best_moco_fine_tuned.keras", "MoCo-SSL", "Stage 2 (End-to-End) [Selected]"),
    ]

    for model_path, model_name, stage_name in checkpoints:
        result = evaluate_checkpoint(model_path, model_name, stage_name, dataset_root)
        if result is not None:
            results.append(result)

    comparison_df = build_comparison_rows(results)
    comparison_path = output_dir / "stage_comparison_table.csv"
    comparison_df.to_csv(comparison_path, index=False)

    detailed_path = output_dir / "stage_comparison_detailed_results.json"
    with open(detailed_path, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 70)
    print("PHASE 3 COMPARISON COMPLETE")
    print("=" * 70)
    print(comparison_df.to_string(index=False))
    print(f"\nSaved comparison table: {comparison_path}")
    print(f"Saved detailed results: {detailed_path}")


if __name__ == "__main__":
    main()