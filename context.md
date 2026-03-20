# Project Context: Self-Supervised Learning for Prostate Cancer Grading

## Overview
This project focuses on automating the Gleason grading of prostate cancer histology images (SICAPv2 dataset) using Self-Supervised Learning (SSL) to overcome the limitation of scarce labeled data.

## Methodology
The project compares four approaches:
1.  **Baseline (Supervised)**: A CNN trained from scratch on labeled data.
2.  **Convolutional Autoencoder (CAE)**: Image reconstruction as pretext task.
3.  **SimCLR**: NT-Xent contrastive loss on dual augmented views.
4.  **MoCo v2** *(In Progress)*: Momentum Contrast with ResNet50 + PANDA data expansion.

## Workflow
1.  **Data Prep**: `data/setup.py` — converts SICAPv2 partition Excel files to CSV, creates `TrainSplit.csv` and `Val.csv`, extracts PANDA-PLUS patches, and builds `Pretrain_Manifest.csv`.
2.  **Pretraining**: `training/` scripts train SSL encoders (CAE, SimCLR, MoCo v2).
3.  **Fine-tuning**: `training/` scripts train classifier head on top of frozen/unfrozen encoder.
4.  **Evaluation**: `evaluation/` scripts generate metrics (F1, Accuracy, Confusion Matrix, t-SNE).

## Current Status & Metrics
-   **Dataset**: SICAPv2 (Patches, 10x magnification) + PANDA-PLUS baseline parquet.
-   **Classes**: NC (Non-Cancerous), G3, G4, G5.
-   **Class Imbalance**: Highly imbalanced (NC > G4 > G3 > G5).
-   **Best Model (CAE)**:
    -   **Accuracy**: ~62.8%
    -   **Weighted F1**: ~0.62
    -   **Weakness**: G5 class has 0 F1 score in some runs.
-   **Structure**: Modular directories (`data/`, `models/`, `training/`, `evaluation/`, `utils/`, `docs/`).

## MoCo v2 Implementation (Stage 1 — 2026-02-24)
### New Files Added
| File | Description |
|---|---|
| `models/moco_model.py` | ResNet50 encoder, MLP projection head, FIFO queue (K=4096), InfoNCE loss |
| `training/moco/pretrain_moco.py` | Full training loop: AMP, cosine LR, gradient clipping, OOM fallback |
| `evaluation/moco/tsne_moco.py` | t-SNE visualization colored by Gleason grade + PANDA source |

### Data Pipeline Changes
- `data/setup.py`: Added `extract_panda_patches()` — decodes PANDA-PLUS baseline parquet → saves `dataset/panda_images/`.
- `data/setup.py`: Added `build_pretrain_manifest()` — merges non-test SICAP training data + PANDA → `dataset/Pretrain_Manifest.csv`.
- `data/generator.py`: Added `MoCoDataGenerator` to load base images via OpenCV safely on the CPU (avoiding Windows threadpool deadlocks).

### Pipeline Fixes (Debugging)
- **Bottleneck**: Pipeline was struggling at 1 min/step due to Python Global Interpreter Lock (GIL) and Windows `tf.data` thread pool deadlocks combining OpenCV with `tf.data.Dataset.map`.
- **Solution (Hybrid CPU/GPU Pipeline)**: 
  - CPU (`generator.py`) handles *only* file IO and basic resizing using `cv2` (immune to TF deadlocks).
  - GPU (`moco_model.py`) handles all complex augmentations (ColorJitter, Blur, Cropping, Grayscale) natively via custom `tf.keras.layers`. This circumvents all CPU locks and runs in milliseconds inside `tf.GradientTape`.
- **AMP Bug**: `tf.image` ops (like `random_contrast`) do not have `float16` kernels. Handled this by explicitly casting tensors to `float32` inside the custom augmentation layers before applying math, and casting back to `float16` before passing to the encoder.

### Config
| Param | Value |
|---|---|
| Backbone | ResNet50 (weights=None) |
| Queue K | 4,096 |
| Batch size | 16 (auto-fallback to 8 on OOM) |
| Patch size | 128×128 |
| Momentum m | 0.999 |
| Temperature τ | 0.2 |
| Mixed Precision | Auto (mixed precision when GPU is available, float32 otherwise) |
| LR schedule | Cosine annealing + 5-epoch warmup |
| PANDA source | Baseline parquet only |

### Run Order
```
# Step 1: Extract PANDA patches + build manifest (run once)
python data/setup.py

# Step 2: Architecture sanity check
python models/moco_model.py

# Step 3: 10-epoch pilot run
python training/moco/pretrain_moco.py --epochs 10 --batch_size 16

# Step 4: t-SNE (after pilot checkpoint saved)
python evaluation/moco/tsne_moco.py --checkpoint output/models/moco/encoder_q_epoch010.weights.h5 --n_samples 500
```

### Success Criteria
- [ ] `data/setup.py` runs without error; `TrainSplit.csv`, `Val.csv`, and `Pretrain_Manifest.csv` are generated successfully.
- [ ] `moco_model.py` prints: ~23M params, queue shape (4096, 128), positive loss value.
- [ ] 10 pilot epochs complete without OOM; loss decreases.
- [ ] `tsne_pilot.png` shows non-random cluster structure.

## Key Challenges to Address
1.  **G5 Classification**: Model struggles with G5 due to extreme scarcity — PANDA data expansion targets this.
2.  **Feature Quality**: MoCo v2 expected to outperform SimCLR/CAE features.
3.  **Validation**: Need cross-fold validation for robust final metrics.

## Review Findings (2026-03-17)
- **Project standing**: Baseline, CAE, and SimCLR pipelines are present, and the protocol cleanup is now reflected in the main training/evaluation flow. CAE remains the strongest historically reported path so far, while MoCo v2 has moved from "pretraining only" to a code-complete end-to-end pipeline that still needs full experiment runs.
- **MoCo status**: `models/moco_model.py` builds successfully in the project `venv`, the pretraining path runs on the current macOS Apple Silicon setup after removing the CPU-side graph stall, and the rebuilt `dataset/Pretrain_Manifest.csv` excludes SICAP test images. New downstream scripts now exist for MoCo fine-tuning and evaluation: `training/moco/finetune_moco.py` and `evaluation/moco/eval_moco.py`.
- **Protocol status**: The train/validation/test split is now aligned across the main supervised and SSL training paths. Training uses `TrainSplit.csv`, checkpoint selection uses `Val.csv`, and `Test.csv` is reserved for final evaluation.
- **Analysis/reporting status**: `evaluation/shared/compare_models.py` no longer depends on partial hard-coded CAE metrics and now reads real evaluation outputs from whichever model result files are present. CAE evaluation output has also been brought into the same saved-metrics format used by baseline, SimCLR, and MoCo.
- **Environment status**: The project currently runs in the repo `venv` with CPU-only TensorFlow on this macOS machine. The default shell `python` is not the project environment, so repo commands should continue to be run through `./venv/bin/python` unless the environment is activated explicitly.
- **Remaining cleanup gap**: Some legacy CAE/SimCLR paths still rely on hard-coded checkpoint names or duplicated artifact save locations. The next cleanup pass should make checkpoint discovery automatic and finish standardizing model-specific output folders.
- **Immediate next step**: Run a complete MoCo pretrain -> fine-tune -> evaluate cycle, then regenerate shared comparisons using only actual saved outputs from baseline, CAE, SimCLR, and MoCo.
