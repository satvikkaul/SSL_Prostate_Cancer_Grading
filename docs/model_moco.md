# Model Architecture & Implementation Guide - MoCo v2

**Project:** Self-Supervised Learning for Prostate Cancer Grading (SICAPv2)
**Approach:** MoCo v2 Contrastive Learning -> Transfer Learning -> Classification
**Date:** March 2026

---

## Table of Contents
1. [Architecture Overview](#architecture-overview)
2. [File Structure & Responsibilities](#file-structure--responsibilities)
3. [SSL Pretraining Pipeline](#ssl-pretraining-pipeline)
4. [Downstream Classification Pipeline](#downstream-classification-pipeline)
5. [Learning Process](#learning-process)
6. [Data Flow](#data-flow)
7. [Data Augmentation Strategy](#data-augmentation-strategy)
8. [Critical Implementation Details](#critical-implementation-details)
9. [Current Training Interpretation](#current-training-interpretation)
10. [Comparison with SimCLR and CAE](#comparison-with-simclr-and-cae)

---

## Architecture Overview

### High-Level Concept

```
┌──────────────────────────────────────────────────────────────────────┐
│                    SELF-SUPERVISED LEARNING                         │
│                    (MOMENTUM CONTRAST / MOCO v2)                    │
│                                                                      │
│  Phase 1: PRETEXT TASK (Unsupervised)                                │
│                                                                      │
│          Same Image -> Two Augmented Views                           │
│   ┌──────────┐                              ┌──────────┐             │
│   │ Query    │ -> Encoder_q -> MLP -> q    │ Key      │ -> Encoder_k │
│   │ View xq  │                              │ View xk  │ -> MLP -> k  │
│   └──────────┘                              └──────────┘             │
│            \                                         /               │
│             \______________ InfoNCE Loss ___________/                │
│                      positive: (q, k)                                │
│                      negatives: queue[K]                             │
│                                                                      │
│  Encoder_k is updated by momentum from Encoder_q                     │
│  Queue stores previous key embeddings as negatives                   │
└──────────────────────────────────────────────────────────────────────┘
                             ↓
                     Transfer Encoder Weights
                             ↓
┌──────────────────────────────────────────────────────────────────────┐
│                   DOWNSTREAM TASK (Supervised)                      │
│                                                                      │
│  Phase 2: CLASSIFICATION                                            │
│  ┌──────────┐      ┌────────────┐      ┌──────────────┐             │
│  │  Image   │  ->  │ MoCo       │  ->  │ Classifier   │  -> [NC/    │
│  │ 128x128  │      │ Encoder    │      │ Head         │     G3/G5/   │
│  └──────────┘      │(Pretrained)│      │(Dropout+DNN) │      G4]     │
│                    └────────────┘      └──────────────┘             │
│                           ↑                                          │
│                   Frozen or Fine-tuned                               │
└──────────────────────────────────────────────────────────────────────┘
```

### Architecture Dimensions

**SSL Pretraining (training/moco/pretrain_moco.py):**
```
Input: 128x128x3 RGB image

Step 1: Two augmented views
  xq = augmenter_q(image)
  xk = augmenter_k(image)

Step 2: Query encoder (encoder_q)
  ResNet50 backbone (weights=None, include_top=False, pooling="avg")
  Output: 2048 feature vector

Step 3: Projection head
  Dense(2048) -> 2048
  BatchNorm   -> 2048
  ReLU        -> 2048   <- proj_relu
  Dense(128)  -> 128
  L2 normalize -> 128   <- final contrastive embedding

Step 4: Key encoder (encoder_k)
  Same architecture as encoder_q
  Not trained directly by gradients
  Updated by momentum from encoder_q

Step 5: Queue
  Size K = 4096
  Stores 128-d normalized key embeddings

Step 6: Contrastive loss
  Positive pair: current q and current k
  Negative pairs: q against all embeddings in the queue
```

**Downstream Classification Head (training/moco/finetune_moco.py):**
```
Encoder feature used: proj_relu (2048-d hidden representation)
  ↓
Dropout(0.2)
  ↓
Dense(256, ReLU)
  ↓
Dense(4, Softmax) -> [NC, G3, G5, G4] probabilities
```

**Important Design Choice:**
- The classifier is attached to `proj_relu`, not the final L2-normalized 128-d contrastive output.
- This gives the downstream classifier a richer representation than the final normalized projection vector.

---

## File Structure & Responsibilities

### 1. **models/moco_model.py** - Core MoCo v2 Architecture

**Purpose:** Defines the ResNet50 encoder, projection head, momentum encoder logic, queue, and GPU-side augmentations.

**Key Components:**

1. `build_encoder()`
```python
ResNet50(include_top=False, pooling="avg", weights=None)
 -> Dense(2048)
 -> BatchNormalization
 -> ReLU
 -> Dense(128)
 -> L2 normalize
```

2. `MoCoV2Queue`
```python
Queue size: 4096
Feature dimension: 128
FIFO replacement strategy
```

3. `MoCoV2Model`
```python
encoder_q: trained with gradients
encoder_k: updated by momentum, no direct gradient updates
temperature: 0.2
momentum: 0.999
```

4. GPU-side augmentation layers
- `RandomResizedCrop`
- `ColorJitter`
- `RandomGrayscale`
- `RandomGaussianBlur`

These are implemented as `tf.keras.layers.Layer` modules so the expensive augmentation path stays inside TensorFlow rather than Python/OpenCV.

---

### 2. **training/moco/pretrain_moco.py** - SSL Pretraining Orchestrator

**Purpose:** Runs MoCo pretraining on `Pretrain_Manifest.csv`.

**Key Responsibilities:**
- Loads all pretraining image paths from the unified manifest
- Creates MoCo model and negative queue
- Applies cosine LR schedule with warmup
- Saves:
  - query encoder weights
  - full training state
  - pretraining log
- Supports resume from:
  - legacy encoder weights
  - full-state checkpoints

**Current Configuration:**
```python
queue_size = 4096
momentum = 0.999
temperature = 0.2
lr = 0.03
weight_decay = 1e-4
save_freq = 10
```

**Outputs:**
- `./output/models/moco/encoder_q_epochXXX.weights.h5`
- `./output/models/moco/state/`
- `./output/results/moco/logs/pretrain_log.csv`

---

### 3. **training/moco/finetune_moco.py** - Downstream Classification

**Purpose:** Loads a pretrained MoCo encoder and trains a Gleason classifier.

**Training Strategy:**

**Stage 1: Train classification head only**
- Freeze the pretrained encoder
- Train only:
  - `Dropout(0.2)`
  - `Dense(256, relu)`
  - `Dense(4, softmax)`

**Stage 2: Fine-tune the encoder**
- Unfreeze most encoder layers
- Keep BatchNorm-like layers frozen by name rule
- Continue training with lower LR

**Saved Outputs:**
- `best_moco_classifier.keras` -> best Stage 1 checkpoint
- `best_moco_fine_tuned.keras` -> best Stage 2 checkpoint
- `best_moco_overall.keras` -> best overall exported model
- `selection_summary.json` -> records which stage won

---

### 4. **evaluation/moco/eval_moco.py** - Held-Out Test Evaluation

**Purpose:** Evaluates the selected MoCo classifier on `Test.csv`.

**Metrics Saved:**
- classification report
- confusion matrix
- per-class ROC curves
- Cohen's kappa
- `evaluation_metrics.txt`

**Default evaluation model priority:**
1. `best_moco_overall.keras`
2. `best_moco_fine_tuned.keras`
3. `best_moco_classifier.keras`

---

### 5. **evaluation/moco/tsne_moco.py** - Representation Analysis

**Purpose:** Extracts MoCo encoder features and visualizes them using t-SNE.

**Use case:**
- qualitative feature quality check
- inspect separation between NC / G3 / G4 / G5
- inspect relationship between SICAP and PANDA patches

---

## SSL Pretraining Pipeline

### Step-by-Step

1. Load image paths from `Pretrain_Manifest.csv`
2. Read and resize base image on CPU with OpenCV
3. Generate two augmented views on TensorFlow side
4. Pass `xq` through `encoder_q`
5. Pass `xk` through `encoder_k`
6. Compute InfoNCE loss:
   - positive pair = `(q, k)`
   - negatives = queue contents
7. Update `encoder_q` by SGD
8. Update queue with current keys
9. Update `encoder_k` via momentum

### Core Training Step

```python
xq = augmenter_q(images, training=True)
xk = augmenter_k(images, training=True)

q = encoder_q(xq, training=True)
k = encoder_k(xk, training=False)

loss = info_nce_loss(q, k)

apply_gradients(encoder_q)
queue.dequeue_and_enqueue(k)
momentum_update(encoder_k <- encoder_q)
```

### Why MoCo Instead of SimCLR?

SimCLR depends heavily on large batch sizes because negatives come from other images in the same batch.

MoCo decouples the number of negatives from batch size by using a queue:
- small batch still possible
- many negatives still available
- more practical under hardware constraints

This is exactly why MoCo was added to strengthen the project after SimCLR underperformed under limited compute.

---

## Downstream Classification Pipeline

### Stage 1: Frozen Encoder Evaluation

**Goal:** Measure feature quality directly.

Procedure:
- load pretrained MoCo encoder
- freeze all encoder layers
- attach classifier head
- train head on `TrainSplit.csv`
- validate on `Val.csv`

Interpretation:
- if Stage 1 works well, the pretrained representation itself is useful

### Stage 2: End-to-End Fine-Tuning

**Goal:** Measure practical task performance after adaptation.

Procedure:
- start from Stage 1 model
- unfreeze most encoder layers
- train with lower LR
- save best fine-tuned checkpoint

Interpretation:
- if Stage 2 improves, the pretrained encoder transfers well and benefits from task-specific adaptation

---

## Learning Process

### What MoCo Learns During Pretraining

MoCo does **not** learn reconstruction.
It learns to make two augmented views of the same histology patch land close together in embedding space while keeping different images apart.

That means it should learn:
- glandular structure
- texture patterns
- stain-invariant morphology
- coarse discriminative tissue features

### Role of the Queue

The queue acts as a memory bank of negatives.

Instead of comparing the query only to the current batch:
- current positive key is the match
- 4096 queued keys act as negatives

This makes contrastive learning stronger even when batch size is only `8` or `16`.

### Role of the Momentum Encoder

`encoder_k` is not updated by direct gradients.
It is updated as:

```python
wk = m * wk + (1 - m) * wq
```

This keeps key representations more stable, which improves queue consistency.

---

## Data Flow

### Pretraining Data Flow

```
Pretrain_Manifest.csv
    ↓
MoCoDataGenerator
    ↓
CPU image loading + resize
    ↓
TensorFlow augmenters create xq / xk
    ↓
encoder_q(xq) -> q
encoder_k(xk) -> k
    ↓
InfoNCE(q, k, queue)
    ↓
Update encoder_q / queue / encoder_k
```

### Fine-Tuning Data Flow

```
TrainSplit.csv / Val.csv
    ↓
DataGenerator
    ↓
MoCo encoder
    ↓
proj_relu features
    ↓
Dropout(0.2)
    ↓
Dense(256, ReLU)
    ↓
Dense(4, Softmax)
```

### Evaluation Data Flow

```
Test.csv
    ↓
Selected MoCo checkpoint
    ↓
Predictions
    ↓
Accuracy / F1 / Kappa / ROC / Confusion Matrix
```

---

## Data Augmentation Strategy

The MoCo augmenter is designed to stay close to MoCo v2 style contrastive learning:

1. **RandomResizedCrop**
   - keeps global structure but changes spatial framing
2. **Horizontal Flip**
   - adds geometric invariance
3. **ColorJitter**
   - reduces over-reliance on stain intensity
4. **RandomGrayscale**
   - reduces color dependence
5. **Gaussian Blur**
   - encourages robust shape/texture encoding

### Important Practical Choice

Base image IO is handled in `data/generator.py` using OpenCV.
Heavy augmentations are handled in `models/moco_model.py` using TensorFlow layers.

This hybrid design was introduced to avoid Python-side bottlenecks and make the pipeline more stable.

---

## Critical Implementation Details

### 1. macOS / Metal Support

For local Apple Silicon development:
- TensorFlow uses the Metal backend when available
- local pilot runs are viable
- full long runs are still better suited to Colab

### 2. CPU Graph-Stall Fix

On CPU-only Apple Silicon execution, the MoCo loss path could stall when graph-compiled.
The current implementation keeps that path eager-friendly for stability during local debugging.

### 3. Checkpoint Semantics

MoCo now has three distinct downstream checkpoints:
- `best_moco_classifier.keras` -> Stage 1 best
- `best_moco_fine_tuned.keras` -> Stage 2 best
- `best_moco_overall.keras` -> official best overall export

### 4. Evaluation Protocol

The corrected split discipline is:
- `TrainSplit.csv` -> training
- `Val.csv` -> model selection
- `Test.csv` -> final evaluation only

This must be preserved in all downstream runs for a fair paper-ready comparison.

---

## Current Training Interpretation

### Pretraining Loss

Early MoCo loss can be noisy because:
- the queue starts random
- the encoder starts random
- strong augmentations make positives difficult

A rising loss in the middle of epoch 1 is not automatically a failure.
What matters more is:
- whether checkpoints save correctly
- whether later epochs stabilize
- whether downstream fine-tuning performs competitively

### Downstream Signal

In this project, the strongest validation signal comes from downstream classification:
- Stage 1 tests raw feature transfer quality
- Stage 2 tests practical fine-tuned performance

If downstream validation improves meaningfully, the pretraining is useful even if the early MoCo loss curve is noisy.

---

## Comparison with SimCLR and CAE

### Compared with CAE

CAE learns by:
- reconstructing the original image

MoCo learns by:
- distinguishing matching and non-matching augmented views

CAE is reconstruction-based SSL.
MoCo is queue-based contrastive SSL.

### Compared with SimCLR

SimCLR:
- same encoder on both views
- negatives come from the current batch only
- sensitive to large batch size

MoCo:
- query encoder + momentum encoder
- negatives come from a queue
- more suitable under realistic hardware constraints

### Important Fairness Note

The current MoCo implementation uses a different encoder backbone from CAE/SimCLR:
- CAE / SimCLR: custom convolutional encoder
- MoCo: ResNet50 backbone

So MoCo is currently a stronger modern SSL baseline, but not a pure same-backbone comparison.

That is acceptable for strengthening the project's SSL diversity, but it should be stated clearly in any report or presentation.

---

## Summary

MoCo v2 in this project is a momentum-based contrastive learning pipeline designed to address the batch-size limitations of SimCLR.

Its key strengths are:
- stable negative queue
- momentum encoder
- strong transfer-ready features
- practical downstream fine-tuning path

Its role in the project is to provide a more compute-realistic contrastive SSL baseline for histopathology under limited hardware conditions.
