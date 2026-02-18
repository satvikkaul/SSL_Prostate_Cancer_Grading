# Self-Supervised Learning for Prostate Cancer Grading (SICAPv2)

**Deep Learning - Final Project**  
**Toronto Metropolitan University (Ryerson University)**

This project implements and compares **three Self-Supervised Learning (SSL)** approaches for histopathological image analysis, specifically targeting prostate cancer Gleason grading using the **SICAPv2 dataset**.

## Project Overview

The goal is to demonstrate how self-supervised pretraining improves deep learning performance on medical imaging tasks with limited labeled data. We compare three approaches:

1. **Baseline (No SSL):** Classifier trained from scratch with random initialization
2. **Autoencoder-SSL:** Reconstruction-based SSL using Convolutional Autoencoder
3. **SimCLR-SSL:** Contrastive learning using SimCLR framework 

### Key Information

* **Dataset:** [SICAPv2](https://data.mendeley.com/datasets/9xxm58dvs3/1) - Prostate Cancer Histopathology (~18,000 patches)
* **Task:** Gleason Grading (4-class classification: NC, G3, G4, G5)
* **Methods:** Three SSL approaches + comprehensive comparison
* **Image Size:** 128×128×3 RGB patches (following paper specification)

### Our Approach

We implement and compare:
- **Reconstruction-based SSL** (Autoencoder): Learns texture/structure features
- **Contrastive SSL** (SimCLR): Learns discriminative features via positive/negative pairs
- **Baseline**: No pretraining (demonstrates SSL benefit)

**Hypothesis:** Contrastive learning will outperform reconstruction-based SSL for classification tasks, as predicted by recent literature.

---

## Project Structure

.
├── data/                         # Data loading & augmentation
│   ├── generator.py              # Shared data generator
│   ├── setup.py                  # Dataset setup script
│   └── augmentations/            # Augmentation pipelines
│       ├── transformations.py    # Common transformations
│       └── aug_simclr.py         # SimCLR-specific augmentations
│
├── models/                       # Model definitions
│   ├── cae_model.py              # Autoencoder architecture
│   └── simclr_model.py           # SimCLR architecture
│
├── training/                     # Training scripts
│   ├── baseline/                 # Baseline (No SSL)
│   ├── cae/                      # Autoencoder (SSL)
│   └── simclr/                   # SimCLR (SSL)
│
├── evaluation/                   # Evaluation scripts
│   ├── baseline/                 # Baseline evaluation
│   ├── cae/                      # Autoencoder evaluation
│   ├── simclr/                   # SimCLR evaluation
│   └── shared/                   # Comparison scripts
│
├── utils/                        # Utility scripts
│   ├── utils_common.py
│   └── utils_retrieval.py
│
├── docs/                         # Documentation
│   ├── [model_cae.md](docs/model_cae.md)       # Autoencoder Architecture
│   └── [model_simclr.md](docs/model_simclr.md) # SimCLR Architecture
│
├── output/                       # Training outputs
└── dataset/                      # Dataset folder

## Setup Instructions

### 1. Prerequisites
Ensure you have Python installed (3.8+ recommended). Then, install the required dependencies:

```bash
pip install -r requirements.txt
```
### 2. Download the Dataset

* Download the SICAPv2 dataset from Mendeley Data.
* Extract the downloaded zip file directly into the dataset/ folder in your project root.
* Ensure your folder structure looks like this:
* dataset/images/ 
* dataset/partition/ (Should contain .xlsx files defining the train/test split)

### 3. Generate CSV Labels

* The training code requires specific CSV files (Train.csv and Test.csv) to map images to their cancer grades. The raw dataset comes with Excel files that the code cannot read by default.
* Run the provided setup script to automatically generate these files:

```Bash
python data/setup.py
```
What this does:

* Scans the dataset/partition/ folder.
* Converts the raw Excel files into dataset/Train.csv and dataset/Test.csv.
* Formats the columns to match the model's expected One-Hot Encoding (NC, G3, G4, G5).

### 4. Configure Training (Optional)

* Open `training/cae/train_cae.py` to adjust training hyperparameters if needed:
* Batch Size: Set to 16 by default. If you run out of memory (OOM error), lower it to 8.
* GPU Settings: The code defaults to using GPU 0. If you are running on CPU, comment out the line os.environ["CUDA_VISIBLE_DEVICES"] = "0".

## Usage

### Training Pipeline

We provide three complete training pipelines for comparison:

---

### **Approach 1: Baseline (No SSL - Train from Scratch)**

Train a classifier with random weight initialization (no pretraining).

```bash
python training/baseline/train_baseline.py
```

**Output:**
- Model: `./output/baseline/best_baseline_classifier.keras`
- Training curves: `./output/baseline/training_curves.png`
- Training time: ~1-2 hours (GPU)

**Purpose:** Establishes baseline performance to measure SSL improvement.

---

### **Approach 2: Autoencoder-SSL (Reconstruction-based)**

#### Step 1: SSL Pretraining
Train autoencoder to reconstruct images (unsupervised):

```bash
python training/cae/train_cae.py
```

**Output:**
- Encoder weights: `./output/models/exp_XXXX/weights/VAE_weights.weights.h5`
- Reconstructions: `./output/models/exp_XXXX/viz/`
- Training time: ~1-2 hours (GPU)

#### Step 2: Fine-tuning
Load pretrained encoder and train classifier:

```bash
# Edit training/cae/finetune_cae.py
python training/cae/finetune_cae.py
```

**Output:**
- Classifier: `./output/final_classifier.keras`
- Results: `./output/classification_results.png`
- Training time: ~1 hour (GPU)

---

### **Approach 3: SimCLR-SSL (Contrastive Learning)** 

#### Step 1: SimCLR Pretraining
Train encoder using contrastive learning (unsupervised):

```bash
python training/simclr/pretrain_simclr.py
```

**Output:**
- Encoder weights: `./output/simclr/encoder_weights.h5`
- Training curves: `./output/simclr/training_curves.png`
- Training time: ~2-3 hours (GPU)

**What it does:**
- Creates two augmented views of each image
- Learns to maximize similarity between views of same image
- Learns to minimize similarity between different images
- No labels required!

#### Step 2: Fine-tuning
Load SimCLR encoder and train classifier:

```bash
python training/simclr/finetune_simclr.py
```

**Output:**
- Classifier: `./output/simclr/best_simclr_classifier.keras`
- Results: `./output/simclr/classification_results.png`
- Training time: ~1 hour (GPU)

---

### **Evaluation & Comparison**

#### Evaluate Each Model
```bash
python evaluation/baseline/eval_baseline.py    # Baseline metrics
python evaluation/cae/eval_cae.py              # Autoencoder metrics
python evaluation/simclr/eval_simclr.py        # SimCLR metrics

#### Compare All Three
```bash
python evaluation/shared/compare_models.py
```

**Output:**
- Comparison table: `./output/model_comparison.csv`
- Comparison plots: `./output/model_comparison_plots.png`
- Summary report: `./output/comparison_report.txt`

**Generates:**
- Accuracy comparison
- Per-class performance
- F1-scores, Cohen's Kappa
- ROC curves
- SSL improvement metrics

---

### Quick Start (Google Colab Pro) 🚀

**Recommended for GPU training!**

1. Upload dataset to Google Drive
2. Open [Google Colab](https://colab.research.google.com/)
3. Clone this repo:
```python
!git clone https://github.com/yourusername/SSL_Prostate_Cancer_Grading.git
%cd SSL_Prostate_Cancer_Grading
```
4. Run training scripts (see [GPU_TRAINING_GUIDE.md](GPU_TRAINING_GUIDE.md))

**Estimated time on Colab Pro:**
- T4 GPU (Free): 4-6 hours total
- A100 GPU (Pro): 2-3 hours total

---## Results & Comparison

### Performance Summary

| Model | Overall Accuracy | Macro F1 | Weighted F1 | Cohen's Kappa |
|-------|-----------------|----------|-------------|---------------|
| **Baseline (No SSL)** | TBD | TBD | TBD | TBD |
| **Autoencoder-SSL** | 62.8% | 0.39 | 0.62 | 0.50 |
| **SimCLR-SSL** | TBD | TBD | TBD | TBD |

*TBD = To be determined after GPU training*

### Per-Class Performance (Autoencoder-SSL)

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| **NC (Non-Cancerous)** | 0.80 | 0.85 | 0.83 | 1727 |
| **G3 (Gleason 3)** | 0.14 | 0.13 | 0.13 | 497 |
| **G4 (Gleason 4)** | 0.54 | 0.65 | 0.59 | 1042 |
| **G5 (Gleason 5)** | 0.00 | 0.00 | 0.00 | 247 |

---

## Methodology & Architecture

### Why Three Approaches?

1. **Baseline (No SSL):** Demonstrates the value of pretraining
2. **Autoencoder:** Reconstruction-based SSL (learns texture/structure)
3. **SimCLR:** Contrastive SSL (learns discriminative features) - 

### SimCLR Framework

```
Input Image (128×128×3)
     ↓
[Data Augmentation Pipeline]
     ├─→ View 1 (Strong augmentation)
     └─→ View 2 (Different strong augmentation)
     ↓
[Shared Encoder]
     ├─→ Features 1 (256-dim)
     └─→ Features 2 (256-dim)
     ↓
[Projection Head]
     ├─→ Embedding 1 (128-dim)
     └─→ Embedding 2 (128-dim)
     ↓
[NT-Xent Loss]
  Goal: Maximize similarity(emb1, emb2)
        Minimize similarity(emb1, other_embeddings)
```

**Key Innovation:** No labels needed! The model learns by identifying which augmented views belong to the same image.

### Augmentation Pipeline (Critical for Histopathology)

**Strong Augmentations:**
- Color jittering (±40% brightness/contrast/saturation, ±10% hue)
- Random crops (80-100% of image)
- Geometric transforms (flips, 90° rotations)
- Gaussian blur (50% probability)

**Why so strong?**
- Histopathology has high stain variation
- Model must learn stain-invariant features
- Stronger augmentation → better generalization

### Encoder Architecture

**Convolutional Encoder (Shared across all methods):**
```python
Input: 128×128×3
  ↓ Conv2D(16, stride=2) + BatchNorm + LeakyReLU + Dropout
  ↓ Conv2D(32, stride=2)
  ↓ Conv2D(64, stride=2) ← Skip connection saved
  ↓ Conv2D(128, stride=2)
  ↓ Conv2D(256, stride=2)
  ↓ Bottleneck: Conv2D(128) → Conv2D(64) → Conv2D(128)
  ↓ Flatten → Dense(256)
Output: 256 features
```

**For Classification:**
```python
Encoder features (256-dim)
  ↓ GlobalMaxPooling2D → Dense(200, ReLU)
  ↓ Dense(4, Softmax)
Output: [NC, G3, G5, G4] probabilities
```

### Training Strategy

**SSL Pretraining:**
- Epochs: 30 (SimCLR), 15 (Autoencoder)
- Batch Size: 64 (SimCLR), 16 (Autoencoder)
- Optimizer: Adam with cosine learning rate decay
- Temperature: 0.5 (SimCLR NT-Xent loss)

**Fine-tuning:**
- Two-stage: Frozen encoder (50 epochs) → Optional unfrozen (0-50 epochs)
- Loss: Focal loss (α=0.5, γ=2.0) for class imbalance
- Optimizer: SGD with momentum=0.9, clipnorm=1.0
- Learning Rate: 1e-5 (stage 1), 5e-5 (stage 2)

### Class Imbalance Handling

**Challenge:** Dataset is heavily imbalanced
- NC: 49% (majority)
- G4: 30%
- G3: 14%
- G5: 7% (minority)

**Solutions Implemented:**
1. **Focal Loss:** Down-weights easy examples, focuses on hard cases
2. **Data Augmentation:** Increases effective training samples
3. **Gradient Clipping:** Prevents training instability

**Not Used:** Class weights (caused training collapse in experiments)

---

## License

This project is for educational purposes as part of CP8321 Deep Learning course requirements.
