# Research Enhancement Plan

**Goal**: Prepare the project for publication (Conference/Journal) by enhancing model performance and conducting rigorous analysis.

**Target**: 10-12 page paper, emphasizing SSL efficacy on imbalanced medical data.

## 0. Protocol & Evaluation Fixes (Priority: Immediate)
*Rationale*: Current training scripts reuse `Test.csv` for validation/model selection, and MoCo manifest construction includes SICAP test images. This must be corrected before any new metrics are treated as reliable.
- [ ] **Separate train / validation / test correctly**:
    - Generate an explicit validation split from `Train.csv`.
    - Update baseline, CAE, and SimCLR training scripts to use validation data only for checkpointing and tuning.
    - Reserve `Test.csv` strictly for final evaluation scripts.
- [ ] **Fix MoCo pretraining protocol**:
    - Build MoCo pretraining manifest from non-test SICAP data plus PANDA only.
    - Avoid using SICAP test images during pretraining or representation analysis intended for model selection.
- [ ] **Regenerate comparisons from clean outputs**:
    - Remove dependence on hard-coded metrics in comparison scripts.
    - Compare methods only after retraining/evaluating under the corrected split protocol.

## 0.5. Complete MoCo Integration (Priority: High)
*Rationale*: MoCo code exists, but the full downstream experiment is not yet complete.
- [ ] **Finish MoCo implementation quality**:
    - Fix encoder training/inference behavior and augmentation fidelity in `models/moco_model.py`.
    - Make checkpoint resume restore the full MoCo state, not just `encoder_q`.
- [ ] **Add downstream MoCo pipeline**:
    - Create MoCo fine-tuning script for Gleason classification.
    - Create MoCo evaluation script with the same metrics/reporting used for CAE and SimCLR.
- [ ] **Integrate MoCo into project comparison outputs**:
    - Add MoCo results to shared comparison tables/plots once trained under the corrected protocol.

## 1. Implement Advanced SSL Methods (Priority: High)
*Rationale*: SimCLR relies on large batch sizes which can be resource-intensive. MoCo v2 (Momentum Contrast) separates the queue size from batch size, allowing for better contrastive learning on limited hardware.
- [ ] **Implement MoCo v2**:
    - Create `models/moco_model.py`: Implement Momentum Encoder and Queue.
    - Create `training/moco/pretrain_moco.py`: Training loop with momentum update.
    - Compare performance vs. SimCLR and CAE.
- [ ] **Explore BYOL (Bootstrap Your Own Latent)** (Secondary):
    - If MoCo v2 performance saturates, explore BYOL which eliminates the need for negative pairs.

## 2. Comprehensive Ablation Studies (Priority: High)
*Rationale*: To publish, we must prove *why* the model works, not just *that* it works.
- [ ] **Augmentation Impact**: Test combinations (Color Jitter vs. Cutout vs. Gaussian Blur).
- [ ] **Encoder Architecture**: ResNet-18 vs. ResNet-50 vs. Custom CNN.
- [ ] **Projection Head**: Linear vs. MLP (2-layer) vs. MLP (3-layer).
- [ ] **Temperature Scaling**: Effect of temperature in NT-Xent loss (0.1, 0.5, 1.0).

## 3. Addressing Class Imbalance (G5 Focus) (Priority: High)
*Rationale*: G5 (Gleason Score 5) is critical but currently has 0 F1 score.
- [ ] **Class-Balanced Sampling**: Modify `data/generator.py` to oversample G5 patches during training.
- [ ] **Weighted Loss**: Tune `alpha` in Focal Loss specifically for G5.
- [ ] **Patch Selection**: Ensure training batches contain a minimum number of G5 samples.

## 4. Fine-tuning Strategy (Priority: Medium)
*Rationale*: Compare linear evaluation (frozen encoder) vs. full fine-tuning.
- [ ] **Linear Probing**: Freeze encoder, train only the head (standard SSL benchmark).
- [ ] **End-to-End**: Unfreeze encoder (with lower LR) after linear probing.
- [ ] **Layer-wise Unfreezing**: Unfreeze top N layers gradually.

## 5. Robust Validation (Priority: Medium)
*Rationale*: A single train/test split is insufficient for research claims.
- [ ] **Cross-Validation**: Implement 5-fold cross-validation.
- [ ] **Statistical Significance**: Compute mean and standard deviation of metrics across folds.

## 6. Visualization & Quality Analysis (Priority: Medium)
*Rationale*: Visual proof of feature separation.
- [ ] **t-SNE / UMAP**: Visualize the latent space (Is G5 separated from G4?).
- [ ] **Attention Maps / Grad-CAM**: Visualize which parts of the patch determining the grade.
