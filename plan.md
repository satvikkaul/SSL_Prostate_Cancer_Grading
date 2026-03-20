# Research Enhancement Plan

**Goal**: Prepare the project for publication (Conference/Journal) by enhancing model performance and conducting rigorous analysis.

**Target**: 10-12 page paper, emphasizing SSL efficacy on imbalanced medical data.

## 0. Protocol & Evaluation Fixes (Priority: Immediate)
*Rationale*: Current training scripts reuse `Test.csv` for validation/model selection, and MoCo manifest construction includes SICAP test images. This must be corrected before any new metrics are treated as reliable.
- [x] **Separate train / validation / test correctly**:
    - `TrainSplit.csv` and `Val.csv` are now used in the main baseline, CAE, SimCLR, and MoCo training paths.
    - `Test.csv` is reserved for evaluation scripts.
- [x] **Fix MoCo pretraining protocol**:
    - MoCo pretraining manifest is built from non-test SICAP data plus PANDA.
    - SICAP test images are excluded from the MoCo pretraining manifest.
- [x] **Regenerate comparison code from clean outputs**:
    - Shared comparison now reads saved metrics files instead of hard-coded CAE values.
- [ ] **Re-run the non-MoCo pipelines under the cleaned protocol on final checkpoints**:
    - Regenerate baseline, CAE, and SimCLR evaluation outputs in the standardized layout before final comparison.

## 0.5. Complete MoCo Integration (Priority: High)
*Rationale*: MoCo code exists, but the full downstream experiment is not yet complete.
- [x] **Finish MoCo implementation quality**:
    - MoCo pretraining now runs on the local macOS Metal setup without the earlier graph stall.
    - Downstream checkpoint selection now exports a single `best_moco_overall.keras` artifact.
- [x] **Add downstream MoCo pipeline**:
    - MoCo fine-tuning and evaluation scripts now exist and run end-to-end.
- [x] **Fix Stage 1 -> Stage 2 handoff in MoCo fine-tuning**:
    - Stage 2 now reloads the best Stage 1 checkpoint before unfreezing the encoder.
- [ ] **Run the first full MoCo experiment on Colab/Drive**:
    - Pretrain longer than the local pilot run, then fine-tune and evaluate from the resulting checkpoint.
- [ ] **Integrate MoCo into final comparison outputs**:
    - Add MoCo to the shared comparison once at least one additional model has fresh saved metrics.

## 0.75. Checkpoint & Artifact Cleanup (Priority: Medium)
*Rationale*: The project now follows the corrected train/validation/test protocol and includes a MoCo downstream path, but some legacy CAE/SimCLR scripts still rely on hard-coded checkpoint names or manual artifact assumptions. This should be cleaned up before the final experiment sweep.
- [ ] **Make checkpoint discovery automatic**:
    - Remove hard-coded CAE weights paths from fine-tuning scripts.
    - Add automatic discovery or CLI selection for the latest valid CAE and SimCLR checkpoints.
- [ ] **Standardize output locations**:
    - Keep CAE, SimCLR, and MoCo classifier outputs/evaluation artifacts in consistent model-specific folders.
    - Minimize legacy duplicate save paths once the new layout is validated.
- [ ] **Fix CAE training protocol drift**:
    - Restore true Stage 1 head-only freezing behavior in `training/cae/finetune_cae.py`.
    - Remove duplicated legacy save targets from CAE training/evaluation once the model-specific paths are validated.
- [ ] **Harden shared comparison execution**:
    - Make the comparison flow fail more gracefully when fewer than two evaluation outputs are present.

## 1. Advanced SSL Expansion (Priority: High)
*Rationale*: MoCo v2 is now implemented, but the paper may still benefit from one more SSL comparison if time allows.
- [x] **Implement MoCo v2**:
    - `models/moco_model.py`, `training/moco/pretrain_moco.py`, `training/moco/finetune_moco.py`, and `evaluation/moco/eval_moco.py` are now in place.
    - Remaining work is experiment execution and comparison, not base implementation.
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
