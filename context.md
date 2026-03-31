# Project Context: Self-Supervised Learning for Prostate Cancer Grading

## Overview
This project focuses on automating the Gleason grading of prostate cancer histology images (SICAPv2 dataset) using Self-Supervised Learning (SSL) to overcome the limitation of scarce labeled data.

## Methodology
The project compares four approaches:
1.  **Baseline (Supervised)**: A CNN trained from scratch on labeled data.
2.  **Convolutional Autoencoder (CAE)**: Image reconstruction as pretext task.
3.  **SimCLR**: NT-Xent contrastive loss on dual augmented views.
4.  **MoCo v2**: Momentum Contrast with ResNet50 + PANDA data expansion.

## Workflow
1.  **Data Prep**: `data/setup.py` — converts SICAPv2 partition Excel files to CSV, creates `TrainSplit.csv` and `Val.csv`, extracts PANDA-PLUS patches, and builds `Pretrain_Manifest.csv`.
2.  **Pretraining**: `training/` scripts train SSL encoders (CAE, SimCLR, MoCo v2).
3.  **Fine-tuning**: `training/` scripts train classifier head on top of frozen/unfrozen encoder.
4.  **Evaluation**: `evaluation/` scripts generate metrics (F1, Accuracy, Confusion Matrix, t-SNE).

## Current Project Phase: **PHASE 4 - ABLATION STUDIES PREP**

### Phase 1 Status: ✅ COMPLETED (March 29, 2026)
**All 4 models successfully trained in Google Colab with Google Drive storage:**

| Model | Test Accuracy | Cohen's κ | Macro F1 | Key Findings |
|-------|--------------|-----------|----------|--------------|
| **MoCo-SSL** | **59.9%** | **0.569** | 0.516 | 🏆 Phase 1 winner: +4.4% vs baseline, 37.9% G5 recall |
| **Baseline** | 57.4% | 0.403 | 0.343 | Supervised learning baseline |
| **CAE-SSL** | 47.3% | 0.158 | 0.350 | Best G3 recall (41.6%) |
| **SimCLR-SSL** | 33.4% | 0.020 | 0.233 | ⚠️ Catastrophic in Phase 1 — recovered in Phase 2 |

*Phase 1 evaluated on different test set sizes (Baseline/CAE/SimCLR: 3,513 samples; MoCo: 2,487). Phase 2 normalized all to 2,487.*

### Phase 2 Status: ✅ COMPLETED (March 29, 2026)
**Objective:** Evaluate Stage 1 (frozen encoder) vs Stage 2 (end-to-end) for each model on the same test set.

**Results — all 6 checkpoints evaluated on 2,487 samples:**

| Model | Stage 1 Acc | Stage 2 Acc | Acc Δ | Stage 1 κ | Stage 2 κ | κ Δ |
|-------|:-----------:|:-----------:|:-----:|:---------:|:---------:|:---:|
| **SimCLR-SSL** | 35.5% | **67.2%** | +31.6% | 0.036 | **0.509** | **+0.474** |
| **CAE-SSL** | 42.1% | 64.9% | +22.8% | 0.121 | 0.489 | +0.368 |
| **MoCo-SSL** | val_loss=0.265* | 59.8% | — | — | 0.414 | — |
| Baseline | N/A | 40.0% | N/A | N/A | 0.108 | N/A |

*MoCo Stage 1 not separately checkpointed; Stage 2 selected via validation loss (0.265 → 0.121, −54.3%).*

**Stage 2 Per-Class Recall:**

| Model | NC | G3 | G4 | G5 |
|-------|:---:|:---:|:---:|:---:|
| SimCLR Stage 2 | 95.8% | 15.2% | 87.8% | 30.3% |
| CAE Stage 2 | 91.1% | **48.6%** | 68.8% | 6.6% |
| MoCo Stage 2 | 90.5% | 17.9% | 69.5% | **37.9%** |
| Baseline | 62.0% | 2.9% | 56.4% | 0.5% |

**Phase 2 Key Findings:**
1. **End-to-end fine-tuning is essential** — Stage 1 alone is near-useless (CAE: κ 0.121 → 0.489; SimCLR: κ 0.036 → 0.509)
2. **SimCLR Stage 2 now leads** at κ=0.509 / 67.2% acc — reversed its Phase 1 collapse
3. **CAE Stage 2 best at G3 recall** (48.6%) — consistent with Phase 1 finding
4. **G5 recall is still the critical gap** — 6.6% (CAE) to 37.9% (MoCo) in Stage 2
5. **Output artifacts saved:** `phase2_comparison/stage_comparison_table.csv` + `stage_comparison_detailed_results.json`

**Technical issues resolved during Phase 2:**
- ✅ Keras 3.x `quantization_config` bug fixed via zip-patch loader (`load_model_compat`) — strips the key from `config.json` in the `.keras` archive before loading, stateless and safe to call multiple times
- ✅ One-hot label mismatch fixed — class order `['NC','G3','G5','G4']` (training order) enforced, not alphabetical
- ✅ Generator cycling bug fixed — labels now read directly from CSV, `model.predict` output trimmed to `n_samples`

**Dataset Structure:**
-   **Dataset**: SICAPv2 (Patches, 10x magnification) + PANDA-PLUS baseline parquet.
-   **Classes**: NC (Non-Cancerous), G3, G4, G5.
-   **Class Imbalance**: Highly imbalanced (NC > G4 > G3 > G5).
-   **Structure**: Modular directories (`data/`, `models/`, `training/`, `evaluation/`, `utils/`, `docs/`).

---

## Remaining Work Queue

### Phase 3: Class Imbalance Interventions (COMPLETED — MIXED OUTCOME)
**Priority:** 🔴 HIGH  
**Target:** Improve G5 recall from 6–38% → 40%+, G3 recall from 15–49% → 50%+  
**Compute cost:** Fine-tuning only (~3–4 hrs Colab). Pretraining not touched.  

**Phase 3 final status (March 31, 2026):**
- ✅ `run_phase3_finetune.ipynb` is now self-contained for Colab; it no longer depends on older notebooks for Phase 3 evaluation
- ✅ Dedicated Phase 3 evaluator added: `phase3_comparison/evaluate_phase3_comparison.py`
- ✅ Optional repo-refresh cell added to the notebook so pushed fixes can be pulled into a live Colab session without re-extracting the dataset
- ✅ CAE Phase 3 fine-tuning completed and evaluated
	- Result: Stage 2 regressed vs Phase 2 overall (55.8% acc, κ=0.341); small G5 lift to 8.1%, but G3 recall dropped sharply to 15.0%
- ✅ SimCLR Phase 3 fine-tuning completed successfully
	- Run dir: `runs/simclr_phase3_20260331_052731`
	- Final blocker resolved by preferring the canonical `encoder_weights.weights.h5` path in `training/simclr/finetune_simclr.py`
- ✅ SimCLR Phase 3 evaluation completed
	- Result: Stage 2 regressed significantly vs Phase 2 (57.8% acc, κ=0.358); minority-class collapse to G5 recall 0.0% and G3 recall 0.6%
- ✅ MoCo Phase 3 fine-tuning and evaluation completed
	- Result: clear improvement over Phase 2 (60.9% acc, κ=0.439, macro F1=0.546, G5 recall=56.1%)
- ✅ Final Phase 3 comparison saved to `phase3_comparison/stage_comparison_table.csv` and `phase3_comparison/stage_comparison_detailed_results.json`

**Phase 3 interpretation:**
- The shared intervention (existing focal loss + capped class weights in Stage 1/2) is **model-dependent**
- **MoCo benefits strongly** and should keep the Phase 3 checkpoint
- **CAE and SimCLR do not benefit overall**; their stronger checkpoints remain the Phase 2 Stage 2 versions
- Recommended locked comparison set for publication / next steps:
  - **CAE:** Phase 2 Stage 2
  - **SimCLR:** Phase 2 Stage 2
  - **MoCo:** Phase 3 Stage 2
  - **Baseline:** existing single-stage model

**✅ Step 1 — Capped class-weighted loss — IMPLEMENTED (all 3 fine-tuning scripts)**  
Changed all three fine-tuning scripts (`finetune_cae.py`, `finetune_simclr.py`, `finetune_moco.py`):
- Class weights computed via `sklearn compute_class_weight('balanced')` (already existed)
- **New:** Weights capped at 3.0: `{k: min(v, 3.0) for k, v in ...}` — prevents the training collapse (accuracy → 15%) seen with uncapped weights
- **New:** `class_weight=class_weights_dict` added to Stage 1 **and** Stage 2 `model.fit()` in all scripts
- Strategy: Focal loss (existing) + capped class weights = complementary; focal loss downweights easy examples, class weights upweight minority gradients

**🔴 Step 2 — Focal loss (γ=2) — already implemented (no change needed)**  
All three scripts already use `focal_loss(alpha=0.5, gamma=2.0)` for Stage 1.

**🔴 Step 3 — Oversampled batches — NOT YET (next low-cost follow-up if needed)**  
- Modify `DataGenerator` to guarantee minimum G5 samples per batch
- This is now primarily relevant for **CAE / SimCLR recovery**, not MoCo

**Current Phase 3 Colab workflow:**  
```bash
# CAE (re-finetune only, ~2 hrs)
python training/cae/finetune_cae.py --epochs_stage1=50 --epochs_stage2=30

# SimCLR (re-finetune only, ~2 hrs)
python training/simclr/finetune_simclr.py --epochs_stage1=50 --epochs_stage2=30

# MoCo (re-finetune only, ~1.5 hrs)
python training/moco/finetune_moco.py --epochs_stage1=30 --epochs_stage2=10
```
The Phase 3 evaluation has now been run; outputs are stored in `phase3_comparison/`, while `phase2_comparison/` remains the canonical source for the best CAE/SimCLR checkpoints.

### Phase 4: Ablation Studies
**Priority:** 🟠 HIGH (required for publication)  
**Status:** ⏸ Full rerun-based Phase 4 deferred due to time / compute limits  
**Recommended baseline set for ablations:** Phase 2 CAE, Phase 2 SimCLR, Phase 3 MoCo, Baseline  
**Original full tasks (deferred):**
- SimCLR temperature sweep: τ ∈ {0.1, 0.3, 0.5} in NT-Xent loss
- Encoder architecture: current custom CNN vs. ResNet-18 vs. ResNet-50
- Projection head depth: linear vs. 2-layer MLP vs. 3-layer MLP
- Augmentation strength: color jitter intensity, crop scale ranges

**Reduced Phase 4 plan (next practical step):**
- Use **existing completed ablations** already present in the project:
	- Phase 2 as a training-stage ablation (Stage 1 vs Stage 2)
	- Phase 3 as a loss/intervention ablation (standard fine-tuning vs class-weighted intervention)
- Prefer **inference-only analyses** over retraining:
	- compare confusion patterns, κ, macro F1, and per-class recall for the locked checkpoints
	- optionally run light test-time perturbation checks on saved checkpoints
- Add **representation analysis** from saved checkpoints:
	- t-SNE / embedding plots for Baseline, CAE Phase 2, SimCLR Phase 2, and MoCo Phase 3
- If any small compute budget remains, do **one micro-ablation only** (preferably a short SimCLR temperature pilot) rather than a full matrix

**Working recommendation:**
- Treat the reduced ablation package as the realistic next step
- Treat the full retraining-based Phase 4 matrix as future work

### Phase 5: Visualization
**Priority:** 🟢 MEDIUM  
**Tasks:**
1. t-SNE plots of encoder embeddings for all 4 models — confirm G5/G3 cluster separation
2. Grad-CAM attention maps to show patch regions driving grading decisions
3. Template exists: `evaluation/moco/tsne_moco.py`

### Phase 6: Cross-Validation
**Priority:** 🟢 MEDIUM  
**Tasks:**
1. 5-fold cross-validation on best locked model / final candidate (currently still SimCLR Phase 2 Stage 2 by overall κ)
2. Report mean ± std — required to validate results are not seed-dependent

---

## Technical Debt & Known Issues

### Class Order — Critical Invariant
- **Constraint:** ALL scripts must use `CLASS_NAMES = ['NC', 'G3', 'G5', 'G4']`
- **Never use alphabetical:** `['NC','G3','G4','G5']` causes G4/G5 label swap, destroys all metrics
- **Root cause:** Training CSVs have columns in this non-alphabetical order, DataGenerator reads them positionally

### Keras Version Hell — SOLVED
- **Issue:** Models saved with Keras 3.x (standalone) cannot be loaded with tf.keras 2.x (bundled in TensorFlow)
- **Symptom:** `TypeError: Error when deserializing class 'Dense' using config={..., 'quantization_config': None}`
- **Solution:** `load_model_compat()` — opens `.keras` zip, recursively strips `quantization_config` key from config.json, writes patched temp file, loads it. **Do not remove this function from notebooks.**

### Test Set Size — RESOLVED (2487 is canonical)
- **Issue was:** MoCo Phase 1 used 2,487 samples vs 3,513 for others
- **Resolution:** Phase 2 re-evaluated all models on the same 2,487-sample `Test.csv`. This is the authoritative comparison. Phase 1 numbers are legacy; use Phase 2 numbers for publication.

### Minority Class Weakness — Open Problem (Post-Phase 3)
- **Issue:** G5 recall 6–38% across all models (class size: ~130 samples vs ~900 NC)
- **Implemented fix:** Fine-tuning scripts now apply focal loss + capped class weights (`class_weight` dict in `model.fit()`) in both Stage 1 and Stage 2
- **Current status:** Final Phase 3 results are now known:
	- **MoCo improved** substantially on G5 recall (37.9% → 56.1%)
	- **CAE improved only slightly on G5** (6.6% → 8.1%) but lost much of its G3 strength
	- **SimCLR regressed badly** on minority classes under the same setup
- **Implication:** future minority-class work should be **model-specific**, not a shared one-size-fits-all weighting recipe

---

## Archived Notes

### MoCo v2 Implementation (2026-02-24)
Successfully implemented MoCo v2 with ResNet50 encoder, MLP projection head, FIFO queue (K=4096), InfoNCE loss. Key achievements:
- Hybrid CPU/GPU pipeline avoiding GIL deadlocks
- AMP support with float16/float32 casting
- Cosine LR schedule with warmup
- Full pretrain → fine-tune → evaluate pipeline

### Review Findings (2026-03-17)
- Protocol cleanup complete: TrainSplit.csv/Val.csv/Test.csv aligned
- All 4 models (Baseline, CAE, SimCLR, MoCo) have end-to-end pipelines
- Environment: macOS Apple Silicon with CPU-only TensorFlow in `./venv/`
- Checkpoint standardization partially complete (MoCo done, CAE/SimCLR legacy paths remain)
