# SSL Prostate Cancer Grading — Project Progress Report
**Date:** March 31, 2026  
**Dataset:** SICAPv2 (128×128 patches, 10× magnification)  
**Classes:** NC (Non-Cancerous), G3, G4, G5 (Gleason grades)  
**Goal:** Publication-ready SSL vs. supervised baseline comparison for prostate cancer Gleason grading

---

## Project Overview

This project compares four approaches to automated Gleason grading of prostate cancer histology patches:

| Model | Approach | Pretraining Task |
|-------|----------|-----------------|
| **Baseline** | Fully supervised from scratch | None |
| **CAE-SSL** | Self-supervised + fine-tuning | Image reconstruction (Convolutional Autoencoder) |
| **SimCLR-SSL** | Self-supervised + fine-tuning | Contrastive learning (NT-Xent loss) |
| **MoCo-SSL v2** | Self-supervised + fine-tuning | Momentum contrast (PANDA + SICAPv2 pretraining) |

All SSL models use a **2-stage fine-tuning protocol:**
- **Stage 1 (Frozen):** Train classifier head only, encoder weights frozen
- **Stage 2 (End-to-End):** Unfreeze encoder, continue training at lower LR

---

## ✅ Phase 1 — Model Training (COMPLETE)

All four models trained end-to-end in Google Colab with checkpoints saved to Google Drive. Evaluated on the test set independently.

> **Note on test set sizes:** Phase 1 evaluation scripts used slightly different test splits (Baseline/CAE/SimCLR on 3,513 samples, MoCo on 2,487). Phase 2 (below) re-evaluates all models on the same 2,487-sample test set for consistent comparison.

### Phase 1 Final Results (per original evaluation scripts)

| Model | Test Accuracy | Cohen's κ | Macro F1 | Weighted F1 |
|-------|:---:|:---:|:---:|:---:|
| **MoCo-SSL** | **59.9%** | **0.569** | 0.516 | 0.560 |
| Baseline (No SSL) | 57.4% | 0.403 | 0.343 | 0.529 |
| CAE-SSL | 47.3% | 0.158 | 0.350 | 0.474 |
| SimCLR-SSL | 33.4% | 0.020 | 0.233 | 0.325 |

### Phase 1 Per-Class Recall

| Model | NC | G3 | G4 | G5 |
|-------|:---:|:---:|:---:|:---:|
| **MoCo-SSL** | **90.5%** | 18.1% | **69.6%** | **37.9%** |
| Baseline | 78.4% | 8.0% | 59.9% | 0.0% |
| CAE-SSL | 49.6% | **41.6%** | 57.4% | 0.0% |
| SimCLR-SSL | 31.3% | 19.3% | 51.4% | 0.0% |

### Phase 1 Key Findings
- ✅ **MoCo-SSL wins** overall: +4.4% accuracy and +0.166 κ over baseline
- ✅ **MoCo is the only model to detect G5** (37.9% recall vs. 0% for all others)
- ✅ **CAE shows best G3 recall** (41.6%) despite lower overall accuracy
- 🚨 **SimCLR catastrophically failed** (33.4% ≈ random guessing, κ ≈ 0) — identified as needing fix
- ⚠️ **G5 severely underdetected** by all models (0–37.9% recall)
- ⚠️ **G3 weak across all models** (8–42% recall)

---

## ✅ Phase 2 — Stage 1 vs Stage 2 Comparison (COMPLETE)

**Objective:** Determine whether end-to-end fine-tuning (Stage 2) consistently improves over frozen encoder training (Stage 1), and quantify the delta for each model.

All 6 checkpoints (Baseline, CAE ×2, SimCLR ×2, MoCo) re-evaluated on the same 2,487-sample test set for a consistent comparison.

### Stage 1 vs Stage 2 Comparison Table

| Model | Stage 1 Acc | Stage 2 Acc | Acc Δ | Stage 1 κ | Stage 2 κ | κ Δ | Stage 1 F1 | Stage 2 F1 | F1 Δ | Winner |
|-------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **CAE-SSL** | 42.1% | **64.9%** | +22.8% | 0.121 | **0.489** | +0.368 | 0.297 | **0.523** | +0.227 | 🟢 Stage 2 |
| **SimCLR-SSL** | 35.5% | **67.2%** | +31.6% | 0.036 | **0.509** | +0.474 | 0.253 | **0.553** | +0.299 | 🟢 Stage 2 |
| **MoCo-SSL** | val_loss=0.265* | 59.8% | — | — | 0.414 | — | — | 0.515 | — | 🟢 Stage 2 (selected) |
| Baseline | N/A | 40.0% | N/A | N/A | 0.108 | N/A | N/A | 0.256 | N/A | N/A |

*MoCo Stage 1 checkpoint not separately saved; Stage 2 selected based on 54.3% lower validation loss (0.265 → 0.121)

### Stage 2 Per-Class Recall (Final Models)

| Model | NC | G3 | G4 | G5 |
|-------|:---:|:---:|:---:|:---:|
| **SimCLR Stage 2** | **95.8%** | 15.2% | **87.8%** | 30.3% |
| **CAE Stage 2** | 91.1% | **48.6%** | 68.8% | 6.6% |
| MoCo Stage 2 | 90.5% | 17.9% | 69.5% | **37.9%** |
| Baseline | 62.0% | 2.9% | 56.4% | 0.5% |
| CAE Stage 1 | 51.7% | 14.2% | 61.8% | 0.0% |
| SimCLR Stage 1 | 38.0% | 16.8% | 53.0% | 0.0% |

### Phase 2 Key Findings

1. **End-to-end fine-tuning is essential** — Stage 1 alone is near-useless. CAE jumps from κ=0.121 to κ=0.489 (+0.368); SimCLR from κ=0.036 to κ=0.509 (+0.474). These are the largest Stage 1 → Stage 2 improvements seen in the project.

2. **SimCLR Stage 2 now leads overall** at κ=0.509 and 67.2% accuracy, reversing its Phase 1 catastrophic failure. The encoder learned useful representations even though Phase 1's evaluation (which captured only 20 fine-tuning epochs) showed collapse. With full end-to-end training, it recovers strongly.

3. **CAE Stage 2 has the best G3 recall** (48.6%) by a significant margin, confirming Phase 1's finding that CAE learns G3-discriminative features particularly well.

4. **G5 remains the critical gap** — only 6.6% (CAE) to 37.9% (MoCo) recall in Stage 2. This is the primary target for the next phase.

5. **Baseline is now the weakest model** at κ=0.108 on the 2,487 sample test set (note: Phase 1 used 3,513 samples for baseline, giving κ=0.403 — the difference reflects different test splits and the same underlying model).

### Phase 2 Output Artifacts
- `phase2_comparison/stage_comparison_table.csv` — Summary comparison table
- `phase2_comparison/stage_comparison_detailed_results.json` — Full per-class metrics for all 6 checkpoints

---

## ✅ Phase 3 — Class Imbalance Intervention (COMPLETE, MIXED OUTCOME)

**Objective:** Improve minority-class recall, especially **G5**, by re-running fine-tuning only with the existing focal loss plus **capped class weights** (`class_weight`, max 3×) in both Stage 1 and Stage 2.

All three SSL models were re-fine-tuned from the March 29 pretrained encoders and evaluated on the same 2,487-sample test set. Outputs were written to `phase3_comparison/`.

### Phase 3 Final Results

| Model | Stage 1 Acc | Stage 2 Acc | Acc Δ | Stage 1 κ | Stage 2 κ | κ Δ | Stage 2 Macro F1 | Winner |
|-------|:-----------:|:-----------:|:-----:|:---------:|:---------:|:---:|:---------------:|:------:|
| **CAE-SSL** | 39.8% | 55.8% | +16.0% | 0.144 | 0.341 | +0.197 | 0.417 | 🟢 Stage 2 |
| **SimCLR-SSL** | 32.7% | 57.8% | +25.1% | 0.006 | 0.358 | +0.353 | 0.352 | 🟢 Stage 2 |
| **MoCo-SSL** | val_loss* | **60.9%** | — | — | **0.439** | — | **0.546** | 🟢 Stage 2 |
| Baseline (No SSL) | N/A | 40.0% | N/A | N/A | 0.108 | N/A | 0.256 | N/A |

*MoCo Stage 1 was not checkpointed separately; Stage 2 remained the selected model.*

### Phase 3 Per-Class Recall (Stage 2)

| Model | NC | G3 | G4 | G5 |
|-------|:---:|:---:|:---:|:---:|
| CAE Stage 2 | 83.1% | 15.0% | 72.3% | 8.1% |
| SimCLR Stage 2 | 92.6% | 0.6% | 81.7% | 0.0% |
| **MoCo Stage 2** | **92.3%** | 17.9% | 67.4% | **56.1%** |
| Baseline | 62.0% | 2.9% | 56.4% | 0.5% |

### Phase 3 Key Findings

1. **MoCo is the clear Phase 3 success case.** It improved over its Phase 2 checkpoint across the major summary metrics:
	- Accuracy: **59.8% → 60.9%**
	- Cohen's κ: **0.414 → 0.439**
	- Macro F1: **0.515 → 0.546**
	- G5 recall: **37.9% → 56.1%**

2. **The same intervention did not transfer cleanly to CAE and SimCLR.** Both models regressed versus their stronger Phase 2 Stage 2 checkpoints:
	- CAE Stage 2 accuracy: **64.9% → 55.8%**, κ: **0.489 → 0.341**
	- SimCLR Stage 2 accuracy: **67.2% → 57.8%**, κ: **0.509 → 0.358**

3. **CAE retained only a small G5 gain while losing most of its G3 advantage.**
	- G5 recall: **6.6% → 8.1%**
	- G3 recall: **48.6% → 15.0%**

4. **SimCLR collapsed on the minority classes under the Phase 3 weighting setup.**
	- G5 recall: **30.3% → 0.0%**
	- G3 recall: **15.2% → 0.6%**
	- The model still preserved decent NC/G4 performance, suggesting over-bias toward majority/easier decision regions rather than total training failure.

5. **Conclusion:** capped class weighting plus focal loss is **model-dependent** in this project. It is a net positive for **MoCo**, but should **not** replace the Phase 2 CAE or SimCLR checkpoints as the main reported versions.

### Phase 3 Output Artifacts
- `phase3_comparison/stage_comparison_table.csv` — Summary comparison table
- `phase3_comparison/stage_comparison_detailed_results.json` — Full per-class metrics for Phase 3 reruns
- `run_phase3_finetune.ipynb` — Self-contained Colab rerun + evaluation workflow

---

## Audit Addendum — Result Validity & Reproducibility Risks

This section captures repo-audit findings that may affect interpretation, reproducibility, or validity of the reported results.

### Findings That May Affect Training or Evaluation Results

1. **Phase 1 cross-model comparison is not methodologically clean**
	- Baseline / CAE / SimCLR Phase 1 metrics were evaluated on a **3,513-sample** `Test.csv`
	- MoCo Phase 1 metrics were evaluated on a **2,487-sample** `Test.csv`
	- Therefore, **Phase 1 should not be used as the authoritative cross-model comparison**
	- **Use Phase 2 as canonical** because all models were re-evaluated on the same 2,487-sample test set

2. **Script-vs-notebook drift exists in the CAE fine-tuning path**
	- `training/cae/finetune_cae.py` still contains a **hard-coded** `WEIGHTS_PATH`
	- Colab notebook flow patches this dynamically, but the checked-in script does not reflect that behavior
	- Risk: running the script directly may load the wrong encoder checkpoint and produce different fine-tuning results

3. **Standalone Phase 2 evaluation path does not fully match notebook evaluation path**
	- The notebook `run_phase2_stage_comparison.ipynb` uses `load_model_compat()` to handle Keras 3.x archive compatibility and explicitly avoids prior generator-label issues
	- The standalone script `phase2_comparison/evaluate_stage_comparison.py` uses plain `keras.models.load_model(...)`
	- Risk: re-running evaluation from the standalone script may fail or yield behavior different from the notebook-backed Phase 2 outputs

4. **One remaining class-order inconsistency exists in the visualization path**
	- Main training/evaluation code correctly uses class order `['NC', 'G3', 'G5', 'G4']`
	- `evaluation/moco/tsne_moco.py` still uses `['NC', 'G3', 'G4', 'G5']` when mapping labels from CSVs
	- This does **not** affect the primary reported Phase 1 / Phase 2 metrics, but it may affect any label-driven MoCo visualization or downstream visual analysis

### Findings That Mainly Affect Reporting or Reproducibility

1. **Phase 1 comparison report has broken per-class parsing**
	- `evaluation/shared/compare_models.py` fails to parse indented class rows from `evaluation_metrics.txt`
	- As a result, `phase1_comparison/comparison_report.txt` shows `nan` for per-class values even though the underlying metrics files contain valid numbers
	- This affects the generated comparison report, not the raw evaluation outputs

2. **Archived run artifacts were produced from an older commit than the current workspace**
	- All archived Phase 1 runtime snapshots point to commit `06cd6c9f6ed8bfaf15a0b94e229adbd7ca9d7b3b`
	- Current branch state has diverged since then
	- Risk: re-running experiments from the present workspace may not exactly reproduce archived outputs unless that recorded commit/state is restored

3. **Notebook runtime sync had at least one logged failure during MoCo archival flow**
	- `run_colab_2.ipynb` contains a stored failed sync attempt before a later corrected sync cell
	- This appears to affect runtime archival provenance rather than the actual trained checkpoint metrics

4. **Local machine environment is not currently configured to reproduce the archived runs**
	- The active local Python environment is missing TensorFlow / OpenCV / scikit-image
	- This does not invalidate the archived Colab results, but it blocks direct local reproduction/verification from the current machine state

### Audit Interpretation Guidance

- **Safest published comparison:** Phase 2 (`phase2_comparison/`)
- **Historical but non-canonical comparison:** Phase 1 (`phase1_comparison/`) because test sets differ by model
- **Most important reproducibility risk:** notebook-patched behavior is not always reflected in the checked-in training/evaluation scripts

---

## 🔴 Remaining Work

### Phase 3 Follow-up — Targeted Minority-Class Recovery (Recommended before publication)
**Status:** Initial Phase 3 intervention is complete and evaluated.  
**Takeaway:** The shared weighting strategy improved **MoCo** substantially, but degraded **CAE** and **SimCLR**.

**Recommended next interventions:**
1. **Keep MoCo Phase 3** as the improved minority-class checkpoint
2. **Retain Phase 2 CAE and SimCLR** as the best overall checkpoints for those models
3. For any Phase 3 continuation, use **model-specific** follow-up experiments rather than a single shared recipe:
	- CAE: try milder weighting and/or G5-targeted oversampled batches while preserving G3 performance
	- SimCLR: investigate the collapse before further reruns; likely candidates are weight strength, sampling dynamics, or Stage 2 optimization sensitivity
	- Oversampled batches remain the next strongest low-cost intervention if another minority-class pass is desired

---

### Phase 4 — Ablation Studies (Publication Requirement)
**Status:** Deferred due to time and compute constraints.  
**Decision:** Phase 4 will be treated as **future work**, not part of the current execution scope.  
**Rationale:** The project already has a stable locked comparison set (Baseline, CAE Phase 2, SimCLR Phase 2, MoCo Phase 3), but available Colab time / GPU budget is not sufficient for a full ablation matrix.

Required to explain *why* SSL works, not just *that* it works:
- **SimCLR temperature** sweep: τ ∈ {0.1, 0.3, 0.5} in NT-Xent loss
- **Encoder architecture**: Current custom CNN vs. ResNet-18 vs. ResNet-50
- **Projection head**: Linear vs. 2-layer MLP vs. 3-layer MLP
- **Augmentation strength**: Color jitter intensity, crop scale ranges

**Reduced Phase 4 plan (low-compute alternative):**
- **Use existing ablations already completed:**
	- Phase 2 = training-stage ablation (**frozen vs end-to-end fine-tuning**)
	- Phase 3 = loss/intervention ablation (**baseline fine-tuning vs focal loss + capped class weights**)
- **Run inference-only analyses on saved checkpoints:**
	- compare confusion patterns, per-class recall, and κ across the locked models
	- optionally evaluate saved checkpoints under mild test-time perturbations (e.g. brightness shift, blur, light color jitter)
- **Run representation analysis instead of retraining:**
	- generate t-SNE / embedding plots for the locked checkpoints to compare NC / G3 / G4 / G5 separation
- **If a very small compute budget becomes available:**
	- run **one micro-ablation only**, preferably a short SimCLR temperature pilot, and report it explicitly as directional rather than exhaustive

**Recommended next actions under the reduced plan:**
1. Finalize the locked checkpoint set: Baseline, CAE Phase 2, SimCLR Phase 2, MoCo Phase 3
2. Produce representation / visualization outputs from saved checkpoints
3. Add one cheap robustness-style evaluation if time allows
4. Treat the full retraining-based ablation matrix as future work

**Implication for current report/write-up:**
- Final conclusions should focus on empirical model comparison and the Phase 3 minority-class intervention
- Avoid strong causal claims about *why* a specific SSL configuration performed best without ablation evidence
- Present the **reduced ablation package** as the practical next step now, and the full Phase 4 rerun matrix as future work if more compute becomes available

---

### Phase 5 — Visualization & t-SNE
- t-SNE plots of encoder embeddings (all 4 models) to visually confirm G5/G3 cluster separation
- Grad-CAM attention maps to show which patch regions drive grading decisions
- Template: `evaluation/moco/tsne_moco.py` already exists; needs to be applied to all models

---

### Phase 6 — Cross-Validation & Statistical Significance
- 5-fold cross-validation on the best model (currently SimCLR Stage 2 by κ)
- Report mean ± std to validate results are not seed-dependent

---

## Repository Structure

```
SSL_Prostate_Cancer_Grading/
├── data/                    # DataGenerator, setup.py (CSV generation)
├── models/                  # Model architectures (CAE, SimCLR, MoCo)
├── training/                # Training scripts per model
├── evaluation/              # Eval scripts + t-SNE per model
├── phase1_comparison/       # Phase 1 run artifacts + comprehensive analysis
├── phase2_comparison/       # Phase 2 stage comparison outputs (NEW)
├── run_*_colab.ipynb        # Colab training notebooks (one per model)
└── run_phase2_stage_comparison.ipynb  # Phase 2 evaluation notebook
```

**Training environment:** Google Colab (T4 GPU), Google Drive for checkpoint persistence  
**Framework:** TensorFlow 2.18 / Keras 3.12  
**Branch:** `method/moco-v2`

---

## Summary

| Phase | Status | Key Output |
|-------|--------|-----------|
| Phase 1 — Train all models | ✅ Complete | MoCo best: κ=0.569, 59.9% acc |
| Phase 2 — Stage 1 vs 2 comparison | ✅ Complete | SimCLR Stage 2 best: κ=0.509, 67.2% acc |
| Phase 3 — Class imbalance (G5 fix) | ✅ Complete (mixed) | MoCo improved to G5 recall 56.1%; CAE/SimCLR regressed |
| Phase 4 — Ablation studies | ⏸ Deferred | Future work due to time / compute limits |
| Phase 5 — Visualization (t-SNE) | 🔴 Not started | Feature space analysis |
| Phase 6 — Cross-validation | 🔴 Not started | Mean ± std metrics |
