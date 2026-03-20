# Team Handoff Plan

## Purpose

This document is for the remaining team work after the MoCo implementation handoff. The goal is to take the current codebase from "pipeline complete" to "paper-ready comparison study" by finishing the remaining experiments, ablations, analysis, and reporting.

## Current Project State

- Baseline, CAE, SimCLR, and MoCo pipelines exist in code.
- MoCo now has a full pretrain -> fine-tune -> evaluate flow.
- The dataset split protocol is cleaned up:
  - `TrainSplit.csv` for training
  - `Val.csv` for checkpoint selection
  - `Test.csv` for final evaluation
- A meaningful Colab MoCo run has already been completed and saved.
- Remaining work is now mostly experiment execution, comparison, ablation, and cleanup.

## Feedback Priorities To Preserve

The formal feedback highlighted these as the most important remaining items:

- add at least one momentum-based SSL method beyond SimCLR:
  - this is now satisfied by MoCo, so the team must make sure MoCo is included in the final comparison tables
- resolve minority-class failure experimentally:
  - especially `G5`, and now also weak `G3` recall in the current MoCo result
- include a small augmentation ablation:
  - especially color jitter strength and crop strength for contrastive SSL
- include end-to-end fine-tuning results:
  - not only frozen-encoder / linear-probe evaluation
- add representation-quality visualization:
  - t-SNE or UMAP for baseline, CAE, and contrastive methods
- strengthen generalization evidence if feasible:
  - cross-fold validation if external dataset evaluation is not realistic

## Team Priorities

The team should follow this order:

1. Finalize the remaining clean experiment runs for baseline, CAE, and SimCLR.
2. Compare all four methods using saved evaluation outputs.
3. Run focused ablations.
4. Investigate class imbalance, especially `G3` and `G5`.
5. Strengthen validation and analysis for the report.
6. Assemble figures, tables, and final paper narrative.

## Phase 1: Finalize Comparable Experiment Outputs

### 1. Baseline

- Re-run the baseline training pipeline under the cleaned split protocol.
- Save the best classifier and final evaluation outputs.
- Confirm `evaluation_metrics.txt` exists under `./output/baseline/`.

Expected scripts:
- `training/baseline/train_baseline.py`
- `evaluation/baseline/eval_baseline.py`

### 2. CAE

- Re-run CAE pretraining if needed using the intended checkpoint source.
- Run both CAE downstream settings:
  - frozen-encoder / head-only fine-tuning
  - end-to-end fine-tuning with encoder unfrozen
- Validate on `Val.csv` and evaluate on `Test.csv`.
- Evaluate on `Test.csv`.
- Save `evaluation_metrics.txt` under `./output/cae/`.

Expected scripts:
- `training/cae/train_cae.py`
- `training/cae/finetune_cae.py`
- `evaluation/cae/eval_cae.py`

Important note:
- CAE still has some legacy checkpoint/output assumptions. Clean results matter more than preserving older artifact naming.

### 3. SimCLR

- Re-run SimCLR pretraining if needed.
- Run both SimCLR downstream settings:
  - frozen-encoder / head-only fine-tuning
  - end-to-end fine-tuning with encoder unfrozen
- Fine-tune using the cleaned train/val split.
- Evaluate on `Test.csv`.
- Save `evaluation_metrics.txt` under `./output/simclr/`.

Expected scripts:
- `training/simclr/pretrain_simclr.py`
- `training/simclr/finetune_simclr.py`
- `evaluation/simclr/eval_simclr.py`

### 4. MoCo

- Keep the recorded Colab MoCo run as the first valid MoCo result.
- Report both MoCo downstream settings when possible:
  - Stage 1 head-only result
  - Stage 2 end-to-end fine-tuned result
- If a second MoCo run is attempted, change only one factor at a time.
- Save all future MoCo run notes in the same style as `docs/colab_run_notes.md`.

Expected scripts:
- `training/moco/pretrain_moco.py`
- `training/moco/finetune_moco.py`
- `evaluation/moco/eval_moco.py`

## Phase 2: Generate Final Model Comparison

- After baseline, CAE, SimCLR, and MoCo all have fresh `evaluation_metrics.txt` files, run:
  - `evaluation/shared/compare_models.py`
- Generate final tables/plots comparing:
  - accuracy
  - macro F1
  - weighted F1
  - Cohen's kappa
  - per-class recall / precision / F1
- Identify:
  - best overall model
  - best minority-class model
  - best `G5` model
  - weakest consistent class across all methods
- Prepare one side-by-side table for:
  - frozen-encoder / linear-probe results
  - end-to-end fine-tuning results

## Phase 3: Required Ablation Studies

The feedback emphasized that the paper should explain *why* the models work, not only report one final score. Use small, controlled ablations.

### A. Fine-Tuning Strategy Ablation

Compare:
- linear probe / head-only training
- full end-to-end fine-tuning
- optional gradual unfreezing if time allows

Track:
- validation loss
- test accuracy
- macro F1
- `G3` and `G5` recall

Required output:
- one compact table showing whether end-to-end fine-tuning improves each SSL method over frozen-encoder evaluation

### B. Augmentation Ablation

Test only a few interpretable settings, not too many:
- base augmentations
- lower color jitter strength
- higher color jitter strength
- tighter crop range vs wider crop range

Goal:
- see whether strong augmentations help representation quality or hurt class sensitivity

Priority:
- run this first for SimCLR
- run it for MoCo only if time and compute remain

### C. Projection Head / Representation Ablation

For contrastive methods, compare:
- current MLP projection head
- a smaller head or simpler head if time allows

Goal:
- see whether downstream performance depends heavily on projection design

### D. Batch / Temperature Sensitivity

Only if time remains:
- try one lower batch size
- try one alternate temperature

Do not turn this into a large sweep. Keep it small and interpretable.

## Phase 4: Class Imbalance Follow-Up

The strongest practical issue remaining is poor minority-class performance, especially inconsistent `G3` and historically weak `G5`.

Team tasks:
- measure per-class recall carefully for every model
- test one class-balancing intervention at a time
- prioritize methods that improve `G5` without fully collapsing `G3`

Suggested interventions:
- class-balanced sampling
- focal loss tuning
- minimum minority-class presence per batch
- selective MixUp / CutMix for minority classes if the simpler methods fail

Success criterion:
- improve minority-class recall without destroying overall performance

## Phase 5: Validation and Analysis

The feedback also asked for more rigorous evaluation than a single headline score.

Minimum analysis work:
- summarize all four models in one comparison table
- include confusion matrices
- include ROC curves
- include representation visualization such as t-SNE / UMAP for:
  - Baseline
  - CAE
  - one contrastive method at minimum

If time allows:
- run cross-validation or repeated runs
- report mean and variance
- discuss result stability, not just the best run

Preferred order:
- first do repeated runs or cross-fold validation on the strongest 1-2 methods
- do not spend major compute on every minor ablation before the main comparison is stable

## Phase 6: Writing / Reporting Responsibilities

Suggested division of work:

### Teammate A: Comparison & Tables

- collect final metrics from all four models
- run shared comparison
- prepare tables for the report

### Teammate B: Ablations

- run controlled end-to-end fine-tuning and augmentation ablations
- summarize the effect of each change

### Teammate C: Class Imbalance & Error Analysis

- analyze `G3` / `G5`
- prepare confusion-matrix discussion
- identify where each model fails
- test one minority-class mitigation strategy at a time

### Teammate D: Visualization & Writing

- produce t-SNE / ROC / confusion-matrix figures
- write methods/results/discussion sections from the saved outputs

## Deliverables Expected From The Team

By the end of the remaining project, the team should produce:

- final fresh evaluation outputs for baseline, CAE, SimCLR, and MoCo
- one comparison table across all methods
- one second table comparing frozen-encoder vs end-to-end fine-tuning
- one short ablation section
- one imbalance/error analysis section
- one visualization section
- one clear written conclusion:
  - which method is best overall
  - which method helps minority classes most
  - what tradeoffs remain

## Practical Rules For The Team

- Do not evaluate on `Test.csv` during tuning.
- Keep each experiment change isolated.
- Save every run's outputs in model-specific folders.
- Record run settings next to the saved outputs.
- Compare methods only after evaluation outputs are generated under the cleaned protocol.
- Prefer a small number of clean, interpretable experiments over many noisy runs.

## Immediate Next Actions

1. Re-run baseline, CAE, and SimCLR to regenerate fresh comparable outputs.
2. For CAE and SimCLR, explicitly capture both frozen-encoder and end-to-end fine-tuning results.
3. Run shared model comparison once at least two additional metrics files exist.
4. Assign ablation, imbalance, and visualization responsibilities across teammates.
5. Build the final results tables and writing outline from the saved artifacts.
