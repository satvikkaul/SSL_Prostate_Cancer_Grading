# Colab Run Notes

## Latest Recorded MoCo Run

- Drive run folder: `/content/drive/MyDrive/Prostate_SSL/runs/moco_20260320_023752/`
- Branch used in Colab: `method/moco-v2`
- Dataset archive used: `MyDrive/Prostate_SSL/dataset.zip`

## Checkpoint Flow

- MoCo pretraining writes encoder checkpoints under:
  - `./output/models/moco/encoder_q_epochXXX.weights.h5`
- MoCo fine-tuning uses a selected encoder checkpoint from the same run.
- MoCo evaluation uses:
  - `./output/moco/best_moco_overall.keras`

## Latest Recorded MoCo Evaluation

- Accuracy: `0.555`
- Macro F1: `0.453`
- Weighted F1: `0.488`
- Cohen's Kappa (Quadratic): `0.450`

### Per-Class Metrics

- `NC`: Precision `0.639`, Recall `0.714`, F1 `0.674`
- `G3`: Precision `0.480`, Recall `0.038`, F1 `0.071`
- `G5`: Precision `0.453`, Recall `0.414`, F1 `0.433`
- `G4`: Precision `0.526`, Recall `0.802`, F1 `0.636`

### AUC-ROC

- `NC`: `0.894`
- `G3`: `0.657`
- `G5`: `0.873`
- `G4`: `0.766`

## Saved Artifacts

- `./output/moco/evaluation_metrics.txt`
- `./output/moco/confusion_matrix.png`
- `./output/moco/roc_curves.png`
- `./output/moco/best_moco_overall.keras`

## Colab Workflow

1. Push the latest code to the `method/moco-v2` branch.
2. Open `run_colab.ipynb` in Colab.
3. Mount Google Drive.
4. Clone the repo branch into `/content/SSL_Prostate_Cancer_Grading`.
5. Install Colab-safe dependencies without overriding Colab's TensorFlow stack.
6. Copy `MyDrive/Prostate_SSL/dataset.zip` to Colab local storage and unzip it.
7. Link `./output` to the Drive-backed run folder.
8. Run MoCo pretraining.
9. Run MoCo fine-tuning.
10. Run MoCo evaluation.
11. Run the sync cell.
12. Terminate the runtime after confirming artifacts are present in Drive.

## Important Colab Notes

- Do not install pinned `tensorflow==2.18.1` on Colab.
- Keep `./output` Drive-backed through the notebook symlink step.
- The sync cell is for notebook/runtime metadata and generated dataset CSVs; main model outputs are already preserved by the Drive-backed `./output` folder.
