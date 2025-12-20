# GPU Training Guide

**For the teammate with GPU access**

This guide provides step-by-step instructions to train all models for the SSL Prostate Cancer Grading project.

---

## Prerequisites

### 1. Check GPU Availability
```bash
python3 -c "import tensorflow as tf; print('GPUs:', tf.config.list_physical_devices('GPU'))"
```
Expected output: `GPUs: [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]`

### 2. Verify Dataset
```bash
ls -la dataset/
```
You should see:
- `images/` folder with .jpg files
- `Train.csv`
- `Test.csv`

If missing, run:
```bash
python setup_data.py
```

---

## Training Schedule (Total: ~6-8 hours)

### Phase 1: Baseline (No SSL) - ~1-2 hours
Train classifier from scratch (random initialization).

```bash
python train_baseline.py
```

**Expected output:**
- `./output/baseline/best_baseline_classifier.keras`
- `./output/baseline/training_curves.png`
- `./output/baseline/training_history.csv`

**Verify:**
```bash
ls -lh output/baseline/best_baseline_classifier.keras
```

---

### Phase 2: SimCLR Pretraining - ~2-3 hours
Contrastive learning (the assignment requirement!).

```bash
python simclr_pretrain.py
```

**Expected output:**
- `./output/simclr/encoder_weights.h5`
- `./output/simclr/simclr_final_model.h5`
- `./output/simclr/training_log.csv`
- `./output/simclr/training_curves.png`

**Monitor progress:**
- Watch training loss decrease (should go from ~6.0 → ~4.0)
- Loss curves saved after each epoch

**Verify:**
```bash
ls -lh output/simclr/encoder_weights.h5
```

---

### Phase 3: SimCLR Fine-Tuning - ~1 hour
Fine-tune SimCLR encoder for classification.

```bash
python fine_tune_simclr.py
```

**Expected output:**
- `./output/simclr/best_simclr_classifier.keras`
- `./output/simclr/classification_results.png`

**Verify:**
```bash
ls -lh output/simclr/best_simclr_classifier.keras
```

---

## Evaluation (Quick - No GPU needed)

### Evaluate All Models
```bash
python evaluate_baseline.py
python evaluate_simclr.py
python evaluate_final.py  # Autoencoder (already done)
```

**Expected outputs:**
- Confusion matrices for each model
- Classification reports
- ROC curves
- Metrics files

---

## Configuration Adjustments

### If Out of Memory (OOM) Error

**In `simclr_pretrain.py`:**
```python
BATCH_SIZE = 32  # Reduce from 64 to 32
```

**In `fine_tune_simclr.py` and `train_baseline.py`:**
```python
BATCH_SIZE = 4  # Reduce from 8 to 4
```

### If Training Takes Too Long

**Quick demo version (in `simclr_pretrain.py`):**
```python
EPOCHS = 10  # Reduce from 30 to 10
```

**Note:** Reduced epochs may hurt accuracy, but will demonstrate the concept.

---

## GPU Selection

If you have multiple GPUs, set the device at the top of each script:

```python
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use GPU 0
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # Use GPU 1
```

---

## Expected Results

### Training Time Summary
| Task | Time (GPU) | Output |
|------|------------|--------|
| Baseline Training | 1-2 hours | Baseline classifier |
| SimCLR Pretraining | 2-3 hours | Pretrained encoder |
| SimCLR Fine-tuning | 1 hour | SimCLR classifier |
| **Total** | **4-6 hours** | All models ready |

### Accuracy Expectations
| Model | Expected Val Accuracy |
|-------|----------------------|
| Baseline (No SSL) | ~45-55% |
| Autoencoder SSL | ~60-63% |
| **SimCLR SSL** | **~65-70%** (Best) |

*SimCLR should outperform both baseline and autoencoder.*

---

## Troubleshooting

### Problem: CUDA Out of Memory
**Solution:** Reduce batch size (see Configuration Adjustments above)

### Problem: "Dataset not found"
**Solution:** Run `python setup_data.py` first

### Problem: Training loss not decreasing
**Solution:**
- Check data augmentation is working
- Verify images are loading correctly
- Try reducing learning rate

### Problem: Validation accuracy stuck at 25-30%
**Solution:** This is class imbalance - normal for first few epochs. Should improve by epoch 10+.

---

## Monitoring Training

### Watch GPU Usage
```bash
watch -n 1 nvidia-smi
```

### Check Training Progress
- SimCLR: Loss should decrease from ~6.0 → ~4.0
- Fine-tuning: Accuracy should increase from ~25% → ~60%+
- Baseline: Similar to fine-tuning but may plateau lower

### Training Curves
All scripts generate plots automatically:
- `training_curves.png` - Loss over time
- `classification_results.png` - Accuracy over time

---

## After Training: Push Results

### 1. Commit Model Files
```bash
git add output/
git commit -m "Add trained models: Baseline, SimCLR, and Autoencoder"
```

### 2. Push to GitHub
```bash
git push origin main
```

### 3. Notify Teammate
Send a message: "✅ All models trained! Pull the latest code."

---

## Quick Test (Before Full Training)

To verify everything works, do a 1-epoch test run:

```bash
# Edit simclr_pretrain.py: Change EPOCHS = 30 to EPOCHS = 1
# Edit train_baseline.py: Change EPOCHS = 50 to EPOCHS = 1
# Edit fine_tune_simclr.py: Change EPOCHS_STAGE_1 = 50 to EPOCHS_STAGE_1 = 1

python train_baseline.py      # Should finish in ~2 minutes
python simclr_pretrain.py      # Should finish in ~4 minutes
python fine_tune_simclr.py     # Should finish in ~2 minutes
```

If all three complete without errors, you're ready for full training!

---

## Questions?

Contact your teammate or check:
- `README.md` - Project overview
- `Model_architecture.md` - Architecture details
- `notes.md` - Experiment logs

Good luck! 🚀
