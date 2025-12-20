# Google Colab Training Guide

**Running SSL Prostate Cancer Grading on Google Colab Pro**

This guide walks you through training all three models (Baseline, Autoencoder-SSL, SimCLR-SSL) on Google Colab.

---

## Prerequisites

### 1. Google Colab Access
- **Free Tier:** T4 GPU (16GB) - Works fine, ~4-6 hours total
- **Colab Pro:** V100/A100 GPU - Faster, ~2-3 hours total
- **Recommended:** Colab Pro for uninterrupted session

### 2. Dataset Preparation

**Option A: Upload to Google Drive (Recommended)**
1. Download SICAPv2 dataset from [Mendeley Data](https://data.mendeley.com/datasets/9xxm58dvs3/1)
2. Extract the zip file
3. Upload to Google Drive: `/MyDrive/datasets/SICAP/`
   - Should contain: `images/` folder and `partition/` folder

**Option B: Upload Directly to Colab**
- Use Colab's file upload feature
- Slower but works for quick tests

---

## Setup Steps

### Step 1: Open Google Colab

1. Go to [Google Colab](https://colab.research.google.com/)
2. Create a new notebook: **File → New Notebook**
3. Enable GPU: **Runtime → Change runtime type → GPU → T4/V100/A100**

### Step 2: Mount Google Drive

```python
from google.colab import drive
drive.mount('/content/drive')
```

Click the link, authorize, and paste the code.

### Step 3: Clone Repository

```python
!git clone https://github.com/satvikkaul/SSL_Prostate_Cancer_Grading.git
%cd SSL_Prostate_Cancer_Grading
```

### Step 4: Install Dependencies

```python
!pip install -q tensorflow pandas numpy matplotlib scikit-learn scikit-image opencv-python openpyxl seaborn
```

### Step 5: Setup Dataset

```python
# Create symbolic link to your Google Drive dataset
!ln -s /content/drive/MyDrive/datasets/SICAP /content/SSL_Prostate_Cancer_Grading/dataset

# Generate CSV files
!python setup_data.py
```

**Verify dataset:**
```python
!ls -lh dataset/
!head -5 dataset/Train.csv
```

You should see `Train.csv`, `Test.csv`, and `images/` folder.

---

## Training Pipeline

### Option A: Sequential Training (Recommended for Beginners)

Run each model one at a time, save results to Google Drive.

#### 1. Train Baseline (No SSL)

```python
# Edit GPU setting if needed
!sed -i 's/CUDA_VISIBLE_DEVICES"] = "0"/CUDA_VISIBLE_DEVICES"] = "0"/' train_baseline.py

# Train
!python train_baseline.py

# Save results to Google Drive
!cp -r output/baseline /content/drive/MyDrive/SSL_Project_Results/
```

**Expected time:** 1-2 hours (T4), 30-45 min (A100)

#### 2. Train SimCLR SSL

```python
# Pretrain encoder
!python simclr_pretrain.py

# Fine-tune classifier
!python fine_tune_simclr.py

# Save results
!cp -r output/simclr /content/drive/MyDrive/SSL_Project_Results/
```

**Expected time:** 3-4 hours total (T4), 1.5-2 hours (A100)

#### 3. Train Autoencoder SSL (Optional - already have results)

```python
# Pretrain encoder
!python Main.py

# Fine-tune classifier
!python fine_tune.py

# Save results
!cp -r output/models /content/drive/MyDrive/SSL_Project_Results/
```

---

### Option B: Automated Sequential Training (Advanced)

Run all models with a single script:

```python
import os
import time
from datetime import datetime

def train_all_models():
    """Train all three models sequentially"""
    
    models = [
        ("Baseline (No SSL)", "train_baseline.py", 1.5),
        ("SimCLR Pretraining", "simclr_pretrain.py", 2.5),
        ("SimCLR Fine-tuning", "fine_tune_simclr.py", 1.0),
    ]
    
    results = []
    
    for model_name, script, est_hours in models:
        print("\n" + "="*70)
        print(f"Starting: {model_name}")
        print(f"Estimated time: {est_hours} hours")
        print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*70 + "\n")
        
        start_time = time.time()
        
        # Run training
        result = os.system(f"python {script}")
        
        elapsed = (time.time() - start_time) / 3600  # hours
        
        results.append({
            'model': model_name,
            'script': script,
            'success': result == 0,
            'time_hours': elapsed
        })
        
        print(f"\n✓ {model_name} completed in {elapsed:.2f} hours")
        
        # Save results to Drive periodically
        os.system("cp -r output /content/drive/MyDrive/SSL_Project_Results/")
    
    # Print summary
    print("\n" + "="*70)
    print("TRAINING SUMMARY")
    print("="*70)
    for r in results:
        status = "✓ SUCCESS" if r['success'] else "✗ FAILED"
        print(f"{r['model']:30s}: {status} ({r['time_hours']:.2f} hours)")
    print("="*70)
    
    return results

# Run training
results = train_all_models()
```

---

## Evaluation

After training completes:

```python
# Evaluate all models
!python evaluate_baseline.py
!python evaluate_simclr.py
!python evaluate_final.py  # Autoencoder (if trained)

# Generate comparison
!python compare_all_models.py

# Save all results
!cp -r output /content/drive/MyDrive/SSL_Project_Results/final_output
!cp *.png /content/drive/MyDrive/SSL_Project_Results/plots/
```

---

## Downloading Results

### Method 1: Via Google Drive
Results are automatically saved to Google Drive during training. Access them at:
```
/MyDrive/SSL_Project_Results/
```

### Method 2: Direct Download from Colab

```python
from google.colab import files

# Download specific files
files.download('output/model_comparison.csv')
files.download('output/model_comparison_plots.png')
files.download('output/comparison_report.txt')

# Or zip everything
!zip -r ssl_results.zip output/
files.download('ssl_results.zip')
```

---

## Monitoring & Troubleshooting

### Check GPU Usage

```python
!nvidia-smi
```

Expected output: GPU utilization ~80-100% during training.

### Monitor Training Progress

```python
# View loss in real-time
!tail -f output/simclr/training_log.csv
```

### Common Issues

#### 1. Out of Memory (OOM)

**Solution:** Reduce batch size

```python
# Edit scripts before running
!sed -i 's/BATCH_SIZE = 64/BATCH_SIZE = 32/' simclr_pretrain.py
!sed -i 's/BATCH_SIZE = 8/BATCH_SIZE = 4/' fine_tune_simclr.py
!sed -i 's/BATCH_SIZE = 8/BATCH_SIZE = 4/' train_baseline.py
```

#### 2. Session Timeout (>12 hours)

**Solution:** Use Colab Pro or train models separately across multiple sessions

```python
# Save checkpoints frequently (already implemented in scripts)
# Resume from checkpoint if needed
```

#### 3. Dataset Not Found

```python
# Verify paths
!ls -la dataset/
!ls -la dataset/images/ | head -10

# If missing, re-run setup
!python setup_data.py
```

#### 4. Import Errors

```python
# Reinstall dependencies
!pip install --upgrade tensorflow keras pandas numpy matplotlib scikit-learn opencv-python
```

---

## Optimization Tips

### 1. Use Mixed Precision Training

```python
# Add to beginning of training scripts
import tensorflow as tf
tf.keras.mixed_precision.set_global_policy('mixed_float16')
```

**Benefit:** ~2x faster training on V100/A100

### 2. Enable XLA Compilation

```python
# In model.compile()
jit_compile=True
```

### 3. Reduce Logging

```python
# Set verbose=0 in model.fit() for faster training
verbose=0
```

---

## Expected Timeline (Colab Pro - A100)

| Task | Time |
|------|------|
| Setup | 5 min |
| Baseline Training | 30 min |
| SimCLR Pretraining | 1-1.5 hours |
| SimCLR Fine-tuning | 30 min |
| Evaluation | 10 min |
| **Total** | **~2.5-3 hours** |

---

## Final Checklist

Before ending your Colab session:

- [ ] All models trained successfully
- [ ] Results saved to Google Drive
- [ ] Evaluation metrics generated
- [ ] Comparison plots created
- [ ] Model files downloaded/backed up
- [ ] Training logs saved

---

## Support

If you encounter issues:
1. Check `GPU_TRAINING_GUIDE.md` for detailed troubleshooting
2. Review training logs in `output/*/training_log.csv`
3. Verify dataset with `!python setup_data.py`

---

**Ready to train? Open Google Colab and start with Step 1!** 🚀
