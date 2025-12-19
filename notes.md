# Project Updates & Changes Log

**Project:** Self-Supervised Learning for Prostate Cancer Grading (SICAPv2)
**Date:** December 2025

---

## 1. New Scripts Created

### `setup_data.py` (The Data Translator)
* **Purpose:** Converts the raw SICAPv2 Excel partition files (`.xlsx`) into machine-readable CSV files (`Train.csv`, `Test.csv`) required by the TensorFlow data generators.
* **Key Logic Implemented:**
    * **Smart Header Cleaning:** Automatically standardizes column names by stripping whitespace and converting to uppercase (e.g., `" NC "` $\rightarrow$ `"NC"`) to prevent "all-zero label" errors.
    * **Class Merging (Business Logic):** Implemented logic to merge **Gleason 4 Cribriform (`G4C`)** into the standard **Gleason 4 (`G4`)** class. This ensures aggressive subtypes are included in training without altering the 4-class model architecture.
    * **Cribriform Filtering:** Explicitly excludes specialized `*Cribriform.xlsx` partition files to prevent data pollution.
    * **Zero-Check Protection:** Validates that required columns exist before saving; prevents the generation of broken CSVs containing all-zero labels.

### `fine_tune.py` (The Downstream Task)
* **Purpose:** Loads the pre-trained SSL encoder and trains a classifier for Gleason Grading.
* **Key Features:**
    * **Encoder Integration:** Reconstructs the exact architecture from the pretext task to ensure weight compatibility.
    * **Weight Loading:** Loads the Self-Supervised Learning (SSL) weights (`VAE_weights.weights.h5`) from Phase 1.
    * **Classification Head:** Appends a `Dropout(0.5)` layer and a `Dense(4, activation='softmax')` layer to classify the 4 grades (NC, G3, G4, G5).
    * **Optimization:** Configured with `CategoricalCrossentropy` and `Adam` optimizer.

### `utils_huleo.py`
* **Purpose:** A helper utility script created to resolve specific library compatibility crashes and provide data augmentation support during the setup phase.

---

## 2. Critical Modifications to Existing Files

### `variational_autoencoder.py`
* **Keras 3 Compatibility:**
    * Updated optimizer initialization from `Adam(lr=...)` to `Adam(learning_rate=...)`.
    * Replaced deprecated `fit_generator()` calls with standard `.fit()`.
    * Updated `ModelCheckpoint` to enforce the `.weights.h5` file extension.

### `Main.py` (Pretext Task)
* **Visualization:** Added a post-training visualization block to generate "Original vs. Reconstructed" images, providing visual verification of SSL performance.
* **Path Management:** Updated file paths to ensure weights and logs are saved correctly to `output/models/exp_XXXX/`.

---

## 3. Fine-Tuning Optimization Strategy

![alt text](./output/classification_result_wo_dataAug.png)

To resolve initial issues with **Overfitting** (Validation Accuracy dropping) and **Loss Fluctuation**, the following critical changes were applied to `fine_tune.py`:

### **A. Freezing the Encoder (Transfer Learning)**
* **Issue:** The untrained classifier head was propagating large errors back into the encoder, destroying the pre-trained feature representations ("Catastrophic Forgetting").
* **Fix:** Explicitly froze the encoder layers to make them non-trainable during the fine-tuning phase.
    ```python
    # Logic applied in fine_tune.py
    for layer in encoder.layers:
        layer.trainable = False
    ```
* **Result:** Stabilizes training by preserving the "knowledge" learned during the SSL pretext task.

### **B. Learning Rate Adjustment**
* **Issue:** The initial Learning Rate (`0.001`) caused the loss to fluctuate aggressively, preventing convergence.
* **Fix:** Reduced the Learning Rate by a factor of 10.
    * **New LR:** `0.0001`
* **Result:** Smoother loss curve and more stable gradient descent steps.

---

## 4. Data Processing Decisions

* **Class Handling:** Confirmed that **G4C** (Cribfriform) is treated as a subset of **G4**. This aligns with the model's 4-class output structure (NC, G3, G4, G5).
* **Data Validation:** Established a strict rule that any data rows with missing labels (all zeros) are blocked at the `setup_data.py` stage.

## 5. Investigation & Fixes: Custom Generator & Class Imbalance (New)

### **A. Missing Data Augmentation**
*   **Issue:** The default `ImageDataGenerator` in `fine_tune.py` applied **zero** augmentation, leading to rapid overfitting.
*   **Fix:** Integrated `my_data_generator.py` (Custom `DataGenerator` class).
    *   **Action:** Modified `my_data_generator.py` to accept raw CSV columns (`y_cols`) instead of hardcoded dictionaries.
    *   **Result:** Enabled Rotation, Shift, Zoom, and Intensity scaling during training.

### **B. Class Imbalance**
*   **Issue:** The dataset is heavily imbalanced (mostly NC, few G5). The model was biased towards the majority class.
*   **Fix:** Implemented Scikit-Learn `compute_class_weight`.
    *   **Action:** Calculated weights inversely proportional to class frequency and passed `class_weight` to `model.fit()`.
    *   **Result:** Misclassifying rare grades now penalizes the model significantly more.

### **C. Performance & Pipeline Optimization (tf.data)**
*   **Issue:** Training was extremely slow with 0% GPU usage (CPU bottleneck).
*   **Fix:** Migrated from Keras `Sequence` to `tf.data.Dataset`.
    *   Implemented a wrapper in `my_data_generator.py` using `tf.py_function` + `tf.data.AUTOTUNE`.
    *   Result: Parallel CPU prefetching, enabling better GPU utilization.

---

# Project Log: Stage 2 Optimization (Fine-Tuning Outcomes)

**Phase:** Downstream Task (Classification)
**Method:** Two-Stage Fine-Tuning (Frozen Head $\to$ Unfrozen Encoder)

![alt text](./output/classification_results_old.png)
![alt text](./output/final_confusion_matrix_model_collapse.png)

### Results Analysis (Run 1: Failed)
1.  **Stage 1 (Head Only, Frozen Encoder):**
    *   **Accuracy:** Stagnated at **~24-25%**.
    *   **Inference:** The pre-trained SSL weights alone (without fine-tuning) were **insufficient** to distinguish specific cancer grades.

2.  **Stage 2 (Unfrozen Encoder, Low LR):**
    *   **Accuracy:** Immediately doubled to **~51%** upon unfreezing.
    *   **Observations:**
        *   **Overfitting:** Validation loss reached a minimum (~1.20) at Epoch 7 (Stage 2) and then began to rise.
        *   **Instability:** High volatility in validation accuracy (large jumps) suggests the Batch Size (16) may be too small relative to the high class weights for rare samples.

### Results Analysis (Run 2: Success with High Volatility)

**Changes Implemented:** `tf.data.Dataset` pipeline, Batch Size 32, Stage 1=5 Epochs, Workers Removed (Native TF Parallelism).

**Outcomes:**
1.  **Model Learning confirmed:** Validation Accuracy hit **~56%** at Epoch 11, breaking the previous "Model Collapse" barrier.
2.  **Volatility persists:** The accuracy graph is still extremely jumpy. The model is suffering from "Gradient Shock" due to high class weights (G5=3.44).

### Final Evaluation (After Fixing Script)

![alt text](./output/final_confusion_matrix.png)

**Confusion Matrix Analysis:**
*   **Improvements:** We have a diagonal! The model is correctly identifying NC, G3, and G4 to some extent.
*   **The Blocking Issue:** **G5 (Class 3) is completely ignored.**
    *   Prediction counts for G5: **0**.
    *   The model has decided it is safer to *never* predict G5 than to risk being wrong, likely because the gradients are too noisy.

### Next Steps
*   **Gradient Clipping:** Implement `clipnorm=1.0` in the `Adam` optimizer to fix the volatility and help the model learn the rare G5 class without crashing.

---

## 6. Final Architecture & Experiments Summary

### **Iterations Tried:**
1. **Image Size Fix:** Changed from 512×512 → 128×128 (matching paper specification)
2. **SSL Training:** Increased from 3 → 15 epochs for better feature learning
3. **Learning Rate Tuning:** Fixed divergence (0.5 → 0.01 → 0.001 → 0.0001 → 1e-5)
4. **Frozen vs Unfrozen Encoder:**
   - Frozen: Failed (training stuck at 42% accuracy)
   - Unfrozen: Successful (reached 62-66% validation accuracy)
5. **Class Weights Experiments:**
   - Moderate (G5=3.0): Caused training instability
   - Extreme (G5=15.0): Complete training collapse (accuracy → 15%)
6. **Focal Loss Experiments:**
   - alpha=0.5, gamma=2.0: Best stability, 62.8% accuracy
   - alpha=0.25, gamma=1.5: Improved G3 (20% recall) but lost G5
   - alpha=0.25, gamma=2.0: Initial test configuration

### **Final Configuration (Best Performing):**
```python
# SSL Pretraining (Main.py)
- Image size: 128×128×3
- Epochs: 15
- Batch size: 16
- Architecture: Conv encoder [16,32,64,128,256] → bottleneck [128,64,128] → decoder
- Loss: MSE (reconstruction)

# Fine-tuning (fine_tune.py)
- Encoder: Unfrozen (all 35 layers trainable)
- Learning rate: 1e-5 (ADAM with momentum=0.9, clipnorm=1.0)
- Loss: Focal loss (alpha=0.5, gamma=2.0)
- Epochs: 50
- Batch size: 8
- Classifier: GMP → Dense(200, ReLU) → Dense(4, Softmax)
```

### **Final Results (Best Model - Epoch 28):**

**Classification Metrics:**
| Class | Support | Recall | Precision | F1-Score |
|-------|---------|--------|-----------|----------|
| NC    | 1727    | 85%    | 80%       | 0.83     |
| G3    | 497     | 13%    | 14%       | 0.13     |
| G5    | 247     | **0%** | **N/A**   | **0.00** |
| G4    | 1042    | 65%    | 54%       | 0.59     |

**Overall Performance:**
- **Accuracy:** 62.8% (2205/3513 correct)
- **Macro F1:** 0.39
- **Weighted F1:** 0.62

**Confusion Matrix Analysis:**
```
True G5 (247 samples) predicted as:
  - NC: 43 (17%)
  - G3: 28 (11%)
  - G5: 0 (0%)   ← Complete failure
  - G4: 176 (71%) ← Dominant misclassification
```

### **Key Findings & Limitations:**

1. **Successful Learning:** NC (85%) and G4 (65%) achieved good recall, demonstrating SSL features work for majority classes

2. **Partial Learning:** G3 improved from 0% → 13% recall with focal loss, showing minority class learning is possible

3. **Fundamental Limitation:** G5 achieved 0% recall across ALL configurations:
   - G5 and G4 are indistinguishable in reconstruction feature space
   - 71% of G5 samples misclassified as G4
   - Autoencoder learns texture/structure, not fine-grained cancer grading patterns

4. **Conclusion:** Reconstruction-based SSL (CAE/VAE) **cannot learn the cellular-level features** required to distinguish high-grade cancer subtypes (G5 vs G4)

### **Recommended Alternatives (Future Work):**
- **Contrastive Learning (SimCLR/MoCo):** Forces discriminative feature learning
- **Supervised Pretraining:** ImageNet-pretrained encoders transfer better to histopathology
- **Multi-Task Learning:** Joint training on related tasks to learn robust features
- **Class Merging:** Combine G4+G5 into "High-Grade" class (clinically justified, would achieve ~68% recall)
