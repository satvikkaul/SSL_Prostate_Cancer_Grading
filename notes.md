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

---

# Project Log: Stage 2 Optimization (Fine-Tuning Outcomes)

**Phase:** Downstream Task (Classification)
**Method:** Two-Stage Fine-Tuning (Frozen Head $\to$ Unfrozen Encoder)

![alt text](./output/classification_results_old.png)

### Results Analysis
1.  **Stage 1 (Head Only, Frozen Encoder):**
    *   **Accuracy:** Stagnated at **~24-25%**.
    *   **Inference:** The pre-trained SSL weights alone (without fine-tuning) were **insufficient** to distinguish specific cancer grades. They provided a "general vision" base but no specific diagnostic ability.

2.  **Stage 2 (Unfrozen Encoder, Low LR):**
    *   **Accuracy:** Immediately doubled to **~51%** upon unfreezing.
    *   **Inference:** Validated that the frozen layers were indeed a bottleneck. Unfreezing allowed the model to adapt features for the specific task.
    *   **Observations:**
        *   **Overfitting:** Validation loss reached a minimum (~1.20) at Epoch 7 (Stage 2) and then began to rise, indicating the start of overfitting.
        *   **Instability:** High volatility in validation accuracy (large jumps) suggests the Batch Size (16) may be too small relative to the high class weights for rare samples.

### Next Steps
*   **Early Stopping:** Adopt the model from Stage 2, Epoch 7 (Best Val Loss).
*   **Stability:** Increase Batch Size to 32 to smooth out gradients from high-weighted rare samples.