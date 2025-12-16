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

* **Class Handling:** Confirmed that **G4C** (Cribriform) is treated as a subset of **G4**. This aligns with the model's 4-class output structure (NC, G3, G4, G5).
* **Data Validation:** Established a strict rule that any data rows with missing labels (all zeros) are blocked at the `setup_data.py` stage.

# Project Log: Stage 2 Optimization (Fine-Tuning)

**Phase:** Downstream Task (Classification)
**Status:** Moving from Linear Probing (Stage 1) to Fine-Tuning (Stage 2)

---

## 1. Context & Motivation

![acc/loss metric](image.png)

After successfully running **Stage 1** (Frozen Encoder), the model achieved a validation accuracy of **~49%**. This confirms that the Self-Supervised Learning (SSL) pretext task successfully learned robust feature representations (better than the 25% random baseline).

To improve performance further, we are initiating **Stage 2**. The goal is to transition the model from a "Generalist" (using generic tissue features) to a "Specialist" (understanding specific cancer grading nuances).

---

## 2. Strategy: Partial Unfreezing (The "Specialist" Approach)

Instead of unfreezing the entire network (which risks destroying pre-trained weights) or keeping it fully frozen (which hits a performance ceiling), I adopted a **Partial Unfreezing Strategy**.

### A. Layer Analysis
I inspected the encoder architecture (40 layers total) and identified two distinct sections:
* **Early Layers (Indices 0–29):** Convolutional blocks responsible for low-level vision (edges, textures, nuclei shapes). These are universal and should remain **FROZEN**.
* **Deep Layers (Indices 30–39):** The "Bottleneck" blocks (`bottle_conv2`, `bottle_conv3`). These handle high-level abstraction and decision-making. These will be **UNFROZEN** to adapt to the specific logic of Gleason Grading.

### B. Implementation Logic
I determined the cutoff point at **Index 30** (`bottle_conv2`).

## 3. Hyperparameter Adjustments
* When unfreezing layers, the model becomes highly sensitive to gradient updates. Large updates can "shock" the pre-trained weights, causing catastrophic forgetting.
* Learning Rate (LR): Reduced from 0.0001 (Stage 1) to 0.00005 (Stage 2).
* Reasoning: A tiny learning rate ensures gentle, precise updates that refine the weights rather than rewriting them.

## 4. Model Preservation
* To ensure we do not lose the successful baseline from Stage 1:
* Backup: The Stage 1 model file was renamed to final_classifier_frozen.keras.
* New Save: The Stage 2 training will write to final_classifier.keras only if it improves upon the metrics.

## 5. Expected Outcome
* By allowing the "Bottleneck" layers to update, we expect the model to learn to distinguish between difficult classes (e.g., Gleason 3 vs. Gleason 4) more effectively, potentially pushing validation accuracy into the 60%+ range.