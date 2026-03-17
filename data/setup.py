import pandas as pd
import os
import glob
import io
import cv2
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

# Configuration
DATASET_DIR = './dataset'
PARTITION_DIR = os.path.join(DATASET_DIR, 'partition')
PANDA_PARQUET_DIR = os.path.join(DATASET_DIR, 'PANDA-PLUS-Bench', 'data')
PANDA_IMAGES_DIR = os.path.join(DATASET_DIR, 'panda_images')
PANDA_PARQUET_FILE = 'baseline-00000-of-00001.parquet'
TRAIN_SPLIT_FILE = 'TrainSplit.csv'
VAL_FILE = 'Val.csv'
CLASS_COLUMNS = ['NC', 'G3', 'G5', 'G4']

def convert_sicap_to_csv(partition_folder):
    # Find all excel files recursively
    excel_files = glob.glob(os.path.join(partition_folder, '**/*.xlsx'), recursive=True)
    
    if not excel_files:
        print("Error: No .xlsx files found in dataset/partition/. Please check your extraction.")
        return

    print(f"Found partition files: {excel_files}")

    for file_path in excel_files:
        filename = os.path.basename(file_path).lower()
        
        # 1. Skip the specialized 'cribriform' partition files
        # (We only want the G4C column from the MAIN files, not the separate files)
        if 'cribfriform' in filename:
            print(f"Skipping specialized partition file: {filename}")
            continue

        # 2. Determine Train vs Test
        if 'train' in filename:
            output_name = 'Train.csv'
        elif 'test' in filename:
            output_name = 'Test.csv'
        else:
            print(f"Skipping {filename} (not train/test)...")
            continue 

        print(f"Processing {filename} -> {output_name}...")
        
        try:
            df = pd.read_excel(file_path)
        except Exception as e:
            print(f"Error reading {filename}: {e}")
            continue
        
        # Prepare new dataframe
        new_df = pd.DataFrame()
        new_df['image_name'] = df['image_name']
        
        # --- MERGING LOGIC ---
        # We look for G4C and merge it into G4
        
        # 1. Handle NC, G3, G5 (Direct Copy)
        for col in ['NC', 'G3', 'G5']:
            if col in df.columns:
                new_df[col] = df[col].fillna(0).astype(int)
            else:
                new_df[col] = 0 # Default to 0 if missing

        # 2. Handle G4 (Merge G4 + G4C)
        # Initialize G4 with existing G4 column or 0
        if 'G4' in df.columns:
            g4_vals = df['G4'].fillna(0).astype(int)
        else:
            g4_vals = 0
            
        # If G4C exists, add it to the G4 bucket (Logical OR)
        if 'G4C' in df.columns:
            g4c_vals = df['G4C'].fillna(0).astype(int)
            # If it's G4 OR G4C, the result is 1
            new_df['G4'] = ((g4_vals == 1) | (g4c_vals == 1)).astype(int)
            print(f"   -> Merged G4C column into G4 for {filename}")
        else:
            new_df['G4'] = g4_vals

        # Save to CSV
        save_path = os.path.join(DATASET_DIR, output_name)
        new_df.to_csv(save_path, index=False)
        print(f"Saved {save_path}")


def create_validation_split(val_fraction=0.2, random_state=42):
    """
    Create an explicit validation split from SICAPv2 Train.csv.

    Output files:
      - dataset/TrainSplit.csv
      - dataset/Val.csv
    """
    train_csv_path = os.path.join(DATASET_DIR, 'Train.csv')
    if not os.path.exists(train_csv_path):
        print(f"ERROR: Train.csv not found at {train_csv_path}")
        return None, None

    df = pd.read_csv(train_csv_path)
    missing_cols = [col for col in ['image_name'] + CLASS_COLUMNS if col not in df.columns]
    if missing_cols:
        print(f"ERROR: Train.csv missing required columns: {missing_cols}")
        return None, None

    labels = np.argmax(df[CLASS_COLUMNS].values, axis=1)
    train_df, val_df = train_test_split(
        df,
        test_size=val_fraction,
        random_state=random_state,
        shuffle=True,
        stratify=labels
    )

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)

    train_split_path = os.path.join(DATASET_DIR, TRAIN_SPLIT_FILE)
    val_path = os.path.join(DATASET_DIR, VAL_FILE)
    train_df.to_csv(train_split_path, index=False)
    val_df.to_csv(val_path, index=False)

    print(f"Validation split created:")
    print(f"  Train split: {len(train_df)} samples -> {train_split_path}")
    print(f"  Val split:   {len(val_df)} samples -> {val_path}")
    return train_split_path, val_path

def extract_panda_patches(target_size=(128, 128), limit=None):
    """
    Reads PANDA-PLUS baseline parquet file, decodes embedded image bytes,
    resizes to target_size, and saves as .jpg to PANDA_IMAGES_DIR.
    Creates a CSV manifest of extracted paths.
    
    Args:
        target_size: (H, W) tuple. Default (128, 128) to match SICAPv2.
        limit: Optional int to cap number of extracted patches (for testing).
    
    Returns:
        List of absolute image paths extracted.
    """
    os.makedirs(PANDA_IMAGES_DIR, exist_ok=True)
    parquet_path = os.path.join(PANDA_PARQUET_DIR, PANDA_PARQUET_FILE)
    
    if not os.path.exists(parquet_path):
        print(f"ERROR: PANDA parquet file not found at: {parquet_path}")
        return []

    print(f"Loading PANDA-PLUS parquet: {PANDA_PARQUET_FILE} ...")
    df = pd.read_parquet(parquet_path)
    print(f"  Loaded {len(df)} rows. Columns: {df.columns.tolist()}")

    # Auto-detect the image column (HF datasets store as dict with 'bytes' key)
    image_col = None
    for col in df.columns:
        sample = df[col].iloc[0]
        if isinstance(sample, dict) and 'bytes' in sample:
            image_col = col
            print(f"  Detected image column (dict/bytes): '{image_col}'")
            break
        elif isinstance(sample, bytes):
            image_col = col
            print(f"  Detected image column (raw bytes): '{image_col}'")
            break

    if image_col is None:
        print("ERROR: Could not auto-detect image column in parquet. Please inspect schema.")
        print("  Available columns:", df.columns.tolist())
        return []

    extracted_paths = []
    rows_to_process = df.iterrows() if limit is None else list(df.iterrows())[:limit]

    for i, (_, row) in enumerate(rows_to_process):
        try:
            img_data = row[image_col]
            # Handle HF dict format: {'bytes': b'...', 'path': '...'}
            if isinstance(img_data, dict):
                img_bytes = img_data.get('bytes', None)
            else:
                img_bytes = img_data

            if img_bytes is None or len(img_bytes) == 0:
                continue

            # Decode bytes → PIL → numpy
            pil_img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
            img_np = np.array(pil_img)

            # Resize to target size
            img_resized = cv2.resize(img_np, (target_size[1], target_size[0]))

            # Save as jpg
            fname = f"panda_{i:06d}.jpg"
            save_path = os.path.join(PANDA_IMAGES_DIR, fname)
            cv2.imwrite(save_path, cv2.cvtColor(img_resized, cv2.COLOR_RGB2BGR))
            extracted_paths.append(os.path.abspath(save_path))

            if (i + 1) % 500 == 0:
                print(f"  Extracted {i+1} patches...")

        except Exception as e:
            print(f"  Warning: Failed to process row {i}: {e}")
            continue

    print(f"PANDA extraction complete. {len(extracted_paths)} patches saved to {PANDA_IMAGES_DIR}")
    return extracted_paths


def build_pretrain_manifest():
    """
    Builds a unified Pretrain_Manifest.csv for MoCo v2 pretraining.
    Merges:
      - Non-test SICAPv2 images (TrainSplit.csv preferred, else Train.csv)
      - ALL extracted PANDA images from PANDA_IMAGES_DIR
    
    Output CSV has a single column: 'image_path' (absolute paths).
    Saved to: dataset/Pretrain_Manifest.csv
    """
    all_paths = []
    
    # 1. Collect non-test SICAPv2 paths
    sicap_img_dir = os.path.abspath(os.path.join(DATASET_DIR, 'images'))
    sicap_csvs = [TRAIN_SPLIT_FILE] if os.path.exists(os.path.join(DATASET_DIR, TRAIN_SPLIT_FILE)) else ['Train.csv']
    for csv_name in sicap_csvs:
        csv_path = os.path.join(DATASET_DIR, csv_name)
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            if 'image_name' in df.columns:
                paths = [os.path.join(sicap_img_dir, name) for name in df['image_name']]
                # Filter out any paths that don't actually exist
                valid = [p for p in paths if os.path.exists(p)]
                all_paths.extend(valid)
                print(f"  SICAPv2 {csv_name}: {len(valid)} valid non-test images found.")
            else:
                print(f"  Warning: {csv_name} has no 'image_name' column.")
        else:
            print(f"  Warning: {csv_name} not found at {csv_path}")

    # 2. Collect PANDA image paths
    if os.path.exists(PANDA_IMAGES_DIR):
        panda_paths = glob.glob(os.path.join(PANDA_IMAGES_DIR, '*.jpg'))
        panda_paths = [os.path.abspath(p) for p in panda_paths]
        all_paths.extend(panda_paths)
        print(f"  PANDA: {len(panda_paths)} images found in {PANDA_IMAGES_DIR}")
    else:
        print(f"  Warning: PANDA images dir not found: {PANDA_IMAGES_DIR}")
        print(f"  Run extract_panda_patches() first.")

    if not all_paths:
        print("ERROR: No images found. Cannot build manifest.")
        return

    manifest_df = pd.DataFrame({'image_path': all_paths})
    manifest_df = manifest_df.drop_duplicates().reset_index(drop=True)
    
    save_path = os.path.join(DATASET_DIR, 'Pretrain_Manifest.csv')
    manifest_df.to_csv(save_path, index=False)
    print(f"\nPretrain_Manifest.csv saved: {len(manifest_df)} total images -> {save_path}")
    return save_path


if __name__ == "__main__":
    # Step 1: Convert SICAPv2 partitions to CSV
    convert_sicap_to_csv(PARTITION_DIR)
    print("SICAPv2 conversion complete.\n")

    # Step 2: Create explicit train/validation split from SICAP Train.csv
    print("--- Creating SICAP validation split ---")
    create_validation_split()

    # Step 3: Extract PANDA patches
    print("--- Extracting PANDA-PLUS patches ---")
    extract_panda_patches(target_size=(128, 128))

    # Step 4: Build unified pretraining manifest
    print("\n--- Building Pretrain Manifest ---")
    build_pretrain_manifest()
    print("\nAll setup complete.")
