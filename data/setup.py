import pandas as pd
import os
import glob

# Configuration
DATASET_DIR = './dataset'
PARTITION_DIR = os.path.join(DATASET_DIR, 'partition')

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

if __name__ == "__main__":
    convert_sicap_to_csv(PARTITION_DIR)
    print("Conversion complete.")