import pandas as pd
import os
import glob

# Configuration
DATASET_DIR = './dataset'
PARTITION_DIR = os.path.join(DATASET_DIR, 'partition')

def convert_sicap_to_csv(partition_folder):
    # Find the excel files (SICAP usually names them Test.xlsx and Train.xlsx)
    excel_files = glob.glob(os.path.join(partition_folder, '*/*.xlsx')) + glob.glob(os.path.join(partition_folder, '*.xlsx'))
    
    if not excel_files:
        print("Error: No .xlsx files found in dataset/partition/. Please check your extraction.")
        return

    print(f"Found partition files: {excel_files}")

    for file_path in excel_files:
        filename = os.path.basename(file_path).lower()
        
        # Determine if this is Train or Test based on filename
        if 'train' in filename:
            output_name = 'Train.csv'
        elif 'test' in filename:
            output_name = 'Test.csv'
        else:
            continue # Skip validation or other files for now if not needed

        print(f"Processing {filename} -> {output_name}...")
        
        # Read the Excel file
        df = pd.read_excel(file_path)
        
        # SICAPv2 Excel usually has columns like 'image_name', 'Gleason_Primary', 'Gleason_Secondary'
        # We need to map this to One-Hot Encoding: NC, G3, G4, G5
        
        # Prepare new dataframe
        new_df = pd.DataFrame()
        new_df['image_name'] = df['image_name']
        
        # Initialize columns with 0
        new_df['NC'] = 0
        new_df['G3'] = 0
        new_df['G4'] = 0
        new_df['G5'] = 0
        
        # Logic to map Gleason scores to G3/G4/G5/NC
        # Note: This logic depends on the specific column names in your SICAP version.
        # Assuming 'Gleason_Primary' exists. If not, we might need to adjust.
        if 'Gleason_Primary' in df.columns:
            # Simple mapping rule (adjust if your project requires different logic)
            # NC usually means Non-Cancerous (Gleason score often NaN or specific label)
            # This is a heuristic; check your wsi_labels if this fails
            
            for index, row in df.iterrows():
                # Example Logic: You might need to check 'Group' or 'Gleason_Score' columns
                # For now, I'll create a placeholder that works if 'NC', 'G3' etc are in the columns
                # OR if there is a 'class' column.
                
                # SOTA Robust method: Check if 'Gleason_Primary' is 3, 4, or 5
                g_score = row.get('Gleason_Primary', 0)
                if pd.isna(g_score) or g_score == 0:
                    new_df.at[index, 'NC'] = 1
                elif g_score == 3:
                    new_df.at[index, 'G3'] = 1
                elif g_score == 4:
                    new_df.at[index, 'G4'] = 1
                elif g_score >= 5:
                    new_df.at[index, 'G5'] = 1
        else:
            # Fallback: Just copying if columns already exist (rare)
            print("Warning: Could not find 'Gleason_Primary' column. Checking for direct labels...")
            # If the excel already has NC/G3/G4/G5 columns, just copy them
            cols = ['NC', 'G3', 'G4', 'G5']
            for c in cols:
                if c in df.columns:
                    new_df[c] = df[c]

        # Save to CSV in the dataset folder
        save_path = os.path.join(DATASET_DIR, output_name)
        new_df.to_csv(save_path, index=False)
        print(f"Saved {save_path}")

if __name__ == "__main__":
    convert_sicap_to_csv(PARTITION_DIR)