import pandas as pd
import numpy as np
import glob
from typing import List

# --- CONFIGURATION ---
BP_FILE_PATTERN = "Examination&Medication_data/Blood Pressure_data/*BPX*.XPT"
MED_FILE_PATTERN = "Examination&Medication_data/Prescription Drugs_data/*.XPT"
OUTPUT_FILENAME = "Examination&Medication_preproccesed_data.csv"

def blood_pressure_pipeline(file_pattern: str) -> pd.DataFrame:
    """
    Ingests, processes, and combines a collection of NHANES blood pressure files,
    handling both mercury (BPX) and oscillometric (BPXO) measurement types.
    """
    print("--- Starting Blood Pressure Data Pipeline ---")
    
    file_paths = sorted(glob.glob(file_pattern))

    if not file_paths:
        raise ValueError(f"No blood pressure files found matching the pattern: {file_pattern}.")
                         
    print(f"Found {len(file_paths)} potential blood pressure files to process.")
    
    list_of_dfs = []

    for path in file_paths:
        df = pd.read_sas(path, format='xport')
        df.columns = [col.upper() for col in df.columns]

        is_oscillometric = 'BPXOSY1' in df.columns
        is_mercury = 'BPXSY1' in df.columns

        if 'SEQN' not in df.columns or not (is_oscillometric or is_mercury):
            print(f"  - SKIPPING: '{path}' is not a valid BP file.")
            continue

        if is_oscillometric:
            print(f"  - Processing '{path}' as Oscillometric (BPXO)")
            systolic_cols = ['BPXOSY1', 'BPXOSY2', 'BPXOSY3']
            diastolic_cols = ['BPXODI1', 'BPXODI2', 'BPXODI3']
            df['BP_Method'] = 1
        else:
            print(f"  - Processing '{path}' as Mercury (BPX)")
            systolic_cols = [col for col in ['BPXSY1', 'BPXSY2', 'BPXSY3'] if col in df.columns]
            diastolic_cols = [col for col in ['BPXDI1', 'BPXDI2', 'BPXDI3'] if col in df.columns]
            df['BP_Method'] = 0

        df['avg_systolic'] = df[systolic_cols].mean(axis=1, skipna=True)
        df['avg_diastolic'] = df[diastolic_cols].mean(axis=1, skipna=True)
        
        features_to_keep = ['SEQN', 'avg_systolic', 'avg_diastolic', 'BP_Method']
        df_subset = df[features_to_keep].copy()
        df_subset.dropna(subset=['avg_systolic', 'avg_diastolic'], how='all', inplace=True)
        
        list_of_dfs.append(df_subset)

    if not list_of_dfs:
        raise RuntimeError("No valid blood pressure data could be processed.")

    print("\nMerging all individual Blood Pressure DataFrames...")
    combined_df = pd.concat(list_of_dfs, ignore_index=True)
    combined_df = combined_df.drop_duplicates(subset=['SEQN'], keep='first')
    
    print("--- Blood Pressure Pipeline Complete ---")
    return combined_df

def medication_pipeline(file_pattern: str) -> pd.DataFrame:
    """
    Ingests and processes prescription medication files, aggregating all drug
    codes for each participant into a list.
    """
    print("\n--- Starting Medication Data Pipeline ---")
    
    file_paths = sorted(glob.glob(file_pattern))

    if not file_paths:
        raise ValueError(f"No medication files found matching the pattern: {file_pattern}.")

    print(f"Found {len(file_paths)} medication files to process.")
    
    list_of_dfs = []
    features_to_keep = ['SEQN', 'RXDDRGID']

    for path in file_paths:
        df = pd.read_sas(path, format='xport')
        df.columns = [col.upper() for col in df.columns]
        df_subset = df[features_to_keep]
        list_of_dfs.append(df_subset)

    print("\nMerging all individual Medication DataFrames...")
    combined_df = pd.concat(list_of_dfs, ignore_index=True)
    
    print("Aggregating medication codes for each participant...")
    # Group by participant and create a list of all their drug codes
    aggregated_meds = combined_df.groupby('SEQN')['RXDDRGID'].apply(list).reset_index()
    aggregated_meds.rename(columns={'RXDDRGID': 'Medication_Codes'}, inplace=True)
    
    print("--- Medication Pipeline Complete ---")
    return aggregated_meds

# --- Main script execution ---
def examination_and_medication_pipeline():
    # Run the blood pressure pipeline
    final_blood_pressure_df = blood_pressure_pipeline(BP_FILE_PATTERN)
    
    # Run the medication pipeline
    final_medication_df = medication_pipeline(MED_FILE_PATTERN)

    # --- Combine the two datasets ---
    print("\n--- Merging Blood Pressure and Medication Data ---")
    # Use a left merge to keep all participants from the BP data
    final_combined_df = pd.merge(final_blood_pressure_df, final_medication_df, on='SEQN', how='left')
    
    # Save the final output
    final_combined_df.to_csv(OUTPUT_FILENAME, index=False)

    # Print Summary
    print(f"\nSuccessfully processed and merged data for {final_combined_df.shape[0]} participants.")
    print(f"Saved the final file to '{OUTPUT_FILENAME}'")