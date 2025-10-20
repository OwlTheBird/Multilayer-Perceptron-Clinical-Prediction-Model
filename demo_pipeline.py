import pandas as pd
import glob
from typing import List


FILE_PATTERN = "Demographics_data/*.xpt"
FEATURES_TO_KEEP = ['SEQN', 'RIAGENDR', 'RIDAGEYR', 'RIDRETH3']
OUTPUT_FILENAME = "demo_preproccesed_data.csv"


def data_ingestion(file_pattern: str, features_to_keep: List[str]) -> pd.DataFrame:
    print("--- Starting Data Ingestion ---") 
    # Find all file paths that match the specified pattern
    file_paths = sorted(glob.glob(file_pattern))

    # Check if any files were found
    if not file_paths:
        raise ValueError(f"No files found matching the pattern: {file_pattern}")
        
    print(f"!!!!!!!!!!! Found {len(file_paths)} files to process.")
    
    list_of_dfs = [] # store df from each file to combine later
    

    for path in file_paths:
        df = pd.read_sas(path, format='xport')
        
        # --- Feature Selection ---
        df_subset = df[features_to_keep]
        list_of_dfs.append(df_subset)
    
    # merge all the individual DataFrames into one single DataFrame
    combined_df = pd.concat(list_of_dfs, ignore_index=True)
    print("Done Merging all into 1 dataFrame")
    
    print("--- Data Ingestion Complete ---")
    return combined_df


# --- Main script execution ---
def demographic_pipeline(file_pattern=FILE_PATTERN, features_to_keep=FEATURES_TO_KEEP, output_name= OUTPUT_FILENAME):
    combined_df = data_ingestion(file_pattern, features_to_keep)
    combined_df.to_csv(output_name, index=False)
    print("saved the file demo_preproccesed_data.csv into the same folder as this python file")