import pandas as pd
import glob
from typing import List


FILE_PATTERN_SMOKE = "Questionnaires_data/Smoking_data/*.xpt"
FILE_PATTERN_ALCHOL = "Questionnaires_data/Physical Alcohol_data/*.xpt"
FILE_PATTERN_PHYSICALACTIVE = "Questionnaires_data/Physical Activity_data/*.xpt"

FEATURES_TO_KEEP_SMOKE = ['SEQN', 'SMQ020', 'SMQ040']
FEATURES_TO_KEEP_ALCHOL = ['SEQN', 'ALQ130']
FEATURES_TO_KEEP_PHYSICALACTIVE = ['SEQN', 'PAQ650', 'PAQ655', 'PAQ665', 'PAQ670']

OUTPUT_FILENAME = "Questionnaire_preproccesed_data.csv"

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

def data_combination(smoke: pd.DataFrame, alchol: pd.DataFrame, physical: pd.DataFrame) -> pd.DataFrame:
     merge = pd.merge(smoke, alchol, on='SEQN', how='inner')
     mergeV2 = pd.merge(merge, physical, on='SEQN', how='inner')
     return mergeV2

# --- Main script execution ---
def questionnaires_pipeline(file_pattern_smoke=FILE_PATTERN_SMOKE, file_pattern_alchol=FILE_PATTERN_ALCHOL, file_pattern_physical=FILE_PATTERN_PHYSICALACTIVE,
                            features_to_keep_smoke=FEATURES_TO_KEEP_SMOKE, features_to_keep_alchol=FEATURES_TO_KEEP_ALCHOL,features_to_keep_physical=FEATURES_TO_KEEP_PHYSICALACTIVE,
                             output_name= OUTPUT_FILENAME):
    
    smoke_df = data_ingestion(file_pattern_smoke, features_to_keep_smoke)
    alchol_df = data_ingestion(file_pattern_alchol, features_to_keep_alchol)
    physical_df = data_ingestion(file_pattern_physical, features_to_keep_physical)

    questionnaire_df = data_combination(smoke_df, alchol_df, physical_df)
    questionnaire_df.to_csv(output_name, index=False)

    print(f"\n Rows of smoke files: {smoke_df.shape[0]}\n")
    print(f"\n Rows of alchol files: {alchol_df.shape[0]}\n")
    print(f"\n Rows of physical files: {physical_df.shape[0]}\n")

    print(f"\n ===== Rows of physical files: {questionnaire_df.shape[0]} ===== \n")
    print("saved the file Questionnaire_preproccesed_data.csv into the same folder as this python file")