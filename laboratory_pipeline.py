import pandas as pd
import glob
from typing import List


FILE_PATTERN_TRIGLYCERIDES = "Laboratory_data/Triglycerides_data/*.xpt"
FILE_PATTERN_HDL_CHOLESTEROL = "Laboratory_data/HDL Cholesterol_data/*.xpt"
FILE_PATTERN_GLUCOSE = "Laboratory_data/Glucose_data/*.xpt"
FILE_PATTERN_FASTING = "Laboratory_data/Fasting Status_data/*.xpt"

FEATURES_TO_KEEP_TRIGLYCERIDES = ['SEQN', 'LBXTR']
FEATURES_TO_KEEP_HDL_CHOLESTEROL = ['SEQN', 'LBDHDD']
FEATURES_TO_KEEP_GLUCOSE = ['SEQN', 'LBXGLU']
FEATURES_TO_KEEP_FASTING = ['SEQN', 'PHAFSTHR']

OUTPUT_FILENAME = "Laboratory_preproccesed_data.csv"

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

def data_combination(TRIGLYCERIDES: pd.DataFrame, HDL_CHOLESTEROL: pd.DataFrame, GLUCOSE: pd.DataFrame, FASTING: pd.DataFrame) -> pd.DataFrame:
     merge = pd.merge(TRIGLYCERIDES, HDL_CHOLESTEROL, on='SEQN', how='inner')
     mergeV2 = pd.merge(merge, GLUCOSE, on='SEQN', how='inner')
     mergeV3 = pd.merge(mergeV2, FASTING, on='SEQN', how='inner')
     return mergeV3

# --- Main script execution ---
def laboratory_pipeline(file_pattern_TRIGLYCERIDES=FILE_PATTERN_TRIGLYCERIDES, file_pattern_HDL_CHOLESTEROL=FILE_PATTERN_HDL_CHOLESTEROL, 
                            file_pattern_GLUCOSE=FILE_PATTERN_GLUCOSE, file_pattern_FASTING=FILE_PATTERN_FASTING,
                            /
                            features_to_keep_TRIGLYCERIDES=FEATURES_TO_KEEP_TRIGLYCERIDES, features_to_keep_HDL_CHOLESTEROL=FEATURES_TO_KEEP_HDL_CHOLESTEROL,
                            features_to_keep_GLUCOSE=FEATURES_TO_KEEP_GLUCOSE, features_to_keep_FASTING=FEATURES_TO_KEEP_FASTING,
                            /
                             output_name= OUTPUT_FILENAME):
    
    TRIGLYCERIDES_df = data_ingestion(file_pattern_TRIGLYCERIDES, features_to_keep_TRIGLYCERIDES)
    HDL_CHOLESTEROL_df = data_ingestion(file_pattern_HDL_CHOLESTEROL, features_to_keep_HDL_CHOLESTEROL)
    GLUCOSE_df = data_ingestion(file_pattern_GLUCOSE, features_to_keep_GLUCOSE)
    FASTING_df = data_ingestion(file_pattern_FASTING, features_to_keep_FASTING)

    laboratory_df = data_combination(TRIGLYCERIDES_df, HDL_CHOLESTEROL_df, GLUCOSE_df, FASTING_df)
    laboratory_df.to_csv(output_name, index=False)

    print(f"\n Rows of TRIGLYCERIDES files: {TRIGLYCERIDES_df.shape[0]}\n")
    print(f"\n Rows of HDL_CHOLESTEROL files: {HDL_CHOLESTEROL_df.shape[0]}\n")
    print(f"\n Rows of GLUCOSE files: {GLUCOSE_df.shape[0]}\n")
    print(f"\n Rows of FASTING files: {FASTING_df.shape[0]}\n")

    print(f"\n ===== Rows of physical files: {laboratory_df.shape[0]} ===== \n")
    print("saved the file Questionnaire_preproccesed_data.csv into the same folder as this python file")