import pandas as pd
import os
import glob # help us extract files path instead of doing it manually
import sqlite3
import numpy as np
import json 

script_dir = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(script_dir, "ELT_Config.json"), "r") as f:
    config = json.load(f)

CYCLE_MAP = {
    "_H": "2013-2014",
    "_I": "2015-2016",
    "_J": "2017-2018",
    "P_": "2017-2020",
    "_L": "2021-2023"
}

DB_NAME = "nhanes_1st.db"

FOLDER_PATH_DB = os.path.join(script_dir, *config["DBs"]["Paths"]["nhanes_1st"].split("/"))

conn = sqlite3.connect(FOLDER_PATH_DB)


def cycle_checker(df: pd.DataFrame, filename: str) -> pd.DataFrame:
        found_cycle = False

        for letter, year in CYCLE_MAP.items():

            if letter in filename:
                df['Cycle'] = year
                found_cycle = True
                return df

        if found_cycle is False:
            raise ValueError(f"Couldnt find a matching letter for the following file {filename}")


def transfer_to_db(folder_Path: str, Feature_Names: list[str], table_name: str, rename_flag: bool, renameDIC: dict) -> None:

    files_list = glob.glob(folder_Path) # get a list of files that end with .xpt
    print(f' We have: {len(files_list)} Files in {table_name} {folder_Path}\n') #this will return the number of files

    all_dfs = []  # Collect all DataFrames
    for file in files_list: # loop thru every file to put it ingest it and put it in our database table
        
        filename = os.path.basename(file) # extract file name

        df = pd.read_sas(file, format= 'xport')
        df = df[Feature_Names]
        if rename_flag:
            for key in renameDIC.keys():
                df = df.rename(columns={key: renameDIC[key]})

        df = cycle_checker(df, filename)
        all_dfs.append(df)

    # Concatenate all DataFrames and write once (idempotent: replace entire table)
    if all_dfs:
        combined_df = pd.concat(all_dfs, ignore_index=True)
        combined_df.to_sql(table_name, conn, if_exists='replace', index=False)
    print(f"Finished Ingestion of {table_name} data\n")

####################################### Demographic
transfer_to_db(os.path.join(script_dir, *config["Paths"]["DEMO"].split("/"))\
                , config["Features"]["DEMO"]\
                    , "Demographics", False, None)

####################################### Body Measures
bodyMeasures_Dic ={
    'BMXWAIST': 'BMXWAIST (target)'
}
transfer_to_db(os.path.join(script_dir, *config["Paths"]["BMS"].split("/"))\
                , config["Features"]["BMS"]\
                    , "Body Measures", True, bodyMeasures_Dic)

####################################### CBC
transfer_to_db(os.path.join(script_dir, *config["Paths"]["CBC"].split("/"))\
                , config["Features"]["CBC"]\
                    , "Complete Blood Count", False, None)

####################################### bio chem
bioProfile_Dic = {
    'LBXSATSI': 'LBXSATSI (target)'
}
transfer_to_db(os.path.join(script_dir, *config["Paths"]["BIO"].split("/"))\
                , config["Features"]["BIO"]\
                    , "BiochemProfile", True, bioProfile_Dic)

####################################### Chol
transfer_to_db(os.path.join(script_dir, *config["Paths"]["CHOL"].split("/"))\
                , config["Features"]["CHOL"]
                    , "Total Cholesterol", False, None)

####################################### HDLCHOL
HDLCHOL_Dic = {
    "LBDHDD": "LBDHDD (target)"
}
transfer_to_db(os.path.join(script_dir, *config["Paths"]["HDLCHOL"].split("/"))\
                , config["Features"]["HDLCHOL"]\
                    , "HDL_Cholesterol", True, HDLCHOL_Dic)

####################################### ACURINE
ACURINE_Dic = {
    "URXUMA": "URXUMA (target)",
    "URXUCR": "URXUCR (target)"
}
transfer_to_db(os.path.join(script_dir, *config["Paths"]["ACURINE"].split("/"))\
                , config["Features"]["ACURINE"]
                    , "Albumin_Creatinie", True, ACURINE_Dic)

####################################### SMOKE
transfer_to_db(os.path.join(script_dir, *config["Paths"]["SMOKE"].split("/"))\
                , config["Features"]["SMOKE"]\
                    , "Smoke", False, None)

####################################### HEARTPROB
transfer_to_db(os.path.join(script_dir, *config["Paths"]["HEARTPROB"].split("/"))\
                , config["Features"]["HEARTPROB"]\
                    , "HeartQuestions", False, None)

####################################### GLUC
transfer_to_db(os.path.join(script_dir, *config["Paths"]["GLUC"].split("/"))\
                , config["Features"]["GLUC"]\
                    , "Glucose", False, None)

####################################### Fasting
transfer_to_db(os.path.join(script_dir, *config["Paths"]["FASTING"].split("/")) \
             , config["Features"]["FASTING"]\
                , "Fasting", False, None)


# this is a Data Harmonization problem
MANUAL_PULSE = 'BPXPLS'
OSCILLO_PULSE_COLS = ['BPXOPLS1', 'BPXOPLS2', 'BPXOPLS3']

# Manual
MANUAL_SY = ['BPXSY1', 'BPXSY2', 'BPXSY3']
MANUAL_DI = ['BPXDI1', 'BPXDI2', 'BPXDI3']
# Oscillometric
OSCILLO_SY = ['BPXOSY1', 'BPXOSY2', 'BPXOSY3']
OSCILLO_DI = ['BPXODI1', 'BPXODI2', 'BPXODI3']

def raw_Vitals(folder_Path: str) -> None:
    files_list = glob.glob(folder_Path)
    print(f'Found {len(files_list)} files in {folder_Path}\n')

    all_dfs = []  # Collect all DataFrames
    for file in files_list:
        filename = os.path.basename(file)
        
        df = pd.read_sas(file, format='xport')
        df = cycle_checker(df, filename) 
        current_cycle = df['Cycle'].iloc[0]
        
        all_needed_cols = [MANUAL_PULSE] + OSCILLO_PULSE_COLS + \
                          MANUAL_SY + MANUAL_DI + OSCILLO_SY + OSCILLO_DI
        
        for col in all_needed_cols:
            if col not in df.columns:
                df[col] = np.nan

        # Logic: If Manual Systolic 1 (BPXSY1) exists, we assume Manual (0).
        # If it is Null, we assume we are relying on Oscillometric (1).
        df['Is_Oscillometric'] = np.where(df[MANUAL_SY[0]].notna(), 0, 1)

        # Calculate mean of oscillo readings
        oscillo_pulse_mean = df[OSCILLO_PULSE_COLS].mean(axis=1)
        # Fill Manual Pulse gaps with Oscillo Mean
        df['Pulse'] = df[MANUAL_PULSE].fillna(oscillo_pulse_mean)

        #HARMONIZE BLOOD PRESSURE
        # We loop through 1, 2, 3. 
        # We fill the Manual column with the Oscillo value if Manual is missing.
        for i in range(3):
            # Harmonize Systolic: BPXSY1 = BPXSY1 (Manual) filled with BPXOSY1 (Oscillo)
            df[MANUAL_SY[i]] = df[MANUAL_SY[i]].fillna(df[OSCILLO_SY[i]])
            
            # Harmonize Diastolic: BPXDI1 = BPXDI1 (Manual) filled with BPXODI1 (Oscillo)
            df[MANUAL_DI[i]] = df[MANUAL_DI[i]].fillna(df[OSCILLO_DI[i]])

        # We now treat MANUAL_SY/DI cols as our "Master" cols
        final_cols = ['SEQN', 'Cycle', 'Is_Oscillometric', 'Pulse'] + MANUAL_SY + MANUAL_DI
        
        df_final = df[final_cols].copy()

        rename_dict = {}
        for col in MANUAL_SY + MANUAL_DI:
            rename_dict[col] = f"{col} (target)"
        df_final = df_final.rename(columns=rename_dict)
        
        all_dfs.append(df_final)
        
        valid_rows = df_final['Pulse'].count()
        print(f"Processed {filename} ({current_cycle}): {valid_rows} rows. Harmonized BP & Pulse.")

    # Concatenate all DataFrames and write once (idempotent: replace entire table)
    if all_dfs:
        combined_df = pd.concat(all_dfs, ignore_index=True)
        combined_df.to_sql('Vitals', conn, if_exists='replace', index=False)
    print(f"\nFinished Ingestion for Vitals. Total rows: {len(combined_df)}")
raw_Vitals(os.path.join(script_dir, *config["Paths"]["Vitals"].split("/")))



def raw_Triglycerides(folder_Path: str, Feature_Names: list[str]) -> None:
    
    files_list = glob.glob(folder_Path)
    print(f' We have: {len(files_list)} Files in Triglycerides {folder_Path}\n')

    all_dfs = []  # Collect all DataFrames
    for file in files_list: 
        filename = os.path.basename(file)

        current_cycle = None
        for key in CYCLE_MAP.keys():
            if key in filename:
                current_cycle = key
                break

        if current_cycle is None:
            print(f"Skipping {filename}: cycle not recognized.")
            raise ValueError(f"Unrecognized cycle in file: {filename}")

        # The 2021-2023 cycle (_L) which has different columns
        if current_cycle == '_L':
            df = pd.read_sas(file, format='xport')
            df = df[['SEQN', 'LBXTLG']] 
            df.rename(columns={'LBXTLG': 'LBXTLG (target)'}, inplace=True)
        
        # All other cycles (_H, _I, _J, P_)
        else:
            df = pd.read_sas(file, format='xport')
            df = df[[c for c in Feature_Names if c in df.columns]]
            if 'LBXTR' in df.columns:
                df.rename(columns={'LBXTR': 'LBXTLG (target)'}, inplace=True)

        df = cycle_checker(df, filename)
        all_dfs.append(df)
        print(f"Ingested {filename}")

    # Concatenate all DataFrames and write once (idempotent: replace entire table)
    if all_dfs:
        combined_df = pd.concat(all_dfs, ignore_index=True)
        combined_df.to_sql('Triglycerides', conn, if_exists='replace', index=False)
    print(f"Finished Ingestion of Triglycerides data. Total rows: {len(combined_df)}")
raw_Triglycerides(os.path.join(script_dir, *config["Paths"]["TRIGLY"].split("/"))\
                , config["Features"]["TRIGLY"])

FINAL_COLUMNS = ['SEQN', 'Cycle', 'ALQ101', 'ALQ130', 'Alcohol_Drinks_Per_Week']

# MAP: Converts New Schema Categories (ALQ121) -> Days Per Week
ALQ121_WEEKLY_MAP = {
    1: 7.0,   # Every day
    2: 5.0,   # Nearly every day
    3: 3.5,   # 3-4 times a week
    4: 2.0,   # 2 times a week
    5: 1.0,   # Once a week
    6: 0.5,   # 2-3 times a month
    7: 0.25,  # Once a month
    8: 0.05,  # 7-11 times in last year (Less than once a month)
    9: 0.05,  # 1-2 times in last year (Less than once a month)
    10: 0.0,  # Never in the last year
    77: None, # Refused
    99: None  # Don't know
}

def raw_Alchol(folder_Path: str, conn) -> None:
    
    files_list = glob.glob(folder_Path)
    print(f' We have: {len(files_list)} Files in Alcohol_data\n')

    all_dfs = []  # Collect all DataFrames
    for file in files_list:
        filename = os.path.basename(file)

        current_cycle_suffix = None
        for suffix in CYCLE_MAP.keys():
            if suffix in filename:
                current_cycle_suffix = suffix
                break
        
        if not current_cycle_suffix:
            print(f"Skipping {filename}: Unknown cycle.")
            continue

        # _H (2013) and _I (2015) are Old Schema. _J, P_, _L are New Schema.
        is_new_schema = current_cycle_suffix in ['_J', 'P_', '_L']

        df = pd.read_sas(file, format='xport')

        # Rename ALQ111 (New Screener) to ALQ101 (Old Screener) for consistent Phase 1 logic
        if 'ALQ111' in df.columns:
            df.rename(columns={'ALQ111': 'ALQ101'}, inplace=True)

        # Ensure ALQ101 exists (fill with NaN if missing to prevent errors)
        if 'ALQ101' not in df.columns:
            df['ALQ101'] = np.nan

        # Identify confirmed non-drinkers (Value 2 = No)
        is_non_drinker = (df['ALQ101'] == 2)

        df['_freq_days_per_week'] = np.nan # Initialize temp column

        if is_new_schema:
            # === Branch B: New Data (2017-2023) ===
            if 'ALQ121' in df.columns:
                df['_freq_days_per_week'] = df['ALQ121'].map(ALQ121_WEEKLY_MAP)
        else:
            # Inputs: ALQ120Q (Quantity) & ALQ120U (Unit)
            if 'ALQ120Q' in df.columns and 'ALQ120U' in df.columns:
                quantity = df['ALQ120Q'].replace(999, np.nan)
                conditions = [
                    (df['ALQ120U'] == 1), # Week
                    (df['ALQ120U'] == 2), # Month
                    (df['ALQ120U'] == 3)  # Year
                ]
                choices = [quantity, quantity / 4.3, quantity / 52]
                df['_freq_days_per_week'] = np.select(conditions, choices, default=np.nan)

        if 'ALQ130' in df.columns:
            intensity = df['ALQ130'].replace({777: np.nan, 999: np.nan})
        else:
            intensity = np.nan

        df['Alcohol_Drinks_Per_Week'] = df['_freq_days_per_week'] * intensity
        df.loc[is_non_drinker, 'Alcohol_Drinks_Per_Week'] = 0.0

        # Clean up columns for SQL
        df = cycle_checker(df, filename)

        # Ensure all FINAL_COLUMNS exist
        for col in FINAL_COLUMNS:
            if col not in df.columns:
                df[col] = None
        
        df_final = df[FINAL_COLUMNS].copy()
        all_dfs.append(df_final)
        print(f"--> Ingested {filename} | New Schema: {is_new_schema}")

    # Concatenate all DataFrames and write once (idempotent: replace entire table)
    if all_dfs:
        combined_df = pd.concat(all_dfs, ignore_index=True)
        combined_df.to_sql('AlcholUsage', conn, if_exists='replace', index=False)
    print(f"Finished Ingestion of Alcohol data. Total rows: {len(combined_df)}")
raw_Alchol(os.path.join(script_dir, *config["Paths"]["ALCHOL"].split("/"))\
                , conn)


conn.close()
print("\n=== All Ingestion Complete. Database connection closed. ===")