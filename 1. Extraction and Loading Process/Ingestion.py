import pandas as pd
import os
import glob # help us extract files path instead of doing it manually
import sqlite3
import numpy as np

CYCLE_MAP = {
    "_H": "2013-2014",
    "_I": "2015-2016",
    "_J": "2017-2018",
    "P_": "2017-2020",
    "_L": "2021-2023"
}

DB_NAME = "nhanes_1st.db"

script_dir = os.path.dirname(os.path.abspath(__file__))
FOLDER_PATH_DB = os.path.join(script_dir, '..', 'databases', DB_NAME)

conn = sqlite3.connect(FOLDER_PATH_DB)

FEATURES_TO_KEEP_DEMO = ['SEQN', 'RIAGENDR', 'RIDAGEYR', 'RIDRETH3', 'INDFMPIR']
FEATURES_TO_KEEP_BMS = ['SEQN', 'BMXBMI', 'BMXHT', 'BMXWAIST']
#FEATURES_TO_KEEP_VITALS = ['SEQN', 'BPXPLS', 'BPXOPLS'] #BPXOPLS is Oscillometric Measurements, while BPXPLS is done using manual device
FEATURES_TO_KEEP_CBC = ['SEQN', 'LBXWBCSI', 'LBXPLTSI', 'LBXHGB', 'LBXMCVSI']
FEATURES_TO_KEEP_BIO = ['SEQN', 'LBXSCR', 'LBXSASSI', 'LBXSATSI', 'LBXSTB', 
                        'LBXSGTSI', 'LBXSUA', 'LBXSNASI', 'LBXSKSI', 'LBXSGL'
]
FEATURES_TO_KEEP_CHOL = ['SEQN', 'LBXTC']
FEATURES_TO_KEEP_HDLCHOL = ['SEQN', 'LBDHDD']
FEATURES_TO_KEEP_TRIGLY = ['SEQN', 'LBXTR']
FEATURES_TO_KEEP_ACURINE = ['SEQN', 'URXUMA', 'URXUCR']
FEATURES_TO_KEEP_SMOKE = ['SEQN', 'SMQ020', 'SMQ040']
FEATURES_TO_KEEP_ALCHOL = ['SEQN', 'ALQ101', 'ALQ120Q', 'ALQ130']
FEATURES_TO_KEEP_HEARTPROB = ['SEQN', 'MCQ160B', 'MCQ160C', 'MCQ160D', 'MCQ160E', 'MCQ160F']

script_dir = os.path.dirname(os.path.abspath(__file__))
FOLDER_PATH_DEMO = os.path.join(script_dir, "Raw Data", "Demo_data", "*.xpt")
FOLDER_PATH_BMS = os.path.join(script_dir, "Raw Data", "bodyMeasures_data", "*.xpt")
FOLDER_PATH_VITALS = os.path.join(script_dir, "Raw Data", "BloodPressure_data", "*.xpt")
FOLDER_PATH_CBC = os.path.join(script_dir, "Raw Data", "BloodCount_data", "*.xpt")
FOLDER_PATH_BIO = os.path.join(script_dir, "Raw Data", "BioChemistry_data", "*.xpt")
FOLDER_PATH_CHOL = os.path.join(script_dir, "Raw Data", "Cholesterol_data", "*.xpt")
FOLDER_PATH_HDLCHOL = os.path.join(script_dir, "Raw Data", "HDLChol_data", "*.xpt")
FOLDER_PATH_TRIGLY = os.path.join(script_dir, "Raw Data", "Triglycerides", "*.xpt")
FOLDER_PATH_ACURINE = os.path.join(script_dir, "Raw Data", "AlbuminNCreatinine_data", "*.xpt")
FOLDER_PATH_SMOKE = os.path.join(script_dir, "Raw Data", "Smoking_data", "*.xpt")
FOLDER_PATH_ALCHOL = os.path.join(script_dir, "Raw Data", "Alcohol_data", "*.xpt")
FOLDER_PATH_HEARTPROB = os.path.join(script_dir, "Raw Data", "heart_related_data", "*.xpt")

def cycle_checker(df: pd.DataFrame, filename: str) -> pd.DataFrame:
        found_cycle = False

        for letter, year in CYCLE_MAP.items():

            if letter in filename:
                df['Cycle'] = year
                found_cycle = True
                return df

        if found_cycle is False:
            raise ValueError(f"Couldnt find a matching letter for the following file {filename}")

def raw_Demographics(folder_Path: str, Feature_Names: list[str]) -> None:

    files_list = glob.glob(folder_Path) # get a list of files that end with .xpt
    print(f' We have: {len(files_list)} Files in Demographics {folder_Path}\n') #this will return the number of files that are in demo folder

    for file in files_list: # loop thru every file to put it ingest it and put it in our database table
        
        filename = os.path.basename(file) # extract file name

        df = pd.read_sas(file, format= 'xport')
        df = df[Feature_Names]

        df = cycle_checker(df, filename)

        # Creates table 'Demographics' if it's missing, and if it exist then append the data
        df.to_sql('Demographics', conn, if_exists='append', index=False)
    conn.close()
    print("Finished Ingestion of Demographics data and DB Connection is closed")
#raw_Demographics(FOLDER_PATH_DEMO, FEATURES_TO_KEEP_DEMO) DO NOT RUN TWICE OR IT WILL CREATE DUPLICATE DATA

def raw_bodyMeasures(folder_Path: str, Feature_Names: list[str]) -> None:
    files_list = glob.glob(folder_Path)
    print(f' We have: {len(files_list)} Files in BodyMeasures {folder_Path}\n')

    for file in files_list:
        filename = os.path.basename(file)

        df = pd.read_sas(file, format= 'xport')
        df = df[Feature_Names]
        df = df.rename(columns={'BMXWAIST': 'BMXWAIST (target)'})

        df = cycle_checker(df, filename)
        df.to_sql('Body Measures', conn, if_exists='append', index=False)
    conn.close()
    print("Finished Ingestion of BodyMeasures data and DB Connection is closed")
#raw_bodyMeasures(FOLDER_PATH_BMS, FEATURES_TO_KEEP_BMS) #DO NOT RUN TWICE OR IT WILL CREATE DUPLICATE DATA


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
        
        df_final.to_sql('Vitals', conn, if_exists='append', index=False)
        
        valid_rows = df_final['Pulse'].count()
        print(f"Processed {filename} ({current_cycle}): Saved {valid_rows} rows. Harmonized BP & Pulse.")

    conn.close()
    print("\nFinished Ingestion for Vitals. DB Connection closed.")
raw_Vitals(FOLDER_PATH_VITALS) #DO NOT RUN TWICE OR IT WILL CREATE DUPLICATE DATA

def raw_CBC(folder_Path: str, Feature_Names: list[str]) -> None:

    files_list = glob.glob(folder_Path) # get a list of files that end with .xpt
    print(f' We have: {len(files_list)} Files in CBC {folder_Path}\n') #this will return the number of files that are in demo folder

    for file in files_list: # loop thru every file to put it ingest it and put it in our database table
        
        filename = os.path.basename(file) # extract file name

        df = pd.read_sas(file, format= 'xport')
        df = df[Feature_Names]

        df = cycle_checker(df, filename)

        df.to_sql('Complete Blood Count', conn, if_exists='append', index=False)
    conn.close()
    print("Finished Ingestion of CBC data and DB Connection is closed")
#raw_CBC(FOLDER_PATH_CBC, FEATURES_TO_KEEP_CBC) #DO NOT RUN TWICE OR IT WILL CREATE DUPLICATE DATA

def raw_bioProfile(bio_folder_Path: str, FEATURES_TO_KEEP_BIO: list[str]) -> None:
    bio_files_list = glob.glob(bio_folder_Path)
    print(f'Found {len(bio_files_list)} Biochemistry files.')

    for file in bio_files_list:
        filename = os.path.basename(file)

        df = pd.read_sas(file, format= 'xport')
        df = df[FEATURES_TO_KEEP_BIO]

        df = cycle_checker(df, filename)

        df.to_sql('Biochem Profile', conn, if_exists='append', index=False)
    conn.close()
    print("Finished Ingestion of Bio Profile data and DB Connection is closed")
#raw_bioProfile(FOLDER_PATH_BIO, FEATURES_TO_KEEP_BIO ) # DO NOT RUN TWICE OR IT WILL CREATE DUPLICATE DATA

def raw_CHOL(folder_Path: str, Feature_Names: list[str]) -> None:

    files_list = glob.glob(folder_Path)
    print(f' We have: {len(files_list)} Files in chOL {folder_Path}\n') #this will return the number of files that are in demo folder

    for file in files_list: # loop thru every file to put it ingest it and put it in our database table
        
        filename = os.path.basename(file) # extract file name

        df = pd.read_sas(file, format= 'xport')
        df = df[Feature_Names]

        df = cycle_checker(df, filename)

        df.to_sql('Total Cholesterol', conn, if_exists='append', index=False)
    conn.close()
    print("Finished Ingestion of chole data and DB Connection is closed")
#raw_CHOL(FOLDER_PATH_CHOL, FEATURES_TO_KEEP_CHOL) #DO NOT RUN TWICE OR IT WILL CREATE DUPLICATE DATA

def raw_HDLCHOL(folder_Path: str, Feature_Names: list[str]) -> None:

    files_list = glob.glob(folder_Path)
    print(f' We have: {len(files_list)} Files in HDLCHOL {folder_Path}\n') #this will return the number of files that are in demo folder

    for file in files_list: # loop thru every file to put it ingest it and put it in our database table
        
        filename = os.path.basename(file) # extract file name

        df = pd.read_sas(file, format= 'xport')
        df = df[Feature_Names]

        df = cycle_checker(df, filename)
        df.rename(columns={"LBDHDD": "LBDHDD (target)"}, inplace=True)

        df.to_sql('HDL_Cholesterol', conn, if_exists='append', index=False)
    conn.close()
    print("Finished Ingestion of HDLCHOL data and DB Connection is closed")
#raw_HDLCHOL(FOLDER_PATH_HDLCHOL, FEATURES_TO_KEEP_HDLCHOL) #DO NOT RUN TWICE OR IT WILL CREATE DUPLICATE DATA

def raw_Triglycerides(folder_Path: str, Feature_Names: list[str]) -> None:
    
    files_list = glob.glob(folder_Path)
    print(f' We have: {len(files_list)} Files in Triglycerides {folder_Path}\n') #this will return the number of files that are in demo folder

    for file in files_list: 
        filename = os.path.basename(file)

        current_cycle = None
        for key in CYCLE_MAP.keys():
            if key in filename:
                current_cycle = key
                break
        

        if current_cycle is None:
            print(f"Skipping {filename}: cycle not recognized.")
            raise ValueError("fix ur fucking shit no cycle recognized")

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

        df.to_sql('Triglycerides', conn, if_exists='append', index=False)
        print(f"Ingested {filename}")

    # conn.close() # Best practice: Close connection outside the loop/function
    print("Finished Ingestion of raw_Triglycerides data")
#raw_Triglycerides(FOLDER_PATH_TRIGLY, FEATURES_TO_KEEP_TRIGLY) #DO NOT RUN TWICE OR IT WILL CREATE DUPLICATE DATA

def raw_ACURINE(folder_Path: str, Feature_Names: list[str]) -> None:

    files_list = glob.glob(folder_Path)
    print(f' We have: {len(files_list)} Files in AlbuminNCreatinine_Data {folder_Path}\n') #this will return the number of files that are in demo folder

    for file in files_list: # loop thru every file to put it ingest it and put it in our database table
        
        filename = os.path.basename(file) # extract file name

        df = pd.read_sas(file, format= 'xport')
        df = df[Feature_Names]

        df = cycle_checker(df, filename)
        df.rename(columns={"URXUMA": "URXUMA (target)"}, inplace=True)
        df.rename(columns={"URXUCR": "URXUCR (target)"}, inplace=True)

        df.to_sql('Albumin_Creatinie', conn, if_exists='append', index=False)
    conn.close()
    print("Finished Ingestion of ACURINE data and DB Connection is closed")
#raw_ACURINE(FOLDER_PATH_ACURINE, FEATURES_TO_KEEP_ACURINE) #DO NOT RUN TWICE OR IT WILL CREATE DUPLICATE DATA

def raw_SMOKE(folder_Path: str, Feature_Names: list[str]) -> None:

    files_list = glob.glob(folder_Path)
    print(f' We have: {len(files_list)} Files in SMOKE_data {folder_Path}\n') #this will return the number of files that are in demo folder

    for file in files_list: # loop thru every file to put it ingest it and put it in our database table
        
        filename = os.path.basename(file) # extract file name

        df = pd.read_sas(file, format= 'xport')
        df = df[Feature_Names]

        df = cycle_checker(df, filename)

        df.to_sql('Smoke', conn, if_exists='append', index=False)
    conn.close()
    print("Finished Ingestion of Smoke data and DB Connection is closed")
#raw_SMOKE(FOLDER_PATH_SMOKE, FEATURES_TO_KEEP_SMOKE) #DO NOT RUN TWICE OR IT WILL CREATE DUPLICATE DATA


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
        # We will use this mask at the very end to force 0.0
        is_non_drinker = (df['ALQ101'] == 2)


        df['_freq_days_per_week'] = np.nan # Initialize temp column

        if is_new_schema:
            # === Branch B: New Data (2017-2023) ===
            # Input: ALQ121 -> Map to standardized weekly value
            if 'ALQ121' in df.columns:
                df['_freq_days_per_week'] = df['ALQ121'].map(ALQ121_WEEKLY_MAP)
        
        else:

            # Inputs: ALQ120Q (Quantity) & ALQ120U (Unit)
            if 'ALQ120Q' in df.columns and 'ALQ120U' in df.columns:
                # Clean 999 (Don't know) in Quantity
                quantity = df['ALQ120Q'].replace(999, np.nan)
                
                conditions = [
                    (df['ALQ120U'] == 1), # Week
                    (df['ALQ120U'] == 2), # Month (Divide by 4.3 avg weeks)
                    (df['ALQ120U'] == 3)  # Year (Divide by 52 weeks)
                ]
                choices = [
                    quantity,
                    quantity / 4.3,
                    quantity / 52
                ]
                df['_freq_days_per_week'] = np.select(conditions, choices, default=np.nan)

        if 'ALQ130' in df.columns:
            # Validation: Treat 999 (Don't know) and 777 (Refused) as NaN
            intensity = df['ALQ130'].replace({777: np.nan, 999: np.nan})
        else:
            intensity = np.nan

        df['Alcohol_Drinks_Per_Week'] = df['_freq_days_per_week'] * intensity


        df.loc[is_non_drinker, 'Alcohol_Drinks_Per_Week'] = 0.0


        # --- 4. Clean up columns for SQL ---
        df = cycle_checker(df, filename)

        # Ensure all FINAL_COLUMNS exist
        for col in FINAL_COLUMNS:
            if col not in df.columns:
                df[col] = None
        
        # Select final columns in order
        df_final = df[FINAL_COLUMNS].copy()

        try:
            df_final.to_sql('AlcholUsage', conn, if_exists='append', index=False)
            print(f"--> Ingested {filename} | New Schema: {is_new_schema}")
        except Exception as e:
            print(f"Error processing {filename}: {e}")

    print("Finished Ingestion of Alcohol data")
#raw_Alchol(FOLDER_PATH_ALCHOL, conn) #DO NOT RUN TWICE OR IT WILL CREATE DUPLICATE DATA


def raw_HEARTPROB(folder_Path: str, Feature_Names: list[str]) -> None:

    files_list = glob.glob(folder_Path)
    print(f' We have: {len(files_list)} Files in HEARTPROB_DATA {folder_Path}\n') #this will return the number of files that are in demo folder

    for file in files_list: # loop thru every file to put it ingest it and put it in our database table
        
        filename = os.path.basename(file) # extract file name

        df = pd.read_sas(file, format= 'xport')
        df = df[Feature_Names]

        df = cycle_checker(df, filename)

        df.to_sql('HeartQuestions', conn, if_exists='append', index=False)
    conn.close()
    print("Finished Ingestion of HEARTPROB data and DB Connection is closed")
#raw_HEARTPROB(FOLDER_PATH_HEARTPROB, FEATURES_TO_KEEP_HEARTPROB) #DO NOT RUN TWICE OR IT WILL CREATE DUPLICATE DATA