from bokeh.core.property.primitive import Null
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
conn = sqlite3.connect(DB_NAME)

FEATURES_TO_KEEP_DEMO = ['SEQN', 'RIAGENDR', 'RIDAGEYR', 'RIDRETH3', 'INDFMPIR']
FEATURES_TO_KEEP_BMS = ['SEQN', 'BMXBMI', 'BMXHT', 'BMXWAIST']
FEATURES_TO_KEEP_VITALS = ['SEQN', 'BPXPLS', 'BPXOPLS'] #BPXOPLS is Oscillometric Measurements, while BPXPLS is done using manual device


script_dir = os.path.dirname(os.path.abspath(__file__))
FOLDER_PATH_DEMO = os.path.join(script_dir, "Raw Data", "Demo_data", "*.xpt")
FOLDER_PATH_BMS = os.path.join(script_dir, "Raw Data", "bodyMeasures_data", "*.xpt")
FOLDER_PATH_VITALS = os.path.join(script_dir, "Raw Data", "BloodPressure_data", "*.xpt")

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
#raw_bodyMeasures(FOLDER_PATH_BMS, FEATURES_TO_KEEP_BMS) DO NOT RUN TWICE OR IT WILL CREATE DUPLICATE DATA


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