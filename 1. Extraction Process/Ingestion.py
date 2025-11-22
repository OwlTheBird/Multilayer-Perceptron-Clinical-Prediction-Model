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
FEATURES_TO_KEEP_BMS = ['SEQN', 'BMXBMI', 'BMXHT']
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

        df = cycle_checker(df, filename)
        df.to_sql('Body Measures', conn, if_exists='append', index=False)
    conn.close()
    print("Finished Ingestion of BodyMeasures data and DB Connection is closed")
#raw_bodyMeasures(FOLDER_PATH_BMS, FEATURES_TO_KEEP_BMS) DO NOT RUN TWICE OR IT WILL CREATE DUPLICATE DATA


# this is a Data Harmonization problem
MANUAL_COL = 'BPXPLS'
OSCILLO_COLS = ['BPXOPLS1', 'BPXOPLS2', 'BPXOPLS3'] 
FINAL_FEATURE = 'Pulse'

def raw_Vitals(folder_Path: str) -> None:
    files_list = glob.glob(folder_Path)
    print(f'Found {len(files_list)} files in {folder_Path}\n')

    for file in files_list:
        filename = os.path.basename(file)
        
        df = pd.read_sas(file, format='xport')
        df = cycle_checker(df, filename) 
        current_cycle = df['Cycle'].iloc[0]
        
        # Ensure Columns Exist and fill missing with NaN values
        if MANUAL_COL not in df.columns:
            df[MANUAL_COL] = np.nan
        for col in OSCILLO_COLS:
            if col not in df.columns:
                df[col] = np.nan

        # we create harmonized pulse to fix the data harmonization problem

        # calculate mean of oscillo(modern and more accurate) readings
        oscillo_mean = df[OSCILLO_COLS].mean(axis=1)
        
        # create pulse and fill NaN in Manual with oscillo Mean
        df[FINAL_FEATURE] = df[MANUAL_COL].fillna(oscillo_mean)

        # FEATURE ENGINEERING: Create "Method Flag"
        # Logic: If 'BPXPLS' (Manual) is NOT Null, then Flag = 0 (Manual).
        # Otherwise, it is Oscillometric, so Flag = 1.
        df['Is_Oscillometric'] = np.where(df[MANUAL_COL].notna(), 0, 1)


        cols_to_keep = ['SEQN', FINAL_FEATURE, 'Is_Oscillometric', 'Cycle']
        df = df[cols_to_keep]
        
        # 6. Save to SQL
        df.to_sql('Vitals', conn, if_exists='append', index=False)
        
        valid_rows = df[FINAL_FEATURE].count()
        print(f"Processed {filename} ({current_cycle}): Saved {valid_rows} rows. Cols: {list(df.columns)}")

    conn.close()
    print("\nFinished Ingestion for Vitals, DB Connection closed. Raw columns dropped. Flag added.")
raw_Vitals(FOLDER_PATH_VITALS)