import pandas as pd
import os
import glob # help us extract files path instead of doing it manually
import sqlite3

CYCLE_MAP = {
    "H": "2013-2014",
    "I": "2015-2016",
    "J": "2017-2018",
    "P": "2017-2020",
    "L": "2021-2023"
}
DB_NAME = "nhanes_1st.db"
conn = sqlite3.connect(DB_NAME)
FEATURES_TO_KEEP = ['SEQN', 'RIAGENDR', 'RIDAGEYR', 'RIDRETH3', 'INDFMPIR']

script_dir = os.path.dirname(os.path.abspath(__file__))
FOLDER_PATH = os.path.join(script_dir, "Raw Data", "Demo_data", "*.xpt")

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
    print(f' We have: {len(files_list)} Files in Demographics {folder_Path}') #this will return the number of files that are in demo folder

    for file in files_list: # loop thru every file to put it ingest it and put it in our database table
        
        filename = os.path.basename(file) # extract file name

        df = pd.read_sas(file, format= 'xport')
        df = df[Feature_Names]

        df = cycle_checker(df, filename)

        # Creates table 'Demographics' if it's missing, and if it exist then append the data
        df.to_sql('Demographics', conn, if_exists='append', index=False)
    conn.close()
    print("Finished Ingestion of Demographics data and DB Connection is closed")

raw_Demographics(FOLDER_PATH, FEATURES_TO_KEEP)