import pandas as pd
import os
import sqlite3
import duckdb

DB_NAME = "nhanes_1st.db"

script_dir = os.path.dirname(os.path.abspath(__file__))
FOLDER_PATH_DB = os.path.join(script_dir, '..', 'databases', DB_NAME)
SQL_FILE_PATH = os.path.join(script_dir, '..', 'databases', 'SQL_queries', 'join_all_tables.sql')
SQL_CARDIO_FILE_PATH = os.path.join(script_dir, '..', 'databases', 'SQL_queries', 'cardiov_target.sql')
SQL_METABOLIC_FILE_PATH = os.path.join(script_dir, '..', 'databases', 'SQL_queries', 'metabolic_target.sql')
SQL_KIDNEY_FILE_PATH = os.path.join(script_dir, '..', 'databases', 'SQL_queries', 'kidney_target.sql')
SQL_LIVER_FILE_PATH = os.path.join(script_dir, '..', 'databases', 'SQL_queries', 'liver_target.sql')

conn = sqlite3.connect(FOLDER_PATH_DB)

con = duckdb.connect()

with open(SQL_FILE_PATH, 'r') as file:
    query_join_all_tables = file.read()

df = pd.read_sql_query(query_join_all_tables, conn)
df = df.loc[:, ~df.columns.duplicated()] # because i did left join on all tables it created duplicate SEQN for each table pandas read it

def add_bp_averages(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates the final Systolic and Diastolic averages based on 
    available harmonized readings.
    """
    # Define the column groups
    sys_cols = ['BPXSY1 (target)', 'BPXSY2 (target)', 'BPXSY3 (target)']
    dia_cols = ['BPXDI1 (target)', 'BPXDI2 (target)', 'BPXDI3 (target)']
    
    # Calculate row-wise mean (axis=1).
    # Pandas .mean() automatically handles NaNs:
    # - If 3 values exist: (V1+V2+V3)/3
    # - If 1 value is missing: (V1+V3)/2
    # - If all are missing: NaN
    df['Final_Harmonized_Systolic'] = df[sys_cols].mean(axis=1)
    df['Final_Harmonized_Diastolic'] = df[dia_cols].mean(axis=1)
    
    return df
df = add_bp_averages(df)

def cardio_target(df: pd.DataFrame , query_path: str )-> pd.DataFrame:
    with open(query_path, 'r') as file:
        query_cardio = file.read()

    con.register('HeartQuestions', df)
    
    result_df = con.execute(query_cardio).df()

    df = df.merge(result_df[['SEQN', 'has_cardiovascular_disease']], on='SEQN', how='left')
    return df
df = cardio_target(df, SQL_CARDIO_FILE_PATH)
print(df['has_cardiovascular_disease'].isnull().sum())


def metabolic_target(df: pd.DataFrame, query_path: str) -> pd.DataFrame:
    with open(query_path, 'r') as file:
        query_metabolic = file.read()

    con.register('table_df', df)
    
    result_df = con.execute(query_metabolic).df()

    df = df.merge(result_df[['SEQN', 'high_waist_circumference', 'high_triglycerides_mg_dl', 'low_hdl_mg_dl', 'high_blood_pressure', 'high_glucose_mg_dl']], on='SEQN', how='left')
    return df
df = metabolic_target(df, SQL_METABOLIC_FILE_PATH)
#print(df.sample(n=30))
print(df[['high_waist_circumference', 'high_triglycerides_mg_dl', 'low_hdl_mg_dl', 'high_blood_pressure', 'high_glucose_mg_dl']].isnull().sum())

def kidney_target(df: pd.DataFrame, query_path: str) -> pd.DataFrame:
    with open(query_path, 'r') as file:
        query_kidney = file.read()

    con.register('table_df', df)

    result_df = con.execute(query_kidney).df()
    df = df.merge(result_df[["SEQN", "ACR_mg_g", "kidney_acr_mg_g"]], on= 'SEQN', how='left')

    return df
df = kidney_target(df, SQL_KIDNEY_FILE_PATH)
#print(df.sample(n=30))
print(df[['ACR_mg_g', 'kidney_acr_mg_g']].isnull().sum())

def liver_target(df: pd.DataFrame, query_path: str) -> pd.DataFrame:
    with open(query_path, 'r') as file:
        query_liver = file.read()
    con.register('table_df', df)

    result_df = con.execute(query_liver).df()
    df = df.merge(result_df[["SEQN", "liver_alt_U_L"]], on= 'SEQN', how='left')

    return df
df = liver_target(df, SQL_LIVER_FILE_PATH)
#print(df.sample(n=30))
print(df[['liver_alt_U_L']].isnull().sum())


inputs = [
    'RIDAGEYR', 'RIAGENDR', 'RIDRETH3', 'INDFMPIR', 'BMXBMI', 'BMXHT', 
    'Pulse', 'LBXWBCSI', 'LBXPLTSI', 'LBXHGB', 'LBXMCVSI', 
    'LBXSCR', 'LBXSASSI', 'LBXSTB', 'LBXSGTSI', 'LBXSUA', 
    'LBXSNASI', 'LBXSKSI', 'LBXTC', 'Alcohol_Drinks_Per_Week', 'SMQ040'
]

targets = [
    'has_cardiovascular_disease', 
    'high_waist_circumference', 'high_triglycerides_mg_dl', 'low_hdl_mg_dl', 'high_blood_pressure', 'high_glucose_mg_dl',
    'kidney_acr_mg_g', 'liver_alt_U_L'
]
total = inputs + targets
df = df[total]

# ===== COLUMN RENAMING (Single Source of Truth) =====
# Rename input feature columns to standardized names before saving to database
# Note: Target columns already have standardized names from SQL queries
COLUMN_RENAMING = {
    # Input Features
    'RIDAGEYR': 'age',
    'RIAGENDR': 'gender',
    'RIDRETH3': 'ethnicity',
    'INDFMPIR': 'income_ratio',
    'BMXBMI': 'body_mass_index',
    'BMXHT': 'height_cm',
    'Pulse': 'heart_rate_bpm',
    'LBXWBCSI': 'white_blood_cells_count',
    'LBXPLTSI': 'platelets_count',
    'LBXHGB': 'hemoglobin_g_dl',
    'LBXMCVSI': 'mean_corpuscular_volume_fL',
    'LBXSCR': 'creatinine_mg_dl',
    'LBXSASSI': 'liver_ast_U_L',
    'LBXSTB': 'bilirubin_mg_dl',
    'LBXSGTSI': 'liver_ggt_U_L',
    'LBXSUA': 'uric_acid_mg_dl',
    'LBXSNASI': 'sodium_mmol_L',
    'LBXSKSI': 'potassium_mmol_L',
    'LBXTC': 'cholesterol_mg_dl',
    'Alcohol_Drinks_Per_Week': 'alcohol_drinks_per_week',
    'SMQ040': 'smoking_status'
}

# Rename columns (only renames input features; targets already have correct names)
df = df.rename(columns=COLUMN_RENAMING)
print("✓ Columns renamed to standardized names")

df.to_sql('raw_dataset', conn, if_exists='replace', index=False)