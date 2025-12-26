"""
02_harmonization.py - Data Harmonization Pipeline

Consolidates ingested NHANES tables into a unified dataset with:
- Blood pressure averaging
- Target variable creation (cardiovascular, metabolic components, kidney ordinal, liver)
- Standardized column naming
- Externalized biological bounds and clinical thresholds

Refactored: 2025-12-26
- Kidney target: Binary -> 3-class ordinal (Normal/Microalbuminuria/Macroalbuminuria)
- Metabolic: Preserved as 5 distinct binary components
- Bounds: Externalized to ELT_Config.json (BMI <= 100, Pulse >= 30)
"""

import pandas as pd
import numpy as np
import os
import sqlite3
import duckdb
import json

script_dir = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(script_dir, "ELT_Config.json"), "r") as f:
    config = json.load(f)

FOLDER_PATH_DB = os.path.join(script_dir, *config["DBs"]["Paths"]["nhanes_1st"].split("/"))
SQL_JOIN_PATH = os.path.join(script_dir, *config["SQL_Queries"]["join_all_tables"].split("/"))

TARGET_QUERIES = {
    "cardio": {
        "path": os.path.join(script_dir, *config["SQL_Queries"]["cardio_target"].split("/")),
        "merge_cols": ["SEQN", "has_cardiovascular_disease"],
        "table_alias": "HeartQuestions"
    },
    "metabolic": {
        "path": os.path.join(script_dir, *config["SQL_Queries"]["metabolic_target"].split("/")),
        "merge_cols": ["SEQN", "high_waist_circumference", "high_triglycerides_mg_dl", 
                       "low_hdl_mg_dl", "high_blood_pressure", "high_glucose_mg_dl"],
        "table_alias": "table_df"
    },
    "kidney": {
        "path": os.path.join(script_dir, *config["SQL_Queries"]["kidney_target"].split("/")),
        "merge_cols": ["SEQN", "ACR_mg_g"],
        "table_alias": "table_df"
    },
    "liver": {
        "path": os.path.join(script_dir, *config["SQL_Queries"]["liver_target"].split("/")),
        "merge_cols": ["SEQN", "ALT_U_L"],
        "table_alias": "table_df"
    }
}

KIDNEY_ACR_MICRO = config["clinical_thresholds"]["kidney_acr_micro"]
KIDNEY_ACR_MACRO = config["clinical_thresholds"]["kidney_acr_macro"]
LIVER_ALT_MALE = config["clinical_thresholds"]["liver_alt_male"]
LIVER_ALT_FEMALE = config["clinical_thresholds"]["liver_alt_female"]

BIOLOGICAL_BOUNDS = {k: tuple(v) for k, v in config["biological_bounds"].items()}

ARTIFACT_THRESHOLD = config["digital_dust_mapping"]["artifact_threshold"]
DIGITAL_DUST_MAPPING = {
    'logical_zero': config["digital_dust_mapping"]["logical_zero"],
    'biological_null': config["digital_dust_mapping"]["biological_null"]
}

# DATABASE CONNECTIONS
conn = sqlite3.connect(FOLDER_PATH_DB)
con = duckdb.connect()

row_log = {}


def log_row_count(stage: str, df: pd.DataFrame) -> None:
    row_log[stage] = len(df)
    print(f"[CONSORT] {stage}: {len(df):,} rows")


def load_base_dataframe() -> pd.DataFrame:

    with open(SQL_JOIN_PATH, 'r') as file:
        query = file.read()
    
    df = pd.read_sql_query(query, conn)
    df = df.loc[:, ~df.columns.duplicated()]
    log_row_count("01_After_Adult_Filter", df)
    print(f"[OK] Loaded base dataset: {len(df)} rows, {len(df.columns)} columns")
    return df


def add_bp_averages(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates final Systolic and Diastolic averages from harmonized readings.
    Handles missing values automatically (mean of available readings).
    BP column names loaded from ELT_Config.json.
    """
    sys_cols = config["Harmonization"]["bp_columns"]["systolic"]
    dia_cols = config["Harmonization"]["bp_columns"]["diastolic"]
    
    df['Final_Harmonized_Systolic'] = df[sys_cols].mean(axis=1)
    df['Final_Harmonized_Diastolic'] = df[dia_cols].mean(axis=1)
    
    print("[OK] Added blood pressure averages")
    return df


def apply_biological_bounds(df: pd.DataFrame) -> pd.DataFrame:
    """
    Zero-Clamp: Replace physiologically impossible values with NaN.
    Bounds loaded from ELT_Config.json for clinical PI adjustability.
    
    NOTE: BMI upper bound relaxed to 100 kg/m² to retain super-obese phenotype.
    NOTE: Pulse lower bound set to 30 bpm to retain athletes/beta-blocker patients.
    TODO: Add beta-blocker medication check for Pulse 30-40 range when RXDRUG data is ingested.
    """
    for col, (low, high) in BIOLOGICAL_BOUNDS.items():
        if col in df.columns:
            invalid_mask = (df[col] < low) | (df[col] > high)
            invalid_count = invalid_mask.sum()
            if invalid_count > 0:
                df.loc[invalid_mask, col] = np.nan
                print(f"[CLAMP] {col}: {invalid_count} values outside [{low}, {high}] -> NULL")
    return df


def apply_digital_dust_cleanup(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean microscopic floating-point artifacts (< 1e-10) based on clinical categorization.
    - Logical Zero: Values that CAN be 0 (income, drinks) -> convert to 0.0
    - Biological Null: Values that CANNOT be 0 in living patients -> convert to NaN
    """
    for col in DIGITAL_DUST_MAPPING['logical_zero']:
        if col in df.columns:
            mask = (df[col].abs() < ARTIFACT_THRESHOLD) & (df[col].abs() > 0) & (df[col].notna())
            count = mask.sum()
            if count > 0:
                df.loc[mask, col] = 0.0
                print(f"[DUST] {col}: {count} artifacts -> 0.0 (Logical Zero)")
    
    for col in DIGITAL_DUST_MAPPING['biological_null']:
        if col in df.columns:
            mask = (df[col].abs() < ARTIFACT_THRESHOLD) & (df[col].abs() > 0) & (df[col].notna())
            count = mask.sum()
            if count > 0:
                df.loc[mask, col] = np.nan
                print(f"[DUST] {col}: {count} artifacts -> NULL (Biological Impossibility)")
    
    return df


def apply_classification_targets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert continuous biomarkers to classification targets.
    
    THREE-STATE LOGIC GATE (preserves Mask Integrity for Multi-Task Loss):
      1.0/2.0 = Elevated Risk (value >= threshold)
      0.0     = Normal (value < threshold AND data is present)
      NaN     = Insufficient Data (preserved for Masked Loss)
    
    KIDNEY (albuminuria_risk): 3-Class Ordinal (KDIGO Guidelines)
      Class 0 (Normal):          ACR < 30 mg/g
      Class 1 (Microalbuminuria): 30 <= ACR <= 300 mg/g
      Class 2 (Macroalbuminuria): ACR > 300 mg/g
      NOTE: Cross-sectional limitation - detects albuminuria, not "chronic" kidney disease.
    
    LIVER (liver_dysfunction): Binary with GENDER-SPECIFIC thresholds
      Male:   ALT > 40 U/L indicates dysfunction
      Female: ALT > 25 U/L indicates dysfunction
    """
    # KIDNEY: 3-Class Ordinal (Normal / Microalbuminuria / Macroalbuminuria)
    # Renamed to 'albuminuria_risk' for cross-sectional honesty
    conditions_kidney = [
        (df['ACR_mg_g'].notna()) & (df['ACR_mg_g'] < KIDNEY_ACR_MICRO),
        (df['ACR_mg_g'].notna()) & (df['ACR_mg_g'] >= KIDNEY_ACR_MICRO) & (df['ACR_mg_g'] <= KIDNEY_ACR_MACRO),
        (df['ACR_mg_g'].notna()) & (df['ACR_mg_g'] > KIDNEY_ACR_MACRO),
    ]
    choices_kidney = [0.0, 1.0, 2.0]
    df['albuminuria_risk'] = np.select(conditions_kidney, choices_kidney, default=np.nan)
    
    # LIVER: Binary with GENDER-SPECIFIC thresholds
    # RIAGENDR: 1 = Male, 2 = Female (NHANES standard)
    conditions_liver = [
        (df['RIAGENDR'] == 1) & (df['ALT_U_L'] > LIVER_ALT_MALE),    # Male threshold
        (df['RIAGENDR'] == 2) & (df['ALT_U_L'] > LIVER_ALT_FEMALE),  # Female threshold
    ]
    choices_liver = [1.0, 1.0]  # Both conditions indicate dysfunction
    
    # Default: 0 if data present, NaN if missing
    has_liver_data = (df['ALT_U_L'].notna()) & (df['RIAGENDR'].notna())
    df['liver_dysfunction'] = np.select(conditions_liver, choices_liver, default=np.nan)
    df.loc[has_liver_data & (df['liver_dysfunction'].isna()), 'liver_dysfunction'] = 0.0
    
    # Report counts for KIDNEY (3-class)
    k0 = (df['albuminuria_risk'] == 0.0).sum()
    k1 = (df['albuminuria_risk'] == 1.0).sum()
    k2 = (df['albuminuria_risk'] == 2.0).sum()
    k_null = df['albuminuria_risk'].isnull().sum()
    print(f"[TARGET] albuminuria_risk: {k0} Normal (0), {k1} Micro (1), {k2} Macro (2), {k_null} NULL")
    
    # Report counts for LIVER (binary, gender-adjusted)
    l_pos = (df['liver_dysfunction'] == 1.0).sum()
    l_neg = (df['liver_dysfunction'] == 0.0).sum()
    l_null = df['liver_dysfunction'].isnull().sum()
    print(f"[TARGET] liver_dysfunction (gender-adjusted): {l_pos} positive, {l_neg} negative, {l_null} NULL")
    
    return df


def apply_sanity_drop(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows where cardiovascular target is NULL (expect minimal)."""
    before = len(df)
    df = df[df['has_cardiovascular_disease'].notna()].copy()
    after = len(df)
    dropped = before - after
    log_row_count("02_After_Sanity_Drop", df)
    print(f"[SANITY] Dropped {dropped} rows with NULL cardiovascular target")
    return df


def validate_mask_integrity(df: pd.DataFrame, target_cols: list) -> None:
    """
    Audit trail: Ensure target NULLs are preserved (not filled with 0).
    
    THREE-STATE LOGIC ensures:
    - Patients with lab data below threshold -> 0.0 (confirmed healthy)
    - Patients with lab data above threshold -> 1.0 or 2.0 (disease)
    - Patients with missing lab data -> NaN (masked in loss function)
    
    This prevents "imputation bias" where missing data is treated as healthy.
    """
    print("\n--- MASK INTEGRITY CHECK ---")
    for col in target_cols:
        if col in df.columns:
            zero_count = (df[col] == 0).sum()
            one_count = (df[col] == 1).sum()
            two_count = (df[col] == 2).sum()  # For ordinal kidney
            null_count = df[col].isnull().sum()
            total = len(df)
            
            if two_count > 0:
                print(f"[MASK] {col}: Class 0={zero_count}, Class 1={one_count}, Class 2={two_count}, NULLs={null_count} ({100*null_count/total:.1f}%)")
            else:
                print(f"[MASK] {col}: {null_count} NULLs ({100*null_count/total:.1f}%), {zero_count} explicit zeros, {one_count} ones")
    print("--- END MASK CHECK ---\n")


def apply_target_query(df: pd.DataFrame, target_name: str, query_path: str, 
                       merge_cols: list[str], table_alias: str) -> pd.DataFrame:
    """
    Generic function to apply a target calculation query and merge results.
    
    Args:
        df: Source DataFrame
        target_name: Name of target for logging
        query_path: Path to SQL query file
        merge_cols: Columns to keep from query result and merge on SEQN
        table_alias: Table name to register in DuckDB
    
    Returns:
        DataFrame with new target columns merged
    """
    with open(query_path, 'r') as file:
        query = file.read()
    
    con.register(table_alias, df)
    result_df = con.execute(query).df()
    
    df = df.merge(result_df[merge_cols], on='SEQN', how='left')
    
    # Log null counts for the new columns (excluding SEQN)
    new_cols = [c for c in merge_cols if c != 'SEQN']
    null_counts = df[new_cols].isnull().sum().to_dict()
    print(f"[OK] Applied {target_name} target | Nulls: {null_counts}")
    
    return df


def select_and_rename_columns(df: pd.DataFrame) -> pd.DataFrame:

    inputs = config["Harmonization"]["inputs"]
    targets = config["Harmonization"]["targets"]
    column_renaming = config["Harmonization"]["column_renaming"]
    
    # Select only needed columns
    total_cols = inputs + targets
    df = df[total_cols]
    
    # Apply renaming
    df = df.rename(columns=column_renaming)
    
    log_row_count("03_Final_Output", df)
    print(f"[OK] Selected {len(inputs)} input features + {len(targets)} targets")
    print(f"[OK] Renamed columns to standardized names")
    return df


def save_to_database(df: pd.DataFrame, table_name: str = "raw_dataset") -> None:
    
    df.to_sql(table_name, conn, if_exists='replace', index=False)
    print(f"[OK] Saved to table '{table_name}': {len(df)} rows")


def print_consort_summary() -> None:
    """Print CONSORT-style row count summary for publication."""
    print("\n" + "=" * 60)
    print("CONSORT DIAGRAM DATA")
    print("=" * 60)
    for stage, count in row_log.items():
        print(f"  {stage}: {count:,}")
    print("=" * 60 + "\n")


# ===== MAIN EXECUTION =====

print("=" * 60)
print("HARMONIZATION PIPELINE - Starting")
print("=" * 60)

# Step 1: Load base data (Adults only, with valid gender)
df = load_base_dataframe()

# Step 2: Add blood pressure averages
df = add_bp_averages(df)

# Step 3: Apply biological bounds (Zero-Clamp) - loads from config
df = apply_biological_bounds(df)

# Step 4: Apply all target queries (consolidated loop)
for target_name, target_config in TARGET_QUERIES.items():
    df = apply_target_query(
        df=df,
        target_name=target_name,
        query_path=target_config["path"],
        merge_cols=target_config["merge_cols"],
        table_alias=target_config["table_alias"]
    )

# Step 5: Apply Classification Targets (3-class kidney, binary liver)
df = apply_classification_targets(df)

# Step 6: Apply Sanity Drop (remove NULL cardiovascular rows)
df = apply_sanity_drop(df)

# Step 7: Select and rename columns
df = select_and_rename_columns(df)

# Step 8: Apply Digital Dust cleanup (after renaming, before mask check)
df = apply_digital_dust_cleanup(df)

# Step 9: Validate mask integrity (audit trail)
validate_mask_integrity(df, config["Harmonization"]["targets"])

# Step 10: Save to database
save_to_database(df)

# Step 11: Print CONSORT summary
print_consort_summary()

# ===== CLEANUP =====
conn.close()
con.close()

print("=" * 60)
print("=== Harmonization Complete. All connections closed. ===")
print("=" * 60)