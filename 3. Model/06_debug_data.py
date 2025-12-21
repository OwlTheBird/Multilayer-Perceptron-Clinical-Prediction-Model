# debug_data.py
"""
Diagnostic script to find the source of NaN values in training data.
"""

import sqlite3
import pandas as pd
import numpy as np
import os

# Get paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(SCRIPT_DIR, '..', 'databases', 'ML_data.db')

# Connect and load
conn = sqlite3.connect(DB_PATH)
data = pd.read_sql("SELECT * FROM training_set", conn)
conn.close()

print("=" * 60)
print("DATA DIAGNOSTIC REPORT")
print("=" * 60)

print(f"\nTotal rows: {len(data)}")
print(f"Total columns: {len(data.columns)}")

# Check for NaN in each column
print("\n--- NaN COUNT PER COLUMN ---")
nan_counts = data.isnull().sum()
nan_cols = nan_counts[nan_counts > 0]
if len(nan_cols) > 0:
    print("Columns with NaN values:")
    for col, count in nan_cols.items():
        pct = count / len(data) * 100
        print(f"  {col}: {count} NaN values ({pct:.2f}%)")
else:
    print("No NaN values found!")

# Check for Inf values
print("\n--- INFINITY CHECK ---")
numeric_cols = data.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
    inf_count = np.isinf(data[col]).sum()
    if inf_count > 0:
        print(f"  {col}: {inf_count} Inf values")

# Check column names exist
print("\n--- COLUMN NAME CHECK ---")
expected_inputs = [
    'age', 'income_ratio', 'body_mass_index', 'height_cm', 'heart_rate_bpm',
    'white_blood_cells_count', 'platelets_count', 'hemoglobin_g_dl', 'mean_corpuscular_volume_fL',
    'creatinine_mg_dl', 'liver_ast_U_L', 'bilirubin_mg_dl', 'liver_ggt_U_L',
    'uric_acid_mg_dl', 'sodium_mmol_L', 'potassium_mmol_L', 'cholesterol_mg_dl',
    'alcohol_drinks_per_week', 'smoking_status',
    'gender_1.0', 'gender_2.0',
    'ethnicity_1.0', 'ethnicity_2.0', 'ethnicity_3.0',
    'ethnicity_4.0', 'ethnicity_6.0', 'ethnicity_7.0'
]

expected_targets = [
    'has_cardiovascular_disease',
    'high_waist_circumference', 'high_triglycerides_mg_dl', 'low_hdl_mg_dl', 'high_blood_pressure', 'high_glucose_mg_dl',
    'kidney_acr_mg_g', 'liver_alt_U_L'
]

print(f"Expected input columns: {len(expected_inputs)}")
print(f"Expected target columns: {len(expected_targets)}")

missing_inputs = [col for col in expected_inputs if col not in data.columns]
missing_targets = [col for col in expected_targets if col not in data.columns]

if missing_inputs:
    print(f"\n⚠️ MISSING INPUT COLUMNS: {missing_inputs}")
if missing_targets:
    print(f"\n⚠️ MISSING TARGET COLUMNS: {missing_targets}")

print("\n--- ACTUAL COLUMNS IN DATABASE ---")
for col in data.columns:
    print(f"  '{col}'")

# Check target value ranges
print("\n--- TARGET VALUE RANGES ---")
for col in expected_targets:
    if col in data.columns:
        series = data[col]
        print(f"  {col}:")
        print(f"    Min: {series.min()}, Max: {series.max()}, NaN: {series.isnull().sum()}")
