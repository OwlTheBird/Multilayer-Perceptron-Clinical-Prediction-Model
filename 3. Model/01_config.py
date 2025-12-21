# config.py
"""
Configuration file for the Multi-Task Learning model.
This is the "control panel" for all settings.
"""

import os

# --- DATABASE PATH ---
# Get the path relative to this config file
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(SCRIPT_DIR, '..', 'databases', 'ML_data.db')

# --- INPUT FEATURES ---
# All inputs are continuous (data is already one-hot encoded from Transformation.py)

CONT_COLS = [
    # Demographics (scaled)
    'age', 'income_ratio',
    
    # Body Measures (scaled)
    'body_mass_index', 'height_cm',
    
    # Vitals (scaled)
    'heart_rate_bpm',
    
    # Blood Count (scaled)
    'white_blood_cells_count', 'platelets_count', 'hemoglobin_g_dl', 'mean_corpuscular_volume_fL',
    
    # Biochemistry (scaled)
    'creatinine_mg_dl', 'liver_ast_U_L', 'bilirubin_mg_dl', 'liver_ggt_U_L',
    'uric_acid_mg_dl', 'sodium_mmol_L', 'potassium_mmol_L', 'cholesterol_mg_dl',
    
    # Lifestyle (scaled)
    'alcohol_drinks_per_week',
    
    # One-Hot Encoded Gender (from 'gender')
    'gender_1.0', 'gender_2.0',
    
    # One-Hot Encoded Race/Ethnicity (from 'ethnicity')
    'ethnicity_1.0', 'ethnicity_2.0', 'ethnicity_3.0',
    'ethnicity_4.0', 'ethnicity_6.0', 'ethnicity_7.0',
    
    # One-Hot Encoded Smoking Status (from 'smoking_status')
    'smoking_status_1.0', 'smoking_status_2.0', 'smoking_status_3.0', 'smoking_status_nan'
]

# --- TARGETS ---
# Mapping specific columns to the 4 Tasks

TARGET_MAPPING = {
    # Head A: Binary Classification
    'cardio': 'has_cardiovascular_disease',
    
    # Head B: Multi-Label Classification (5 labels for metabolic syndrome)
    'metabolic': [
        'high_waist_circumference',
        'high_triglycerides_mg_dl',
        'low_hdl_mg_dl',
        'high_blood_pressure',
        'high_glucose_mg_dl'
    ],
    
    # Head C: Regression (Log Scale)
    'kidney': 'kidney_acr_mg_g',
    
    # Head D: Regression (Log Scale)
    'liver': 'liver_alt_U_L'
}

# --- HYPERPARAMETERS ---
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 20
HIDDEN_DIM = 128
