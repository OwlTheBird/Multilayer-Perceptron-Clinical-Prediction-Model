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
# Updated to match ETL output: kidney is 3-class ordinal, liver is binary

TARGET_MAPPING = {
    # Head A: Binary Classification (CVD detection)
    'cardio': 'has_cardiovascular_disease',
    
    # Head B: Multi-Label Classification (5 metabolic syndrome components)
    'metabolic': [
        'high_waist_circumference',
        'high_triglycerides_mg_dl',
        'low_hdl_mg_dl',
        'high_blood_pressure',
        'high_glucose_mg_dl'
    ],
    
    # Head C: 3-Class Ordinal Classification (albuminuria staging per KDIGO)
    # Classes: 0=Normal (<30), 1=Microalbuminuria (30-300), 2=Macroalbuminuria (>300)
    'kidney': 'albuminuria_risk',
    
    # Head D: Binary Classification (liver dysfunction)
    # Gender-adjusted threshold: Male ALT>40, Female ALT>25
    'liver': 'liver_dysfunction'
}

# --- CLASS WEIGHTS (from EDA Summary.md) ---
# Addresses class imbalance to prevent majority-class bias

CLASS_WEIGHTS = {
    # Binary tasks: pos_weight = n_negative / n_positive
    'cvd_pos_weight': 7.36,            # 88:12 imbalance (30,018 vs 4,079)
    'liver_pos_weight': 5.62,          # 85:15 imbalance (24,373 vs 4,335)
    'triglycerides_pos_weight': 3.76,  # 79:21 imbalance
    'hdl_pos_weight': 2.40,            # 71:29 imbalance
    'bp_pos_weight': 1.78,             # 64:36 imbalance
    'waist_pos_weight': None,          # 60:40 balanced, no weight needed
    'glucose_pos_weight': None,        # 59:41 balanced, no weight needed
    
    # Kidney 3-class: inverse frequency weights normalized
    # Class counts: [26217, 3265, 689] -> weights favor rare classes
    'kidney_class_weights': [0.073, 0.59, 2.8]  # [Normal, Micro, Macro]
}

# --- TASK WEIGHTS ---
# Reduce contribution of high-missing targets (>50% NaN)

TASK_WEIGHTS = {
    'cvd': 1.0,              # 0% missing
    'metabolic_waist': 1.0,  # 14.5% missing
    'metabolic_trig': 0.5,   # 58.3% missing (fasting-only subset)
    'metabolic_hdl': 1.0,    # 15.4% missing
    'metabolic_bp': 1.0,     # 15.8% missing
    'metabolic_glucose': 0.5,# 59.5% missing (fasting-only subset)
    'kidney': 1.0,           # 11.5% missing
    'liver': 1.0             # 15.8% missing
}

# --- HYPERPARAMETERS ---
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 20
HIDDEN_DIM = 256  # Widened from 128 to reduce gradient conflict
WEIGHT_DECAY = 1e-4  # L2 regularization for multicollinearity

# --- OPTIMAL THRESHOLDS (from threshold optimization) ---
# Calibrated for clinical recall targets
OPTIMAL_THRESHOLDS = {
    'cvd': 0.33,    # For ~80% recall (default 0.5 gives only 21%)
    'liver': 0.44   # For ~70% recall (default 0.5 gives only 60%)
}
