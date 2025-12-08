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
    'RIDAGEYR', 'INDFMPIR',
    
    # Body Measures (scaled)
    'BMXBMI', 'BMXHT',
    
    # Vitals (scaled)
    'Pulse',
    
    # Blood Count (scaled)
    'LBXWBCSI', 'LBXPLTSI', 'LBXHGB', 'LBXMCVSI',
    
    # Biochemistry (scaled)
    'LBXSCR', 'LBXSASSI', 'LBXSTB', 'LBXSGTSI',
    'LBXSUA', 'LBXSNASI', 'LBXSKSI', 'LBXTC',
    
    # Lifestyle (scaled)
    'Alcohol_Drinks_Per_Week', 'SMQ040',
    
    # One-Hot Encoded Gender
    'RIAGENDR_1.0', 'RIAGENDR_2.0',
    
    # One-Hot Encoded Race/Ethnicity
    'RIDRETH3_1.0', 'RIDRETH3_2.0', 'RIDRETH3_3.0',
    'RIDRETH3_4.0', 'RIDRETH3_6.0', 'RIDRETH3_7.0'
]

# --- TARGETS ---
# Mapping specific columns to the 4 Tasks

TARGET_MAPPING = {
    # Head A: Binary Classification
    'cardio': 'Cardiovascular_target',
    
    # Head B: Multi-Label Classification (5 labels for metabolic syndrome)
    'metabolic': [
        'Waist_Label',
        'Triglycerides_Label',
        'HDL_Label',
        'BP_Label',
        'Glucose_Label'
    ],
    
    # Head C: Regression (Log Scale)
    'kidney': 'ACR_Log',
    
    # Head D: Regression (Log Scale)
    'liver': 'ALT_Log'
}

# --- HYPERPARAMETERS ---
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 20
HIDDEN_DIM = 128
