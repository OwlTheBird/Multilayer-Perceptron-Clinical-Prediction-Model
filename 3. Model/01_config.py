# config.py
"""
Configuration file for the Multi-Task Learning model.
Loads settings from config.json for easier management.
"""

import os
import json

# --- LOAD CONFIG FROM JSON ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(SCRIPT_DIR, 'config.json')

with open(CONFIG_PATH, 'r') as f:
    _config = json.load(f)

# --- DATABASE PATH ---
DB_PATH = os.path.join(SCRIPT_DIR, _config['database']['path'])

# --- INPUT FEATURES ---
CONT_COLS = _config['features']['continuous_columns']

# --- TARGETS ---
TARGET_MAPPING = _config['targets']

# --- CLASS WEIGHTS ---
CLASS_WEIGHTS = _config['class_weights']

# --- TASK WEIGHTS ---
TASK_WEIGHTS = _config['task_weights']

# --- HYPERPARAMETERS ---
BATCH_SIZE = _config['hyperparameters']['batch_size']
LEARNING_RATE = _config['hyperparameters']['learning_rate']
EPOCHS = _config['hyperparameters']['epochs']
HIDDEN_DIM = _config['hyperparameters']['hidden_dim']
DROPOUT_RATE = _config['hyperparameters']['dropout_rate']
WEIGHT_DECAY = _config['hyperparameters']['weight_decay']
FOCAL_GAMMA = _config['hyperparameters']['focal_gamma']

# --- KIDNEY ORDINAL WEIGHTS ---
KIDNEY_ORDINAL_WEIGHTS = _config['kidney_ordinal_weights']

# --- OPTIMAL THRESHOLDS ---
OPTIMAL_THRESHOLDS = _config['optimal_thresholds']
