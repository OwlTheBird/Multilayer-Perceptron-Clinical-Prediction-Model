# dataset.py
"""
PyTorch Dataset for loading NHANES multi-task data from ML_data.db.
Handles the data loading and tensor conversion.
Includes NaN handling for target columns.
"""

import sqlite3
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset


class NHANESMultiTaskDataset(Dataset):
    """
    Custom PyTorch Dataset for the NHANES Multi-Task Learning project.
    Loads data from SQLite database and converts to PyTorch tensors.
    Handles NaN values in target columns with masks for proper loss calculation.
    """
    
    def __init__(self, db_path, table_name, cont_cols, target_mapping):
        """
        Args:
            db_path: Path to the SQLite database
            table_name: Name of the table to load ('training_set' or 'testing_set')
            cont_cols: List of continuous input column names
            target_mapping: Dictionary mapping task names to target column(s)
        """
        # Connect and load data
        conn = sqlite3.connect(db_path)
        query = f"SELECT * FROM {table_name}"
        self.data = pd.read_sql(query, conn)
        conn.close()
        
        # Store column info for external access
        self.cont_cols = cont_cols
        self.target_mapping = target_mapping
        
        # Convert inputs to tensor (all continuous since data is pre-processed)
        self.x_cont = torch.tensor(
            self.data[cont_cols].values, 
            dtype=torch.float32
        )
        
        # === HANDLE NaN VALUES IN TARGETS ===
        # Create masks (1 = valid, 0 = was NaN) and fill NaN with defaults
        
        # Task A: Cardiovascular (Binary Classification)
        cardio_data = self.data[target_mapping['cardio']].copy()
        self.mask_cardio = torch.tensor(~cardio_data.isnull().values, dtype=torch.float32).unsqueeze(1)
        cardio_data = cardio_data.fillna(0)  # Fill NaN with 0 (will be masked anyway)
        self.y_cardio = torch.tensor(cardio_data.values, dtype=torch.float32).unsqueeze(1)
        
        # Task B: Metabolic Syndrome (Multi-Label Classification, 5 labels)
        metabolic_data = self.data[target_mapping['metabolic']].copy()
        self.mask_metabolic = torch.tensor(~metabolic_data.isnull().values, dtype=torch.float32)
        metabolic_data = metabolic_data.fillna(0)  # Fill NaN with 0
        self.y_metabolic = torch.tensor(metabolic_data.values, dtype=torch.float32)
        
        # Task C: Kidney Function - Ordinal Binary Decomposition
        # Converts 3-class ordinal to 2-node binary encoding:
        #   Normal (0) -> [0, 0]  (ACR < 30)
        #   Micro (1)  -> [1, 0]  (30 <= ACR < 300)
        #   Macro (2)  -> [1, 1]  (ACR >= 300)
        # Node A: Is ACR >= 30? | Node B: Is ACR >= 300?
        kidney_raw = self.data[target_mapping['kidney']].copy()
        self.mask_kidney = torch.tensor(~kidney_raw.isnull().values, dtype=torch.float32)
        
        # Convert to ordinal encoding
        kidney_ordinal = np.zeros((len(kidney_raw), 2), dtype=np.float32)
        kidney_vals = kidney_raw.fillna(-1).values
        # Node A: Class 1 or 2 -> 1
        kidney_ordinal[:, 0] = (kidney_vals >= 1).astype(np.float32)
        # Node B: Class 2 -> 1
        kidney_ordinal[:, 1] = (kidney_vals >= 2).astype(np.float32)
        # Set masked values to 0 (they'll be masked in loss anyway)
        kidney_ordinal[kidney_vals < 0] = 0
        
        self.y_kidney = torch.tensor(kidney_ordinal, dtype=torch.float32)
        self.mask_kidney = self.mask_kidney.unsqueeze(1).expand(-1, 2)  # Expand mask to [N, 2]
        
        # Task D: Liver Function (Binary Classification)
        # Gender-adjusted threshold already applied in ETL
        liver_data = self.data[target_mapping['liver']].copy()
        self.mask_liver = torch.tensor(~liver_data.isnull().values, dtype=torch.float32).unsqueeze(1)
        liver_data = liver_data.fillna(0)  # Fill NaN with 0 (masked anyway)
        self.y_liver = torch.tensor(liver_data.values, dtype=torch.float32).unsqueeze(1)
        
        # Print stats
        print(f"Loaded {len(self.data)} samples from '{table_name}'")
        print(f"Input features: {len(cont_cols)} continuous columns")
        print(f"Valid targets - Cardio: {self.mask_cardio.sum().int().item()}, "
              f"Metabolic: {self.mask_metabolic.sum(dim=0).int().tolist()}, "
              f"Kidney: {self.mask_kidney.sum().int().item()}, "
              f"Liver: {self.mask_liver.sum().int().item()}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Returns a tuple of (inputs, targets, masks) for a given index.
        Masks indicate which targets are valid (1) vs were NaN (0).
        """
        return (
            self.x_cont[idx],           # Continuous inputs
            self.y_cardio[idx],         # Cardio target
            self.y_metabolic[idx],      # Metabolic targets (5 labels)
            self.y_kidney[idx],         # Kidney target
            self.y_liver[idx],          # Liver target
            self.mask_cardio[idx],      # Cardio mask
            self.mask_metabolic[idx],   # Metabolic mask (5 labels)
            self.mask_kidney[idx],      # Kidney mask
            self.mask_liver[idx]        # Liver mask
        )

