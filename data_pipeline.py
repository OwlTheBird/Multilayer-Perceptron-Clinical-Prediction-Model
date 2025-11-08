"""
NHANES Data Collection Pipeline
Systematically loads and merges data from multiple sources for kidney disease analysis
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class NHANESDataPipeline:
    """
    Pipeline for collecting and merging NHANES data across multiple cycles and data categories
    """
    
    # --- Data Configuration ---
    def __init__(self, base_path: str):
        self.base_path = Path(base_path)
        self.data_config = self._initialize_data_config()
        
    def _initialize_data_config(self) -> Dict:
        """
        Define all data sources organized by category and cycle.
        FIXED: Corrected 2013-2014 demographics file placement.
        """
        config = {
            '2013-2014': {
                'cycle_suffix': '_H',
                'demographics': [
                    'DEMO_H.xpt',        # ✅ CORRECT BASE: Demographics (Located in LaboratoryData folder in original config, but should be the base file)
                    'BIOPRO_H.xpt'       # Biochemistry Profile
                ],
                'examination': [
                    'BMX_H.xpt',      # Body Measures
                    'BPX_H.xpt',      # Blood Pressure
                    'DXXAAC_H.xpt',   # Dual-Energy X-ray Absorptiometry
                    'MGX_H.xpt'       # Muscle Strength - Grip Test
                ],
                'laboratory': [
                    'ALB_CR_H (1).xpt',  # Albumin & Creatinine - Urine
                    'CBC_H.xpt',         # Complete Blood Count
                    'GLU_H.xpt'          # Glucose
                ],
                'questionnaire': [
                    'BPQ_H.xpt',      # Blood Pressure & Cholesterol
                    'CDQ_H.xpt',      # Cardiovascular Health
                    'DIQ_H.xpt',      # Diabetes
                    'HSQ_H.xpt',      # Current Health Status
                    'PAQ_H.xpt',      # Physical Activity
                    'RXQ_RX_H.xpt',   # Prescription Medications
                    'SMQ_H.xpt'       # Smoking - Cigarette Use
                ]
            },
            '2015-2016': {
                'cycle_suffix': '_I',
                'demographics': [
                    'DEMO_I.xpt',
                    'BIOPRO_I.xpt'
                ],
                'examination': [
                    'BMX_I.xpt',      
                    'BPX_I.xpt',      
                    'DXXAG_I.xpt'     
                ],
                'laboratory': [
                    'ALB_CR_I.xpt',   
                    'CBC_I.xpt',      
                    'GLU_I.xpt'       
                ],
                'questionnaire': [
                    'BPQ_I.xpt',      
                    'CDQ_I.xpt',      
                    'DIQ_I.xpt',      
                    'HSQ_I.xpt',      
                    'PAQ_I.xpt',      
                    'RXQ_RX_I.xpt',   
                    'SMQ_I.xpt'       
                ]
            },
            '2017-2018': {
                'cycle_suffix': '_J',
                'demographics': [
                    'DEMO_J.xpt', 
                    'BIOPRO_J.xpt'
                ],
                'examination': [
                    'BMX_J.xpt',      
                    'BPX_J.xpt',      
                    'DXXAG_J.xpt'     
                ],
                'laboratory': [
                    'ALB_CR_J.xpt',   
                    'CBC_J.xpt',      
                    'GLU_J.xpt'       
                ],
                'questionnaire': [
                    'BPQ_J.xpt',      
                    'CDQ_J.xpt',      
                    'DIQ_J.xpt',      
                    'HSQ_J.xpt',      
                    'PAQ_J.xpt',      
                    'RXQ_RX_J.xpt',   
                    'SMQ_J.xpt'       
                ]
            },
            '2017-2020': {
                'cycle_suffix': 'P_',
                'demographics': [
                    'P_DEMO.xpt',
                    'P_BIOPRO.xpt' 
                ],
                'examination': [
                    'P_BMX.xpt',      
                    'P_BPXO.xpt',     
                    'P_DXXFEM.xpt'    
                ],
                'laboratory': [
                    'P_ALB_CR.xpt',   
                    'P_CBC.xpt',      
                    'P_GLU.xpt'       
                ],
                'questionnaire': [
                    'P_BPQ.xpt',      
                    'P_CDQ.xpt',      
                    'P_DIQ.xpt',      
                    'P_HSQ.xpt',      
                    'P_PAQ.xpt',      
                    'P_RXQ_RX.xpt',   
                    'P_SMQ.xpt'       
                ]
            },
            '2021-2022': {
                'cycle_suffix': '_L',
                'demographics': [
                    'DEMO_L.XPT',     
                    'BIOPRO_L.XPT'    # Biochemistry Profile moved here from lab
                ],
                'examination': [
                    'BMX_L.XPT',      
                    'BPXO_L.xpt'      # Oscillometric BP (uses BPXOSY1 instead of BPXSY1)
                ],
                'laboratory': [
                    'ALB_CR_L.XPT',   
                    'CBC_L.XPT',      
                    'GLU_L.XPT'       
                ],
                'questionnaire': [
                    'BPQ_L.XPT',      
                    'DIQ_L.XPT',      
                    'HSQ_L.XPT',      
                    'PAQ_L.XPT',      
                    'RXQ_RX_L.XPT',   
                    'SMQ_L.XPT'       
                ]
            },
        }
        return config
    
    # --- Loading & Merging Methods (Unchanged, except for fixed indentation in multi-cycle method) ---
    def load_single_file(self, file_path: Path, category: str) -> Optional[pd.DataFrame]:
        # ... (Method body remains the same) ...
        try:
            if not file_path.exists():
                logger.warning(f"File not found: {file_path}")
                return None
            
            df = pd.read_sas(file_path)
            logger.info(f"Loaded {category}: {file_path.name} - {len(df)} rows, {len(df.columns)} columns")
            return df
            
        except Exception as e:
            logger.error(f"Error loading {file_path}: {str(e)}")
            return None
    
    def load_cycle_data(self, cycle: str) -> Dict[str, List[pd.DataFrame]]:
        # ... (Method body remains the same) ...
        if cycle not in self.data_config:
            raise ValueError(f"Cycle {cycle} not configured")
        
        logger.info(f"Loading data for cycle: {cycle}")
        cycle_config = self.data_config[cycle]
        loaded_data = {}
        
        # Load Demographics Data
        logger.info("Loading Demographics Data...")
        loaded_data['demographics'] = []
        for filename in cycle_config['demographics']:
            file_path = self.base_path / 'Demographics Data' / filename
            df = self.load_single_file(file_path, 'Demographics')
            if df is not None:
                loaded_data['demographics'].append(df)
        
        # Load Examination Data
        logger.info("Loading Examination Data...")
        loaded_data['examination'] = []
        for filename in cycle_config['examination']:
            file_path = self.base_path / 'Examination Data' / filename
            df = self.load_single_file(file_path, 'Examination')
            if df is not None:
                loaded_data['examination'].append(df)
        
        # Load Laboratory Data
        logger.info("Loading Laboratory Data...")
        loaded_data['laboratory'] = []
        for filename in cycle_config['laboratory']:
            file_path = self.base_path / 'LaboratoryData' / filename
            df = self.load_single_file(file_path, 'Laboratory')
            if df is not None:
                loaded_data['laboratory'].append(df)
        
        # Load Questionnaire Data
        logger.info("Loading Questionnaire Data...")
        loaded_data['questionnaire'] = []
        for filename in cycle_config['questionnaire']:
            file_path = self.base_path / 'Questionnaire Data' / filename
            df = self.load_single_file(file_path, 'Questionnaire')
            if df is not None:
                loaded_data['questionnaire'].append(df)
        
        return loaded_data
    
    def merge_dataframes(self, 
                         base_df: pd.DataFrame, 
                         dataframes: List[pd.DataFrame], 
                         merge_key: str = 'SEQN',
                         how: str = 'inner') -> pd.DataFrame:
        # ... (Method body remains the same) ...
        result = base_df.copy()
        initial_rows = len(result)
        
        for i, df in enumerate(dataframes):
            if df is None or df.empty:
                continue
            
            before_merge = len(result)
            result = pd.merge(result, df, on=merge_key, how=how, suffixes=('', f'_dup{i}'))
            after_merge = len(result)
            
            logger.info(f"Merge {i+1}: {before_merge} -> {after_merge} rows, {len(result.columns)} columns")
        
        logger.info(f"Final merged data: {initial_rows} -> {len(result)} rows ({len(result.columns)} columns)")
        return result
    
    def create_master_dataset(self, cycle: str, merge_how: str = 'inner') -> pd.DataFrame:
        # ... (Method body remains the same, but benefits from the config fix) ...
        logger.info(f"Creating master dataset for cycle: {cycle}")
        
        # Load all data
        loaded_data = self.load_cycle_data(cycle)
        
        # Start with demographics as base
        if not loaded_data['demographics']:
            raise ValueError("No demographics data loaded - cannot create master dataset")
        
        base_df = loaded_data['demographics'][0].copy()
        base_df['Cycle'] = cycle
        logger.info(f"Base dataset: {len(base_df)} rows, {len(base_df.columns)} columns")
        
        # Merge all demographics files
        if len(loaded_data['demographics']) > 1:
            logger.info("Merging additional demographics files...")
            base_df = self.merge_dataframes(base_df, loaded_data['demographics'][1:], how=merge_how)
        
        # Merge laboratory data
        if loaded_data['laboratory']:
            logger.info("Merging laboratory data...")
            base_df = self.merge_dataframes(base_df, loaded_data['laboratory'], how=merge_how)
        
        # Merge examination data
        if loaded_data['examination']:
            logger.info("Merging examination data...")
            base_df = self.merge_dataframes(base_df, loaded_data['examination'], how=merge_how)
        
        # Merge questionnaire data
        if loaded_data['questionnaire']:
            logger.info("Merging questionnaire data...")
            base_df = self.merge_dataframes(base_df, loaded_data['questionnaire'], how=merge_how)
        
        # Standardize column names for cycles using oscillometric BP (BPXOSY1 -> BPXSY1)
        if cycle in ['2017-2020', '2021-2022']:
            if 'BPXOSY1' in base_df.columns and 'BPXSY1' not in base_df.columns:
                base_df['BPXSY1'] = base_df['BPXOSY1']
                logger.info(f"Renamed BPXOSY1 to BPXSY1 for {cycle} cycle")
        
        logger.info(f"Master dataset created: {len(base_df)} rows, {len(base_df.columns)} columns")
        return base_df
    
    # --- New Imputation Method ---
    def impute_median(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """
        Imputes missing values using the median of each specified column.
        """
        df_imputed = df.copy()
        
        for col in columns:
            if col in df_imputed.columns:
                median_val = df_imputed[col].median()
                df_imputed[col].fillna(median_val, inplace=True)
                logger.info(f"Imputed {col} with median value: {median_val:.2f} (Filled {df[col].isnull().sum() - df_imputed[col].isnull().sum()} NAs)")
            else:
                logger.warning(f"Imputation skipped: Column '{col}' not found in DataFrame.")
                
        return df_imputed

    # --- Core Processing Method ---
    def process_for_kidney_head(self, df: pd.DataFrame, imputation_strategy: str = 'median') -> pd.DataFrame:
        """
        Selects kidney-specific features, calculates the target, and handles missing data.
        """
        
        # Define the 9 essential columns (8 predictors + 1 raw target)
        base_columns = ['SEQN', 'Cycle', 'RIAGENDR', 'RIDAGEYR', 'BMXBMI', 'BPXSY1', 'LBXGLU', 'LBXSCH']
        
        # 1. Check for available UACR-related columns
        uracr_columns = {
            'URDACT': 'Direct UACR',
            'URXUMA': 'Urine albumin (mg/L)',
            'URXUCR': 'Urine creatinine (mg/dL)'
        }
        
        available_columns = {col: col in df.columns for col in uracr_columns}
        
        # 2. Feature Selection
        df_clean = df[base_columns].copy()
        
        # 3. Target Calculation - Try different methods to get Log_UACR
        if available_columns['URDACT']:
            # Method 1: Use URDACT directly if available
            df_clean['Log_UACR'] = np.log(df['URDACT'] + 1)  # Add 1 to avoid log(0)
            logger.info("Calculated Log_UACR from URDACT column")
            
        elif all(available_columns[col] for col in ['URXUMA', 'URXUCR']):
            # Method 2: Calculate UACR from URXUMA and URXUCR if available
            df_clean['UACR'] = (df['URXUMA'] / df['URXUCR']) * 100  # Convert to mg/g
            df_clean['Log_UACR'] = np.log(df_clean['UACR'] + 1)
            logger.info("Calculated Log_UACR from URXUMA/URXUCR columns")
            
            # Add the calculated UACR to the columns to keep for imputation
            base_columns.append('UACR')
        else:
            # Method 3: If no UACR data is available, log a warning and set to NaN
            logger.warning("No UACR data available. Setting Log_UACR to NaN.")
            df_clean['Log_UACR'] = np.nan
            
        # 4. Drop any temporary columns we don't need anymore
        df_clean = df_clean.drop(columns=[col for col in ['UACR'] if col in df_clean.columns])
        
        continuous_cols = ['BMXBMI', 'BPXSY1', 'LBXGLU', 'LBXSCH', 'Log_UACR']
        
        # 3. Handling Missing Data (Imputation vs. Deletion)
        if imputation_strategy == 'median':
            logger.info("Applying Median Imputation...")
            df_clean = self.impute_median(df_clean, continuous_cols)
            
            # Drop any remaining NAs (e.g., from categorical features or if target calculation failed)
            df_final = df_clean.dropna()
            
        elif imputation_strategy == 'delete':
            logger.info("Applying Listwise Deletion (Complete Case Analysis)...")
            df_final = df_clean.dropna()
        else:
            raise ValueError("Invalid imputation_strategy. Use 'median' or 'delete'.")
        
        logger.info(f"Final processed dataset rows: {len(df_final):,}. Columns: {len(df_final.columns)}")
        return df_final


    # --- Multi-Cycle Method (Fixed and Enhanced) ---
    def create_multi_cycle_dataset(self, cycles: List[str], merge_how: str = 'inner', 
                                   imputation_strategy: str = 'median') -> pd.DataFrame:
        
        """
        Create a combined dataset across multiple cycles and applies final cleaning.
        """
        logger.info(f"Creating multi-cycle dataset for: {', '.join(cycles)}")
    
        all_cycles = []
        for cycle in cycles:
            try:
                # ... (Cycle processing output - kept for user feedback) ...
                cycle_df = self.create_master_dataset(cycle, merge_how=merge_how)
                
                if cycle_df is not None and not cycle_df.empty:
                    all_cycles.append(cycle_df)
                
            except Exception as e:
                logger.error(f"Failed to process cycle {cycle}: {str(e)}", exc_info=True)
        
        if not all_cycles:
            raise ValueError("❌ No cycles were successfully processed!")
        
        # 1. Combine all cycles
        combined_df = pd.concat(all_cycles, axis=0, ignore_index=True)
        logger.info(f"Raw combined dataset: {len(combined_df):,} rows, {len(combined_df.columns)} columns")

        # 2. Apply Feature Selection, Target Calculation, and Imputation/Deletion
        final_df = self.process_for_kidney_head(combined_df, imputation_strategy=imputation_strategy)
        
        logger.info("\n" + "="*50)
        logger.info("FINAL PROCESSED DATASET SUMMARY")
        logger.info("="*50)
        logger.info(f"Final Rows: {len(final_df):,}")
        logger.info(f"Final Columns: {len(final_df.columns)}")
        
        return final_df

# Convenience function for quick usage (Unchanged)
def quick_load_cycle(base_path: str, cycle: str = '2013-2014', merge_how: str = 'inner') -> pd.DataFrame:
    pipeline = NHANESDataPipeline(base_path)
    return pipeline.create_master_dataset(cycle, merge_how=merge_how)