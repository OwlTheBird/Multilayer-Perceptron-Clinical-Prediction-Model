import os
import sqlite3
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import KNNImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# --- Configuration ---
DB_NAME = "ML_data.db"
SOURCE_DB_NAME = "nhanes_1st.db"

# Column Definitions (using standardized names from harmonization)
INPUTS = [
    'age', 'gender', 'ethnicity', 'income_ratio', 'body_mass_index', 'height_cm', 
    'heart_rate_bpm', 'white_blood_cells_count', 'platelets_count', 'hemoglobin_g_dl', 'mean_corpuscular_volume_fL', 
    'creatinine_mg_dl', 'liver_ast_U_L', 'bilirubin_mg_dl', 'liver_ggt_U_L', 'uric_acid_mg_dl', 
    'sodium_mmol_L', 'potassium_mmol_L', 'cholesterol_mg_dl', 'alcohol_drinks_per_week', 'smoking_status'
]

TARGETS = [
    'has_cardiovascular_disease', 
    'high_waist_circumference', 'high_triglycerides_mg_dl', 'low_hdl_mg_dl', 'high_blood_pressure', 'high_glucose_mg_dl',
    'kidney_acr_mg_g', 'liver_alt_U_L'
]

CATEGORICAL_INPUTS = ['gender', 'ethnicity', 'smoking_status']
CONTINUOUS_INPUTS = [col for col in INPUTS if col not in CATEGORICAL_INPUTS]

# --- Paths ---
script_dir = os.path.dirname(os.path.abspath(__file__))
FOLDER_PATH_DB = os.path.join(script_dir, '..', 'databases', DB_NAME)
FOLDER_PATH_1ST_DB = os.path.join(script_dir, '..', 'databases', SOURCE_DB_NAME)
SQL_RAW_DATASET_FILE_PATH = os.path.join(script_dir, '..', 'databases', 'SQL_queries', 'get_raw_dataset.sql')

def get_raw_dataset(db_path: str, query_path: str) -> pd.DataFrame:
    """Loads the raw dataset from the source database."""
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Source database not found at {db_path}")
    
    conn = sqlite3.connect(db_path)
    try:
        with open(query_path, 'r') as file:
            query = file.read()
        df = pd.read_sql_query(query, conn)
        return df
    finally:
        conn.close()

def preprocess_data(df: pd.DataFrame):
    """
    Executes the transformation pipeline:
    1. Drop rows where ALL targets are NaN.
    2. Split into Train and Test.
    3. Fit Preprocessing (Encoding, Scaling, Imputation) on Train.
    4. Transform both Train and Test.
    5. Save to ML_data.db.
    """
    print(f"Initial shape: {df.shape}")

    # 1. Filter Useless Rows (All Targets NaN)
    # Check if all target columns are null for a row
    target_nulls = df[TARGETS].isnull().all(axis=1)
    df_clean = df[~target_nulls].copy()
    print(f"Rows dropped (All Targets NaN): {target_nulls.sum()}")
    print(f"Shape after filtering: {df_clean.shape}")

    # Separate Inputs (X) and Targets (y)
    X = df_clean[INPUTS]
    y = df_clean[TARGETS]

    # 2. Train/Test Split
    # Stratify is tricky with multi-label/multi-task. 
    # For now, we use random split as per standard practice unless specific stratification is requested.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"Train set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")

    # 3. Define Preprocessing Pipelines
    
    # Categorical Pipeline: One-Hot Encoding
    # handle_unknown='ignore' is safe for production if new categories appear (though unlikely for Gender/Race)
    # sparse_output=False to get dense arrays for easier dataframe reconstruction
    cat_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

    # Continuous Pipeline: Scaling -> Imputation
    # KNN Imputer requires scaled data for correct distance calculation
    num_transformer = Pipeline(steps=[
        ('scaler', StandardScaler()),
        ('imputer', KNNImputer(n_neighbors=5))
    ])

    # Combine into ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_transformer, CONTINUOUS_INPUTS),
            ('cat', cat_transformer, CATEGORICAL_INPUTS)
        ],
        verbose_feature_names_out=False # Keep column names clean
    )

    # 4. Fit on Train, Transform Both
    print("Fitting preprocessor on Training set...")
    X_train_processed = preprocessor.fit_transform(X_train)
    
    print("Transforming Test set...")
    X_test_processed = preprocessor.transform(X_test)

    # Reconstruct DataFrames with column names
    # Get feature names from the transformer
    # Note: OneHotEncoder generates new names like RIAGENDR_1, RIAGENDR_2
    feature_names = preprocessor.get_feature_names_out()
    
    X_train_df = pd.DataFrame(X_train_processed, columns=feature_names, index=X_train.index)
    X_test_df = pd.DataFrame(X_test_processed, columns=feature_names, index=X_test.index)

    # Reattach Targets
    # We use the index to align them correctly
    train_df = pd.concat([X_train_df, y_train], axis=1)
    test_df = pd.concat([X_test_df, y_test], axis=1)

    return train_df, test_df

def save_to_db(train_df: pd.DataFrame, test_df: pd.DataFrame, db_path: str):
    """Saves the processed datasets to the ML database."""
    conn = sqlite3.connect(db_path)
    try:
        print(f"Saving to {db_path}...")
        train_df.to_sql('training_set', conn, if_exists='replace', index=False)
        test_df.to_sql('testing_set', conn, if_exists='replace', index=False)
        print("Successfully saved 'training_set' and 'testing_set'.")
    finally:
        conn.close()

if __name__ == "__main__":
    # 1. Load
    print("Loading raw dataset...")
    df_raw = get_raw_dataset(FOLDER_PATH_1ST_DB, SQL_RAW_DATASET_FILE_PATH)
    
    # 2. Process
    print("Processing data...")
    train_df, test_df = preprocess_data(df_raw)
    
    # 3. Save
    save_to_db(train_df, test_df, FOLDER_PATH_DB)
    print("Transformation complete.")