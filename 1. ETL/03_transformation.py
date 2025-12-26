import os
import json
import sqlite3
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


from skmultilearn.model_selection import iterative_train_test_split
HAS_SKMULTILEARN = True


script_dir = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(script_dir, "ELT_Config.json"), "r") as f:
    config = json.load(f)


SOURCE_DB_PATH = os.path.join(script_dir, *config["DBs"]["Paths"]["nhanes_1st"].split("/"))
OUTPUT_DB_PATH = os.path.join(script_dir, "..", "databases", config["Transformation"]["db_output"])
SQL_RAW_DATASET_PATH = os.path.join(script_dir, *config["SQL_Queries"]["get_raw_dataset"].split("/"))


COLUMN_RENAMING = config["Harmonization"]["column_renaming"]
INPUTS = list(COLUMN_RENAMING.values())


TARGETS = config["Harmonization"]["targets"]


CATEGORICAL_INPUTS = config["Transformation"]["categorical_inputs"]
CONTINUOUS_INPUTS = [col for col in INPUTS if col not in CATEGORICAL_INPUTS]


TEST_SIZE = config["Transformation"]["test_size"]
IMPUTER_MAX_ITER = config["Transformation"]["imputer"]["max_iter"]
IMPUTER_RANDOM_STATE = config["Transformation"]["imputer"]["random_state"]
RANDOM_STATE = config["Transformation"]["random_state"]


row_log = {}

def log_row_count(stage: str, count: int) -> None:
    """Log row count for CONSORT diagram traceability."""
    row_log[stage] = count
    print(f"[CONSORT] {stage}: {count:,} rows")


def get_raw_dataset(db_path: str, query_path: str) -> pd.DataFrame:
    """Loads the raw dataset from the source database."""
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Source database not found at {db_path}")
    
    conn = sqlite3.connect(db_path)
    try:
        with open(query_path, 'r') as file:
            query = file.read()
        df = pd.read_sql_query(query, conn)
        log_row_count("01_Loaded_From_DB", len(df))
        return df
    finally:
        conn.close()


def stratified_split(X: pd.DataFrame, y: pd.DataFrame, test_size: float = 0.2):
    if HAS_SKMULTILEARN:
        print("[OK] Using IterativeStratification for multi-label split")
        
        # 1. Create proxy target matrix (NaN -> -1 for splitting algorithm)
        y_proxy = y.fillna(-1).values
        
        # 2. Generate indices using proxy (returns numpy arrays)
        X_train_arr, y_train_arr, X_test_arr, y_test_arr = iterative_train_test_split(
            X.values, y_proxy, test_size=test_size
        )
        
        # 3. Convert back to DataFrames (indices are lost, so we create new ones)
        X_train = pd.DataFrame(X_train_arr, columns=X.columns)
        X_test = pd.DataFrame(X_test_arr, columns=X.columns)
        
        # For targets, restore NaN values: -1 back to NaN
        y_train = pd.DataFrame(y_train_arr, columns=y.columns)
        y_test = pd.DataFrame(y_test_arr, columns=y.columns)
        
        y_train = y_train.replace(-1, np.nan)
        y_test = y_test.replace(-1, np.nan)
        
        return X_train, X_test, y_train, y_test
    
    else:
        print("[WARN] Using random split (skmultilearn not available)")
        from sklearn.model_selection import train_test_split
        return train_test_split(X, y, test_size=test_size, random_state=RANDOM_STATE)


def preprocess_data(df: pd.DataFrame):
    print(f"Initial shape: {df.shape}")
    log_row_count("02_Initial_Shape", len(df))


    target_nulls = df[TARGETS].isnull().all(axis=1)
    df_clean = df[~target_nulls].copy()
    print(f"Rows dropped (All Targets NaN): {target_nulls.sum()}")
    print(f"Shape after filtering: {df_clean.shape}")
    log_row_count("03_After_Useless_Drop", len(df_clean))

 
    X = df_clean[INPUTS]
    y = df_clean[TARGETS]


    X_train, X_test, y_train, y_test = stratified_split(X, y, test_size=TEST_SIZE)
    
    log_row_count("04_Train_Set", len(X_train))
    log_row_count("05_Test_Set", len(X_test))
    print(f"Train set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    

    if 'albuminuria_risk' in y_test.columns:
        kidney_dist = y_test['albuminuria_risk'].value_counts(dropna=False)
        print(f"[VALIDATE] Albuminuria class distribution in Test Set:")
        print(kidney_dist)
        

        for cls in [0.0, 1.0, 2.0]:
            if cls not in kidney_dist.index:
                print(f"[WARN] Albuminuria Class {cls} is MISSING from test set!")


    

    cat_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)


    num_transformer = Pipeline(steps=[
        ('imputer', IterativeImputer(max_iter=IMPUTER_MAX_ITER, random_state=IMPUTER_RANDOM_STATE)),
        ('scaler', StandardScaler())
    ])


    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_transformer, CONTINUOUS_INPUTS),
            ('cat', cat_transformer, CATEGORICAL_INPUTS)
        ],
        verbose_feature_names_out=False
    )

    # 4. Fit on Train, Transform Both
    # CRITICAL: Fit ONLY on training data
    print("Fitting preprocessor on Training set...")
    X_train_processed = preprocessor.fit_transform(X_train)
    
    print("Transforming Test set...")
    X_test_processed = preprocessor.transform(X_test)

    feature_names = preprocessor.get_feature_names_out()
    
    X_train_df = pd.DataFrame(X_train_processed, columns=feature_names)
    X_test_df = pd.DataFrame(X_test_processed, columns=feature_names)
    
    y_train_reset = y_train.reset_index(drop=True)
    y_test_reset = y_test.reset_index(drop=True)

    train_df = pd.concat([X_train_df, y_train_reset], axis=1)
    test_df = pd.concat([X_test_df, y_test_reset], axis=1)

    return train_df, test_df


def save_to_db(train_df: pd.DataFrame, test_df: pd.DataFrame, db_path: str):
    """Saves the processed datasets to the ML database."""
    conn = sqlite3.connect(db_path)
    try:
        print(f"Saving to {db_path}...")
        train_df.to_sql('training_set', conn, if_exists='replace', index=False)
        test_df.to_sql('testing_set', conn, if_exists='replace', index=False)
        print(f"Successfully saved 'training_set' ({len(train_df)} rows) and 'testing_set' ({len(test_df)} rows).")
    finally:
        conn.close()


def print_consort_summary() -> None:
    """Print CONSORT-style row count summary for publication."""
    print("\n" + "=" * 60)
    print("CONSORT DIAGRAM DATA (Transformation Stage)")
    print("=" * 60)
    for stage, count in row_log.items():
        print(f"  {stage}: {count:,}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    # 1. Load
    print("=" * 60)
    print("TRANSFORMATION PIPELINE - Starting")
    print("=" * 60)
    print(f"[CONFIG] Source DB: {SOURCE_DB_PATH}")
    print(f"[CONFIG] Output DB: {OUTPUT_DB_PATH}")
    print(f"[CONFIG] Test Size: {TEST_SIZE}")
    print(f"[CONFIG] Imputer: max_iter={IMPUTER_MAX_ITER}, random_state={IMPUTER_RANDOM_STATE}")
    print(f"[CONFIG] Inputs: {len(INPUTS)} features")
    print(f"[CONFIG] Targets: {len(TARGETS)} targets")
    print(f"[CONFIG] Categorical: {CATEGORICAL_INPUTS}")
    print()
    
    print("Loading raw dataset...")
    df_raw = get_raw_dataset(SOURCE_DB_PATH, SQL_RAW_DATASET_PATH)
    
    # 2. Process
    print("\nProcessing data...")
    train_df, test_df = preprocess_data(df_raw)
    
    # 3. Save
    save_to_db(train_df, test_df, OUTPUT_DB_PATH)
    
    # 4. Print CONSORT summary
    print_consort_summary()
    
    print("=" * 60)
    print("=== Transformation Complete. ===")
    print("=" * 60)