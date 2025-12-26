"""
Comprehensive EDA Summary Generator
Executes analysis from notebooks 01-07 and generates a full column summary
"""

import pandas as pd
import numpy as np
import sqlite3
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# DATA LOADING
# ============================================================================
print("="*80)
print("LOADING DATA FROM DATABASE")
print("="*80)

conn = sqlite3.connect('../databases/nhanes_1st.db')
df = pd.read_sql_query('SELECT * FROM raw_dataset', conn)
conn.close()

print(f"Dataset loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")

# Define target columns (to be highlighted in summary)
TARGET_COLUMNS = [
    'has_cardiovascular_disease',  # Binary - CVD
    'high_waist_circumference',    # Binary - Metabolic
    'high_triglycerides_mg_dl',    # Binary - Metabolic
    'low_hdl_mg_dl',               # Binary - Metabolic
    'high_blood_pressure',         # Binary - Metabolic
    'high_glucose_mg_dl',          # Binary - Metabolic
    'albuminuria_risk',            # Multi-class - Kidney
    'liver_dysfunction'            # Binary - Liver
]

# ============================================================================
# 01. DATA EXPLORATION
# ============================================================================
print("\n" + "="*80)
print("01. DATA EXPLORATION")
print("="*80)

print(f"\nTotal Records: {len(df):,}")
print(f"Total Features: {df.shape[1]}")
print(f"Memory Usage: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")

# Column types
print(f"\nColumn Types:")
print(f"  - Numeric (float64): {len(df.select_dtypes(include=['float64']).columns)}")
print(f"  - Integer (int64): {len(df.select_dtypes(include=['int64']).columns)}")

# ============================================================================
# 02. DATA QUALITY ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("02. DATA QUALITY ANALYSIS")
print("="*80)

# Missing values
missing = df.isnull().sum()
missing_pct = (missing / len(df)) * 100
missing_df = pd.DataFrame({
    'Missing': missing,
    'Percent': missing_pct,
    'Is_Target': [col in TARGET_COLUMNS for col in df.columns]
})

print("\nMissing Values Summary:")
for col in df.columns:
    m = missing[col]
    p = missing_pct[col]
    target_flag = "[TARGET]" if col in TARGET_COLUMNS else ""
    print(f"  {col}: {m:,} ({p:.1f}%) {target_flag}")

# Duplicates
duplicates = df.duplicated().sum()
print(f"\nDuplicate Rows: {duplicates:,} ({duplicates/len(df)*100:.1f}%)")

# ============================================================================
# 03. UNIVARIATE ANALYSIS - Statistics for all columns
# ============================================================================
print("\n" + "="*80)
print("03. UNIVARIATE ANALYSIS")
print("="*80)

column_summaries = []

for col in df.columns:
    is_target = col in TARGET_COLUMNS
    summary = {
        'Column': col,
        'Is_Target': is_target,
        'Dtype': str(df[col].dtype),
        'Non_Null': df[col].notnull().sum(),
        'Missing': df[col].isnull().sum(),
        'Missing_Pct': (df[col].isnull().sum() / len(df)) * 100
    }
    
    if df[col].dtype in ['float64', 'int64']:
        valid_data = df[col].dropna()
        summary['Mean'] = valid_data.mean()
        summary['Std'] = valid_data.std()
        summary['Min'] = valid_data.min()
        summary['Max'] = valid_data.max()
        summary['Median'] = valid_data.median()
        summary['Q1'] = valid_data.quantile(0.25)
        summary['Q3'] = valid_data.quantile(0.75)
        summary['Unique'] = df[col].nunique()
        
        # Check if binary
        unique_vals = valid_data.unique()
        if len(unique_vals) == 2 and set(unique_vals).issubset({0, 1, 0.0, 1.0}):
            summary['Type'] = 'Binary'
            pos_count = (valid_data == 1).sum()
            neg_count = (valid_data == 0).sum()
            summary['Positive_Count'] = pos_count
            summary['Negative_Count'] = neg_count
            summary['Positive_Pct'] = (pos_count / len(valid_data)) * 100
            summary['Imbalance_Ratio'] = max(pos_count, neg_count) / min(pos_count, neg_count) if min(pos_count, neg_count) > 0 else float('inf')
        elif len(unique_vals) <= 10:
            summary['Type'] = 'Categorical'
            summary['Value_Counts'] = df[col].value_counts(dropna=True).to_dict()
        else:
            summary['Type'] = 'Continuous'
            # Skewness and Kurtosis
            if len(valid_data) > 3:
                summary['Skewness'] = valid_data.skew()
                summary['Kurtosis'] = valid_data.kurtosis()
    
    column_summaries.append(summary)

# ============================================================================
# 04. BIVARIATE ANALYSIS - Correlations
# ============================================================================
print("\n" + "="*80)
print("04. BIVARIATE ANALYSIS")
print("="*80)

# Calculate correlations
numeric_cols = df.select_dtypes(include=[np.number]).columns
corr_matrix = df[numeric_cols].corr()

# Find highest correlations (excluding diagonal)
high_corrs = []
for i, col1 in enumerate(corr_matrix.columns):
    for j, col2 in enumerate(corr_matrix.columns):
        if i < j:  # Upper triangle only
            corr_val = corr_matrix.loc[col1, col2]
            if abs(corr_val) > 0.3:  # Threshold
                high_corrs.append({
                    'Feature_1': col1,
                    'Feature_2': col2,
                    'Correlation': corr_val
                })

high_corrs = sorted(high_corrs, key=lambda x: abs(x['Correlation']), reverse=True)

print("\nHigh Correlations (|r| > 0.3):")
for hc in high_corrs[:15]:
    print(f"  {hc['Feature_1']} <-> {hc['Feature_2']}: {hc['Correlation']:.3f}")

# ============================================================================
# 05. OUTLIER ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("05. OUTLIER ANALYSIS")
print("="*80)

outlier_summary = []

for col in df.select_dtypes(include=[np.number]).columns:
    valid_data = df[col].dropna()
    if len(valid_data) == 0:
        continue
    
    # IQR method
    Q1 = valid_data.quantile(0.25)
    Q3 = valid_data.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    iqr_outliers = ((valid_data < lower_bound) | (valid_data > upper_bound)).sum()
    
    # Z-score method
    z_scores = np.abs(stats.zscore(valid_data))
    z_outliers = (z_scores > 3).sum()
    
    outlier_summary.append({
        'Column': col,
        'IQR_Outliers': iqr_outliers,
        'IQR_Outlier_Pct': (iqr_outliers / len(valid_data)) * 100,
        'Z_Outliers': z_outliers,
        'Z_Outlier_Pct': (z_outliers / len(valid_data)) * 100,
        'Is_Target': col in TARGET_COLUMNS
    })

print("\nOutliers by Column (IQR and Z-Score methods):")
for o in outlier_summary:
    target_flag = "[TARGET]" if o['Is_Target'] else ""
    print(f"  {o['Column']}: IQR={o['IQR_Outliers']:,} ({o['IQR_Outlier_Pct']:.1f}%), Z-Score={o['Z_Outliers']:,} ({o['Z_Outlier_Pct']:.1f}%) {target_flag}")

# ============================================================================
# 06. MULTIVARIATE ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("06. MULTIVARIATE ANALYSIS")
print("="*80)

# Feature groups correlations with targets
print("\nFeature correlations with target variables:")
for target in TARGET_COLUMNS:
    if target in df.columns:
        print(f"\n  {target}:")
        for col in df.columns:
            if col != target and col not in TARGET_COLUMNS:
                if df[col].dtype in ['float64', 'int64'] and df[target].dtype in ['float64', 'int64']:
                    valid_mask = df[col].notnull() & df[target].notnull()
                    if valid_mask.sum() > 10:
                        corr = df.loc[valid_mask, col].corr(df.loc[valid_mask, target])
                        if abs(corr) > 0.1:  # Only show meaningful correlations
                            print(f"    {col}: {corr:.3f}")

# ============================================================================
# 07. GENERATE COMPREHENSIVE SUMMARY
# ============================================================================
print("\n" + "="*80)
print("07. GENERATING COMPREHENSIVE SUMMARY")
print("="*80)

# Generate markdown summary
print("\nGenerating Summary.md...")

# Categorize columns
demographics = ['age', 'gender', 'ethnicity', 'income_ratio']
physical = ['body_mass_index', 'height_cm', 'heart_rate_bpm']
blood_tests = ['white_blood_cells_count', 'platelets_count', 'hemoglobin_g_dl', 'mean_corpuscular_volume_fL']
kidney = ['creatinine_mg_dl', 'uric_acid_mg_dl']
liver = ['liver_ast_U_L', 'bilirubin_mg_dl', 'liver_ggt_U_L']
electrolytes = ['sodium_mmol_L', 'potassium_mmol_L']
lipids = ['cholesterol_mg_dl']
lifestyle = ['alcohol_drinks_per_week', 'smoking_status']

# Output the summary data as JSON for later use
import json

output_data = {
    'dataset_info': {
        'total_records': len(df),
        'total_features': df.shape[1],
        'age_range': f"{df['age'].min():.0f} - {df['age'].max():.0f}",
        'mean_age': df['age'].mean(),
        'mean_bmi': df['body_mass_index'].mean()
    },
    'column_summaries': column_summaries,
    'high_correlations': high_corrs[:15],
    'outlier_summary': outlier_summary,
    'target_columns': TARGET_COLUMNS,
    'column_groups': {
        'demographics': demographics,
        'physical': physical,
        'blood_tests': blood_tests,
        'kidney': kidney,
        'liver': liver,
        'electrolytes': electrolytes,
        'lipids': lipids,
        'lifestyle': lifestyle,
        'targets': TARGET_COLUMNS
    }
}

# Save to JSON for reference
with open('eda_summary_data.json', 'w') as f:
    json.dump(output_data, f, indent=2, default=str)

print("Summary data saved to eda_summary_data.json")

# Now generate the markdown file
print("\n" + "="*80)
print("COLUMN SUMMARY TABLES FOR MARKDOWN")
print("="*80)

# Print formatted output for each column
for summary in column_summaries:
    col = summary['Column']
    is_target = summary['Is_Target']
    target_marker = ">>> [TARGET] <<<" if is_target else ""
    
    print(f"\n### `{col}` {target_marker}")
    print(f"- **Type:** {summary.get('Type', 'Unknown')}")
    print(f"- **Non-Null:** {summary['Non_Null']:,} | **Missing:** {summary['Missing']:,} ({summary['Missing_Pct']:.1f}%)")
    
    if summary.get('Type') == 'Binary':
        print(f"- **Positive (1):** {summary.get('Positive_Count', 0):,} ({summary.get('Positive_Pct', 0):.1f}%)")
        print(f"- **Negative (0):** {summary.get('Negative_Count', 0):,}")
        print(f"- **Imbalance Ratio:** {summary.get('Imbalance_Ratio', 0):.2f}:1")
    elif summary.get('Type') == 'Continuous':
        print(f"- **Mean:** {summary.get('Mean', 0):.2f} | **Std:** {summary.get('Std', 0):.2f}")
        print(f"- **Min:** {summary.get('Min', 0):.2f} | **Max:** {summary.get('Max', 0):.2f}")
        print(f"- **Median:** {summary.get('Median', 0):.2f} | **Q1:** {summary.get('Q1', 0):.2f} | **Q3:** {summary.get('Q3', 0):.2f}")
        if 'Skewness' in summary:
            print(f"- **Skewness:** {summary.get('Skewness', 0):.2f} | **Kurtosis:** {summary.get('Kurtosis', 0):.2f}")
    elif summary.get('Type') == 'Categorical':
        print(f"- **Unique Values:** {summary.get('Unique', 0)}")
        if 'Value_Counts' in summary:
            print(f"- **Distribution:** {summary['Value_Counts']}")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
