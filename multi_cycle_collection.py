"""
Multi-Cycle Data Collection Script
Collects and merges all NHANES data across all available cycles
"""

from data_pipeline import NHANESDataPipeline
import pandas as pd
import numpy as np

# Initialize pipeline
base_path = r'E:\MY PROJECT\Python\UNI Project\Data'
pipeline = NHANESDataPipeline(base_path)

print("=" * 80)
print("MULTI-CYCLE NHANES DATA COLLECTION")
print("=" * 80)

# =============================================================================
# OPTION 1: Load Each Cycle Separately (Recommended for large datasets)
# =============================================================================
print("\n" + "=" * 80)
print("OPTION 1: Loading Each Cycle Separately")
print("=" * 80)

cycles = ['2013-2014', '2015-2016', '2017-2018',  '2017-2020' , '2021-2022'   ]
cycle_datasets = {}

for cycle in cycles:
    print(f"\n{'='*60}")
    print(f"Processing Cycle: {cycle}")
    print(f"{'='*60}")
    
    try:
        # Load and merge all data for this cycle
        df_cycle = pipeline.create_master_dataset(cycle, merge_how='outer')
        cycle_datasets[cycle] = df_cycle
        
        # Show summary
        print(f"\n✓ {cycle} completed:")
        print(f"  - Participants: {df_cycle['SEQN'].nunique()}")
        print(f"  - Total rows: {len(df_cycle)}")
        print(f"  - Total columns: {len(df_cycle.columns)}")
        
        # Save individual cycle
        output_file = f'master_dataset_{cycle.replace("-", "_")}.csv'
        df_cycle.to_csv(output_file, index=False)
        print(f"  - Saved to: {output_file}")
        
    except Exception as e:
        print(f"\n✗ Error processing {cycle}: {str(e)}")
        continue

print("\n" + "=" * 80)
print(f"Individual cycle datasets created: {len(cycle_datasets)}")
print("=" * 80)


# =============================================================================
# OPTION 2: Combine All Cycles into One Dataset
# =============================================================================
print("\n" + "=" * 80)
print("OPTION 2: Combining All Cycles into Single Dataset")
print("=" * 80)

try:
    # Method A: Using the pipeline's built-in multi-cycle function
    all_cycles = ['2013-2014', '2015-2016', '2017-2018', '2017-2020' , '2021-2022'   ]
    df_combined = pipeline.create_multi_cycle_dataset(all_cycles, merge_how='outer')
    
    print(f"\n✓ Combined dataset created:")
    print(f"  - Total participants: {df_combined['SEQN'].nunique()}")
    print(f"  - Total rows: {len(df_combined)}")
    print(f"  - Total columns: {len(df_combined.columns)}")
    print(f"  - Cycles included: {df_combined['Cycle'].unique().tolist()}")
    
    # Participants per cycle
    print(f"\n  Participants by cycle:")
    cycle_counts = df_combined.groupby('Cycle')['SEQN'].nunique().sort_index()
    for cycle, count in cycle_counts.items():
        print(f"    - {cycle}: {count} participants")
    
    # Save combined dataset
    df_combined.to_csv('master_dataset_all_cycles.csv', index=False)
    df_combined.to_parquet('master_dataset_all_cycles.parquet', index=False)
    print(f"\n  ✓ Combined dataset saved (CSV and Parquet)")
    
except Exception as e:
    print(f"\n✗ Error creating combined dataset: {str(e)}")


# =============================================================================
# OPTION 3: Load Specific Categories Across All Cycles
# =============================================================================
print("\n" + "=" * 80)
print("OPTION 3: Loading Specific Categories Across All Cycles")
print("=" * 80)

# Example: Collect only Examination data across all cycles
print("\nCollecting EXAMINATION data across all cycles...")

examination_data_all_cycles = []

for cycle in cycles:
    print(f"\n  Loading {cycle}...")
    try:
        loaded_data = pipeline.load_cycle_data(cycle)
        
        # Start with demographics as base
        base_df = loaded_data['demographics'][0].copy()
        base_df['Cycle'] = cycle
        
        # Merge only examination data
        if loaded_data['examination']:
            df_exam = pipeline.merge_dataframes(base_df, loaded_data['examination'])
            examination_data_all_cycles.append(df_exam)
            print(f"    ✓ {cycle}: {len(df_exam)} rows, {len(df_exam.columns)} columns")
        else:
            print(f"    ✗ No examination data found")
            
    except Exception as e:
        print(f"    ✗ Error: {str(e)}")

# Combine all examination data
if examination_data_all_cycles:
    df_examination_combined = pd.concat(examination_data_all_cycles, axis=0, ignore_index=True)
    print(f"\n✓ Combined Examination Dataset:")
    print(f"  - Total rows: {len(df_examination_combined)}")
    print(f"  - Total columns: {len(df_examination_combined.columns)}")
    print(f"  - Unique participants: {df_examination_combined['SEQN'].nunique()}")
    
    df_examination_combined.to_csv('examination_data_all_cycles.csv', index=False)
    print(f"  ✓ Saved to: examination_data_all_cycles.csv")


# Example: Collect only Laboratory data across all cycles
print("\n\nCollecting LABORATORY data across all cycles...")

laboratory_data_all_cycles = []

for cycle in cycles:
    print(f"\n  Loading {cycle}...")
    try:
        loaded_data = pipeline.load_cycle_data(cycle)
        
        # Start with demographics as base
        base_df = loaded_data['demographics'][0].copy()
        base_df['Cycle'] = cycle
        
        # Merge only laboratory data
        if loaded_data['laboratory']:
            df_lab = pipeline.merge_dataframes(base_df, loaded_data['laboratory'])
            laboratory_data_all_cycles.append(df_lab)
            print(f"    ✓ {cycle}: {len(df_lab)} rows, {len(df_lab.columns)} columns")
        else:
            print(f"    ✗ No laboratory data found")
            
    except Exception as e:
        print(f"    ✗ Error: {str(e)}")

# Combine all laboratory data
if laboratory_data_all_cycles:
    df_laboratory_combined = pd.concat(laboratory_data_all_cycles, axis=0, ignore_index=True)
    print(f"\n✓ Combined Laboratory Dataset:")
    print(f"  - Total rows: {len(df_laboratory_combined)}")
    print(f"  - Total columns: {len(df_laboratory_combined.columns)}")
    print(f"  - Unique participants: {df_laboratory_combined['SEQN'].nunique()}")
    
    df_laboratory_combined.to_csv('laboratory_data_all_cycles.csv', index=False)
    print(f"  ✓ Saved to: laboratory_data_all_cycles.csv")


# =============================================================================
# OPTION 4: Custom Category Combinations
# =============================================================================
print("\n" + "=" * 80)
print("OPTION 4: Custom Category Combinations Across All Cycles")
print("=" * 80)

# Example: Demographics + Laboratory + Examination only (no questionnaire)
print("\nCollecting Demographics + Laboratory + Examination across all cycles...")

custom_data_all_cycles = []

for cycle in cycles:
    print(f"\n  Processing {cycle}...")
    try:
        loaded_data = pipeline.load_cycle_data(cycle)
        
        # Start with demographics
        base_df = loaded_data['demographics'][0].copy()
        base_df['Cycle'] = cycle
        
        # Merge demographics files if multiple
        if len(loaded_data['demographics']) > 1:
            base_df = pipeline.merge_dataframes(base_df, loaded_data['demographics'][1:])
        
        # Merge laboratory
        if loaded_data['laboratory']:
            base_df = pipeline.merge_dataframes(base_df, loaded_data['laboratory'])
        
        # Merge examination
        if loaded_data['examination']:
            base_df = pipeline.merge_dataframes(base_df, loaded_data['examination'])
        
        custom_data_all_cycles.append(base_df)
        print(f"    ✓ {cycle}: {len(base_df)} rows, {len(base_df.columns)} columns")
        
    except Exception as e:
        print(f"    ✗ Error: {str(e)}")

# Combine all custom data
if custom_data_all_cycles:
    df_custom_combined = pd.concat(custom_data_all_cycles, axis=0, ignore_index=True)
    print(f"\n✓ Combined Custom Dataset (Demo + Lab + Exam):")
    print(f"  - Total rows: {len(df_custom_combined)}")
    print(f"  - Total columns: {len(df_custom_combined.columns)}")
    print(f"  - Unique participants: {df_custom_combined['SEQN'].nunique()}")
    
    df_custom_combined.to_csv('custom_data_all_cycles.csv', index=False)
    print(f"  ✓ Saved to: custom_data_all_cycles.csv")


# =============================================================================
# SUMMARY STATISTICS
# =============================================================================
print("\n" + "=" * 80)
print("FINAL SUMMARY")
print("=" * 80)

if 'df_combined' in locals():
    print("\nComplete Dataset (All Categories, All Cycles):")
    print(f"  Total participants: {df_combined['SEQN'].nunique()}")
    print(f"  Total rows: {len(df_combined)}")
    print(f"  Total columns: {len(df_combined.columns)}")

if 'df_examination_combined' in locals():
    print(f"\nExamination Data Only:")
    print(f"  Total participants: {df_examination_combined['SEQN'].nunique()}")
    print(f"  Total rows: {len(df_examination_combined)}")
    print(f"  Total columns: {len(df_examination_combined.columns)}")

if 'df_laboratory_combined' in locals():
    print(f"\nLaboratory Data Only:")
    print(f"  Total participants: {df_laboratory_combined['SEQN'].nunique()}")
    print(f"  Total rows: {len(df_laboratory_combined)}")
    print(f"  Total columns: {len(df_laboratory_combined.columns)}")

print("\n" + "=" * 80)
print("DATA COLLECTION COMPLETE!")
print("=" * 80)
print("\nGenerated files:")
print("  - Individual cycle datasets: master_dataset_[cycle].csv")
print("  - Combined all cycles: master_dataset_all_cycles.csv")
print("  - Examination only: examination_data_all_cycles.csv")
print("  - Laboratory only: laboratory_data_all_cycles.csv")
print("  - Custom combination: custom_data_all_cycles.csv")

# Quick verification
print("\n" + "=" * 80)
print("QUICK VERIFICATION")
print("=" * 80)
try:
    import os
    if os.path.exists('master_dataset_all_cycles.csv'):
        df_verify = pd.read_csv('master_dataset_all_cycles.csv')
        print(f"\n✓ master_dataset_all_cycles.csv:")
        print(f"  Rows: {len(df_verify):,}")
        if 'Cycle' in df_verify.columns:
            print(f"  Cycles: {sorted(df_verify['Cycle'].unique())}")
            print(f"  Rows per cycle:")
            for cycle, count in df_verify['Cycle'].value_counts().sort_index().items():
                print(f"    {cycle}: {count:,}")
except Exception as e:
    print(f"Verification skipped: {e}")
