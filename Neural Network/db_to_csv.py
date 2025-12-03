import sqlite3
import pandas as pd
import os
from pathlib import Path

def get_table_names(db_path):
    """Get list of table names from the SQLite database."""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        return [table[0] for table in tables]
    except Exception as e:
        print(f"Error getting table names: {e}")
        return []
    finally:
        if conn:
            conn.close()

def convert_db_to_csv(db_path, output_dir='datasets'):
    """Convert all tables in SQLite database to CSV files and combine them."""
    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Get all table names
    tables = get_table_names(db_path)
    
    if not tables:
        print("No tables found in the database.")
        return
    
    print(f"Found {len(tables)} tables in the database.")
    
    all_data = []
    
    try:
        # Connect to the SQLite database
        conn = sqlite3.connect(db_path)
        
        # Process each table
        for table in tables:
            try:
                # Read table into pandas DataFrame
                query = f"SELECT * FROM {table}"
                df = pd.read_sql_query(query, conn)
                
                # Add data type column
                df['data_type'] = 'training' if 'train' in table.lower() else 'testing'
                
                # Save individual CSV
                csv_file = output_path / f"{table}.csv"
                df.to_csv(csv_file, index=False)
                print(f"Successfully converted table '{table}' to {csv_file}")
                
                # Add to combined data
                all_data.append(df)
                
            except Exception as e:
                print(f"Error processing table '{table}': {e}")
        
        # Combine all data if we have multiple tables
        if len(all_data) > 1:
            combined_df = pd.concat(all_data, ignore_index=True)
            combined_file = output_path / "combined_data.csv"
            combined_df.to_csv(combined_file, index=False)
            print(f"\nSuccessfully combined all data into: {combined_file}")
            print(f"Total records: {len(combined_df)}")
            print(f"Training records: {len(combined_df[combined_df['data_type'] == 'training'])}")
            print(f"Testing records: {len(combined_df[combined_df['data_type'] == 'testing'])}")
    
    except Exception as e:
        print(f"Error processing database: {e}")
    
    finally:
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    db_path = r"e:\MY PROJECT\Machine Learning\Kidney Neural Network\datasets\ML_data.db"
    output_dir = r"e:\MY PROJECT\Machine Learning\Kidney Neural Network\datasets"
    
    print(f"Starting database conversion for: {db_path}")
    convert_db_to_csv(db_path, output_dir)
    print("Conversion completed.")
