import sqlite3
import os

db_path = "ML_data.db"
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Get all tables
cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = cursor.fetchall()
print("Tables in ML_data.db:")
for t in tables:
    print(f"  - {t[0]}")

# For each table, get column info
for table in tables:
    tname = table[0]
    print(f"\n=== {tname} ===")
    cursor.execute(f"PRAGMA table_info({tname})")
    columns = cursor.fetchall()
    print("Columns:")
    for col in columns:
        print(f"  {col[1]} ({col[2]})")
    
    # Get row count
    cursor.execute(f"SELECT COUNT(*) FROM {tname}")
    count = cursor.fetchone()[0]
    print(f"Row count: {count}")

conn.close()
