import pandas as pd
import os
import sqlite3

DB_NAME = "nhanes_1st.db"

script_dir = os.path.dirname(os.path.abspath(__file__))
FOLDER_PATH_DB = os.path.join(script_dir, '..', 'databases', DB_NAME)
db_path = os.path.normpath(FOLDER_PATH_DB)

conn = sqlite3.connect(FOLDER_PATH_DB)
query = "SELECT * FROM Demographics"

df = pd.read_sql_query(query, conn)

print(df.head(5))