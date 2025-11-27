import pandas as pd
import os
import sqlite3
import duckdb

DB_NAME = "nhanes_1st.db"

script_dir = os.path.dirname(os.path.abspath(__file__))
FOLDER_PATH_DB = os.path.join(script_dir, '..', 'databases', DB_NAME)
SQL_FILE_PATH = os.path.join(script_dir, '..', 'databases', 'SQL_queries', 'join_all_tables.sql')
SQL_CARDIO_FILE_PATH = os.path.join(script_dir, '..', 'databases', 'SQL_queries', 'cardiov_target.sql')

conn = sqlite3.connect(FOLDER_PATH_DB)
con = duckdb.connect()

with open(SQL_FILE_PATH, 'r') as file:
    query_join_all_tables = file.read()

df = pd.read_sql_query(query_join_all_tables, conn)
df = df.loc[:, ~df.columns.duplicated()] # because i did left join on all tables it created duplicate SEQN for each table pandas read it

def cardio_target(df: pd.DataFrame , query_path: str )-> pd.DataFrame:
    with open(query_path, 'r') as file:
        query_cardio = file.read()

    con.register('HeartQuestions', df)
    
    result_df = con.execute(query_cardio).df()

    df = df.merge(result_df[['SEQN', 'Cardiovascular_target']], on='SEQN', how='left')
    return df
df = cardio_target(df, SQL_CARDIO_FILE_PATH)

#print(result_df)
print(df['Cardiovascular_target'].isnull().sum())