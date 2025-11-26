import sqlite3


conn = sqlite3.connect("nhanes_1st.db")
cursor = conn.cursor()

#drop table
#cursor.execute("DROP TABLE IF EXISTS chemi")
#cursor.execute("DROP TABLE IF EXISTS HDL_Cholesterol")
#cursor.execute("DROP TABLE IF EXISTS Triglycerides")
cursor.execute("DROP TABLE IF EXISTS AlcoholUsage")

conn.commit()
conn.close()
print("Table has been dropped successfully.")