import sqlite3


conn = sqlite3.connect("nhanes_1st.db")
cursor = conn.cursor()

#drop vitals table
cursor.execute("DROP TABLE IF EXISTS Vitals")


conn.commit()
conn.close()
print("Table 'Vitals' has been dropped successfully.")