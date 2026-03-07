import sqlite3
import pandas as pd

conn = sqlite3.connect("insurance.db")

query = """
SELECT CLAIM_AMOUNT, PREMIUM_AMOUNT, INCIDENT_SEVERITY
FROM claims
WHERE CLAIM_AMOUNT > 5000
"""

df = pd.read_sql(query, conn)

print(df.head())