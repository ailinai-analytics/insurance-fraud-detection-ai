import pandas as pd
import sqlite3

print("Loading data...")

df = pd.read_csv("outputs/insurance_clean.csv")

print("Creating database...")

conn = sqlite3.connect("insurance.db")

df.to_sql("claims", conn, if_exists="replace", index=False)

print("Database created: insurance.db")