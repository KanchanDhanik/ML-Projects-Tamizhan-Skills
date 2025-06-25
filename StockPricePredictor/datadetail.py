import pandas as pd
df = pd.read_csv("NASDAQ_Composite_Full_History.csv")
print("Columns:", df.columns.tolist())
print("First 5 rows:")
print(df.head())