import pandas as pd
from sklearn.datasets import load_wine
import os

# Load dataset
data = load_wine(as_frame=True)
df = data.frame

# Full dataset (v2)
os.makedirs('data/v2', exist_ok=True)
df.to_csv('data/v2/wine.csv', index=False)

# Partial dataset (v1) - e.g., first 50% of rows
os.makedirs('data/v1', exist_ok=True)
df_v1 = df.sample(frac=0.5, random_state=42)
df_v1.to_csv('data/v1/wine.csv', index=False)

print(f"Created v1 dataset with {len(df_v1)} rows in 'data/v1/wine.csv'")
print(f"Created v2 dataset with {len(df)} rows in 'data/v2/wine.csv'")
