import json
import pandas as pd

# Path to dataset
dataset_path = "../data/Sarcasm_Headlines_Dataset.json"

# Load JSON dataset
with open(dataset_path, "r", encoding="utf-8") as file:
    data = [json.loads(line) for line in file]

# Convert to DataFrame
df = pd.DataFrame(data)

# Keep only the columns we need
df = df[["headline", "is_sarcastic"]]

# Display basic information
print("Dataset loaded successfully!")
print()

print("Number of headlines:", len(df))
print()

print("Columns:")
print(df.columns.tolist())
print()

print("First 5 rows:")
print(df.head())
print()

print("Dataset information:")
print(df.info())
print()

print("Sarcasm distribution:")
print(df["is_sarcastic"].value_counts())