import json
import pandas as pd
import re


# -----------------------------
# 1. Load dataset
# -----------------------------

dataset_path = "../data/Sarcasm_Headlines_Dataset.json"

with open(dataset_path, "r", encoding="utf-8") as file:
    data = [json.loads(line) for line in file]

df = pd.DataFrame(data)

# Keep only required columns
df = df[["headline", "is_sarcastic"]]


# -----------------------------
# 2. Text cleaning function
# -----------------------------

def clean_text(text):
    # Convert to lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)

    # Remove punctuation and special characters
    text = re.sub(r"[^a-zA-Z\s]", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


# -----------------------------
# 3. Apply cleaning
# -----------------------------

df["cleaned_headline"] = df["headline"].apply(clean_text)


# -----------------------------
# 4. Display results
# -----------------------------

print("Cleaning completed successfully!")
print()

print("Original vs Cleaned Headlines:")
print()

for i in range(5):
    print("Original :", df.iloc[i]["headline"])
    print("Cleaned  :", df.iloc[i]["cleaned_headline"])
    print("-" * 80)


# -----------------------------
# 5. Check for empty headlines
# -----------------------------

empty_headlines = df["cleaned_headline"].eq("").sum()

print()
print("Empty headlines after cleaning:", empty_headlines)
print("Total headlines:", len(df))