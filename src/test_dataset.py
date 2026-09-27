import json

dataset_path = "../data/Sarcasm_Headlines_Dataset.json"

with open(dataset_path, "r", encoding="utf-8") as file:
    data = [json.loads(line) for line in file]

print("Number of headlines:", len(data))

print("\nFirst headline:")
print(data[0])