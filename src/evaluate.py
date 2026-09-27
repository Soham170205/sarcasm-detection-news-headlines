
import json
import re
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


# ==========================================
# 1. Load Dataset
# ==========================================

dataset_path = "../data/Sarcasm_Headlines_Dataset.json"

with open(dataset_path, "r", encoding="utf-8") as file:
    data = [json.loads(line) for line in file]

df = pd.DataFrame(data)

# Keep only required columns
df = df[["headline", "is_sarcastic"]]


# ==========================================
# 2. Clean Text
# ==========================================

def clean_text(text):
    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", " ", text)

    # Replace punctuation with spaces
    text = re.sub(r"[^a-zA-Z\s]", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


df["cleaned_headline"] = df["headline"].apply(clean_text)


# ==========================================
# 3. Separate Features and Labels
# ==========================================

X = df["cleaned_headline"]
y = df["is_sarcastic"]


# ==========================================
# 4. Recreate Same Test Split
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ==========================================
# 5. Load Saved Model and Vectorizer
# ==========================================

model_path = "../models/sarcasm_model.pkl"
vectorizer_path = "../models/tfidf_vectorizer.pkl"

model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)

print("Model loaded successfully!")
print("Vectorizer loaded successfully!")
print()


# ==========================================
# 6. Transform Test Data
# ==========================================

X_test_tfidf = vectorizer.transform(X_test)


# ==========================================
# 7. Make Predictions
# ==========================================

y_pred = model.predict(X_test_tfidf)


# ==========================================
# 8. Create Confusion Matrix
# ==========================================

cm = confusion_matrix(y_test, y_pred)

print("Confusion Matrix:")
print(cm)
print()


# ==========================================
# 9. Display Confusion Matrix
# ==========================================

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Not Sarcastic", "Sarcastic"]
)

disp.plot()

plt.title("Sarcasm Detection - Confusion Matrix")
plt.tight_layout()

plt.show()

