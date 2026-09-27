import json
import re
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


# ==========================================
# 1. Load Dataset
# ==========================================

dataset_path = "../data/Sarcasm_Headlines_Dataset.json"

with open(dataset_path, "r", encoding="utf-8") as file:
    data = [json.loads(line) for line in file]

df = pd.DataFrame(data)

df = df[["headline", "is_sarcastic"]]

print("Dataset loaded!")
print("Total headlines:", len(df))
print()


# ==========================================
# 2. Clean Text
# ==========================================

def clean_text(text):
    text = text.lower()

    text = re.sub(
        r"http\S+|www\S+|https\S+",
        " ",
        text
    )

    text = re.sub(
        r"[^a-zA-Z\s]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


df["cleaned_headline"] = df["headline"].apply(clean_text)


# ==========================================
# 3. Features and Labels
# ==========================================

X = df["cleaned_headline"]
y = df["is_sarcastic"]


# ==========================================
# 4. Train/Test Split
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training samples:", len(X_train))
print("Testing samples :", len(X_test))
print()


# ==========================================
# 5. Final TF-IDF
# ==========================================

vectorizer = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 2),
    min_df=2
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print("TF-IDF vectorization completed!")
print("Training matrix:", X_train_tfidf.shape)
print("Testing matrix :", X_test_tfidf.shape)
print()


# ==========================================
# 6. Final Logistic Regression
# ==========================================

print("Training final Logistic Regression model...")

model = LogisticRegression(
    C=2.0,
    max_iter=1000,
    random_state=42
)

model.fit(
    X_train_tfidf,
    y_train
)

print("Training completed!")
print()


# ==========================================
# 7. Test Model
# ==========================================

y_pred = model.predict(X_test_tfidf)

accuracy = accuracy_score(
    y_test,
    y_pred
)

print("==========================================")
print("FINAL MODEL RESULTS")
print("==========================================")

print()
print(
    "Accuracy:",
    round(accuracy * 100, 2),
    "%"
)

print()
print("Classification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=[
            "Not Sarcastic",
            "Sarcastic"
        ]
    )
)


# ==========================================
# 8. Save Final Model
# ==========================================

model_path = "../models/sarcasm_model.pkl"
vectorizer_path = "../models/tfidf_vectorizer.pkl"

joblib.dump(
    model,
    model_path
)

joblib.dump(
    vectorizer,
    vectorizer_path
)

print()
print("==========================================")
print("FINAL MODEL SAVED")
print("==========================================")

print("Model     :", model_path)
print("Vectorizer:", vectorizer_path)