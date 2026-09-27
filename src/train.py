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

# Keep only required columns
df = df[["headline", "is_sarcastic"]]

print("Dataset loaded successfully!")
print("Total headlines:", len(df))
print()


# ==========================================
# 2. Clean Text
# ==========================================

def clean_text(text):
    # Convert to lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", " ", text)

    # Replace punctuation with spaces
    text = re.sub(r"[^a-zA-Z\s]", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


df["cleaned_headline"] = df["headline"].apply(clean_text)

print("Text cleaning completed!")
print()


# ==========================================
# 3. Separate Features and Labels
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

print("Train/Test split completed!")
print()

print("Total samples      :", len(X))
print("Training samples   :", len(X_train))
print("Testing samples    :", len(X_test))
print()


# ==========================================
# 5. TF-IDF Vectorization
# ==========================================

vectorizer = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 2),
    min_df=2
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print("TF-IDF vectorization completed!")
print()

print("Training matrix shape:", X_train_tfidf.shape)
print("Testing matrix shape :", X_test_tfidf.shape)
print()

print("Number of features:",
      len(vectorizer.get_feature_names_out()))
print()


# ==========================================
# 6. Train Logistic Regression
# ==========================================

print("Training Logistic Regression model...")

model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

model.fit(X_train_tfidf, y_train)

print("Model training completed!")
print()


# ==========================================
# 7. Make Predictions
# ==========================================

y_pred = model.predict(X_test_tfidf)


# ==========================================
# 8. Evaluate Model
# ==========================================

accuracy = accuracy_score(y_test, y_pred)

print("==========================================")
print("MODEL RESULTS")
print("==========================================")

print()
print("Accuracy:", accuracy)
print("Accuracy (%):", round(accuracy * 100, 2), "%")

print()
print("Classification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=["Not Sarcastic", "Sarcastic"]
))


# ==========================================
# 9. Save Model and Vectorizer
# ==========================================

model_path = "../models/sarcasm_model.pkl"
vectorizer_path = "../models/tfidf_vectorizer.pkl"

joblib.dump(model, model_path)
joblib.dump(vectorizer, vectorizer_path)

print()
print("==========================================")
print("MODEL SAVED")
print("==========================================")

print("Model     :", model_path)
print("Vectorizer:", vectorizer_path)


