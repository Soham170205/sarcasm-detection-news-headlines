import json
import re
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


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
# 4. Create Train/Test Split
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ==========================================
# 5. Create Validation Split
# ==========================================

X_train, X_validation, y_train, y_validation = train_test_split(
    X_train,
    y_train,
    test_size=0.2,
    random_state=42,
    stratify=y_train
)


print("Data split completed!")
print()
print("Training samples  :", len(X_train))
print("Validation samples:", len(X_validation))
print("Testing samples   :", len(X_test))
print()


# ==========================================
# 6. Settings to Test
# ==========================================

c_values = [
    0.1,
    0.5,
    1.0,
    2.0,
    5.0
]

ngram_settings = [
    (1, 1),
    (1, 2)
]


# ==========================================
# 7. Run Experiments
# ==========================================

best_accuracy = 0
best_c = None
best_ngram = None

print("==========================================")
print("MODEL TUNING")
print("==========================================")
print()

for ngram_range in ngram_settings:

    print("Testing ngram_range:", ngram_range)
    print()

    # Create TF-IDF vectorizer
    vectorizer = TfidfVectorizer(
        max_features=10000,
        ngram_range=ngram_range,
        min_df=2
    )

    # Fit ONLY on training data
    X_train_tfidf = vectorizer.fit_transform(X_train)

    # Transform validation data
    X_validation_tfidf = vectorizer.transform(
        X_validation
    )

    for c in c_values:

        print("Testing C =", c)

        # Create model
        model = LogisticRegression(
            C=c,
            max_iter=1000,
            random_state=42
        )

        # Train
        model.fit(
            X_train_tfidf,
            y_train
        )

        # Predict validation set
        validation_predictions = model.predict(
            X_validation_tfidf
        )

        # Calculate accuracy
        accuracy = accuracy_score(
            y_validation,
            validation_predictions
        )

        print(
            "Validation accuracy:",
            round(accuracy * 100, 2),
            "%"
        )

        print()

        # Save best configuration
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_c = c
            best_ngram = ngram_range


# ==========================================
# 8. Display Best Configuration
# ==========================================

print("==========================================")
print("BEST CONFIGURATION")
print("==========================================")

print()
print("Best C:", best_c)
print("Best ngram_range:", best_ngram)
print(
    "Best validation accuracy:",
    round(best_accuracy * 100, 2),
    "%"
)

print()


# ==========================================
# 9. Evaluate Best Configuration
#    on Untouched Test Set
# ==========================================

print("==========================================")
print("FINAL TEST EVALUATION")
print("==========================================")
print()

# Create final vectorizer
final_vectorizer = TfidfVectorizer(
    max_features=10000,
    ngram_range=best_ngram,
    min_df=2
)

# Fit on the complete training portion
X_train_final = final_vectorizer.fit_transform(
    pd.concat([X_train, X_validation])
)

# Transform untouched test data
X_test_final = final_vectorizer.transform(
    X_test
)

# Create final model
final_model = LogisticRegression(
    C=best_c,
    max_iter=1000,
    random_state=42
)

# Train on all training data
final_model.fit(
    X_train_final,
    pd.concat([y_train, y_validation])
)

# Predict test data
test_predictions = final_model.predict(
    X_test_final
)

# Calculate final accuracy
test_accuracy = accuracy_score(
    y_test,
    test_predictions
)

print(
    "Final test accuracy:",
    round(test_accuracy * 100, 2),
    "%"
)

print()

print("Baseline accuracy: 84.43%")

print(
    "Difference:",
    round(
        (test_accuracy - 0.8443) * 100,
        2
    ),
    "percentage points"
)