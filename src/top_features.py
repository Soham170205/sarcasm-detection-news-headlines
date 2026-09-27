import joblib
import numpy as np


# ==========================================
# 1. Load Model and Vectorizer
# ==========================================

model_path = "../models/sarcasm_model.pkl"
vectorizer_path = "../models/tfidf_vectorizer.pkl"

model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)

print("Model loaded successfully!")
print("Vectorizer loaded successfully!")
print()


# ==========================================
# 2. Words to Hide from Public Display
# ==========================================

blocked_words = {
    "fuck",
    "fucking",
    "fucked",
    "fucks",
    "shit",
    "shitty",
    "bullshit",
    "bitch",
    "bitches",
    "ass",
    "asses",
    "asshole",
    "assholes",
    "damn",
    "hell",
    "crap",
    "dick",
    "dicks",
    "piss",
    "pissed"
}


# ==========================================
# 3. Get Feature Names and Coefficients
# ==========================================

feature_names = np.array(
    vectorizer.get_feature_names_out()
)

coefficients = model.coef_[0]


# ==========================================
# 4. Sort Features
# ==========================================

# Sort from strongest positive to weakest
sorted_positive_indices = np.argsort(
    coefficients
)[::-1]

# Sort from strongest negative to weakest
sorted_negative_indices = np.argsort(
    coefficients
)


# ==========================================
# 5. Filter Features
# ==========================================

def is_clean_feature(feature):
    """
    Returns False if a feature contains
    a blocked vulgar word.
    """

    words = feature.lower().split()

    for word in words:
        if word in blocked_words:
            return False

    return True


# ==========================================
# 6. Get Top Clean Features
# ==========================================

sarcastic_features = []

for index in sorted_positive_indices:

    feature = feature_names[index]

    if is_clean_feature(feature):

        sarcastic_features.append(
            (feature, coefficients[index])
        )

    if len(sarcastic_features) == 20:
        break


non_sarcastic_features = []

for index in sorted_negative_indices:

    feature = feature_names[index]

    if is_clean_feature(feature):

        non_sarcastic_features.append(
            (feature, coefficients[index])
        )

    if len(non_sarcastic_features) == 20:
        break


# ==========================================
# 7. Display Sarcastic Features
# ==========================================

print("==========================================")
print("TOP 20 CLEAN FEATURES ASSOCIATED WITH")
print("SARCASM")
print("==========================================")
print()

for feature, coefficient in sarcastic_features:

    print(
        f"{feature:30s} "
        f"{coefficient:.4f}"
    )


# ==========================================
# 8. Display Non-Sarcastic Features
# ==========================================

print()
print("==========================================")
print("TOP 20 CLEAN FEATURES ASSOCIATED WITH")
print("NON-SARCASM")
print("==========================================")
print()

for feature, coefficient in non_sarcastic_features:

    print(
        f"{feature:30s} "
        f"{coefficient:.4f}"
    )


# ==========================================
# 9. Note
# ==========================================

print()
print("==========================================")
print("NOTE")
print("==========================================")
print(
    "Profanity is hidden from this public feature "
    "display only."
)
print(
    "The trained model and dataset remain unchanged."
)