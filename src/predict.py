import re
import csv
import os
import joblib

model_path = "../models/sarcasm_model.pkl"
vectorizer_path = "../models/tfidf_vectorizer.pkl"

model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)


def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def save_prediction(headline, prediction, sarcastic_probability,
                    not_sarcastic_probability, confidence):

    file_path = "../prediction_results.csv"

    file_exists = os.path.exists(file_path)

    with open(file_path, "a", newline="", encoding="utf-8") as file:

        writer = csv.writer(file)

        # Create column headings only the first time
        if not file_exists:
            writer.writerow([
                "Headline",
                "Prediction",
                "Sarcastic probability",
                "Not sarcastic probability",
                "Confidence"
            ])

        writer.writerow([
            headline,
            prediction,
            f"{sarcastic_probability * 100:.2f}%",
            f"{not_sarcastic_probability * 100:.2f}%",
            f"{confidence * 100:.2f}%"
        ])


def predict_headline(headline):

    cleaned_headline = clean_text(headline)

    headline_tfidf = vectorizer.transform([cleaned_headline])

    prediction_value = model.predict(headline_tfidf)[0]

    probabilities = model.predict_proba(headline_tfidf)[0]

    not_sarcastic_probability = probabilities[0]
    sarcastic_probability = probabilities[1]

    confidence = max(probabilities)

    if prediction_value == 1:
        prediction = "SARCASTIC"
    else:
        prediction = "NOT SARCASTIC"

    print()
    print("=" * 55)
    print("RESULT")
    print("=" * 55)
    print()

    print("Headline:")
    print(headline)

    print()

    print("Prediction              :", prediction)

    print(
        "Sarcastic probability   :",
        f"{sarcastic_probability * 100:.2f}%"
    )

    print(
        "Not sarcastic probability:",
        f"{not_sarcastic_probability * 100:.2f}%"
    )

    print(
        "Confidence              :",
        f"{confidence * 100:.2f}%"
    )

    print("=" * 55)

    # Save the result
    save_prediction(
        headline,
        prediction,
        sarcastic_probability,
        not_sarcastic_probability,
        confidence
    )

    print()
    print("Prediction saved to prediction_results.csv")


print()
print("=" * 55)
print("          SARCASM DETECTION MODEL")
print("=" * 55)
print()

print("Model loaded successfully!")

print()
print("Enter a news headline to classify.")
print("Type 'exit' to close the program.")

while True:

    print()

    headline = input("Headline: ")

    if headline.lower().strip() == "exit":
        print()
        print("Exiting Sarcasm Detection Model...")
        break

    if not headline.strip():
        print()
        print("Please enter a headline.")
        continue

    predict_headline(headline)