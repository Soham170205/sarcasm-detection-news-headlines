"""
Flask API for the Sarcasm Detector demo.
Run from anywhere:  python web/backend/app.py
"""
import csv
import re
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS

# ---------- Paths (absolute, so it works no matter where you run it from) ----------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "models" / "sarcasm_model.pkl"
VECTORIZER_PATH = PROJECT_ROOT / "models" / "tfidf_vectorizer.pkl"
RESULTS_CSV = PROJECT_ROOT / "prediction_results.csv"

# ---------- Load model once at startup ----------
model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)
feature_names = np.array(vectorizer.get_feature_names_out())
coefficients = model.coef_[0]
vocab = vectorizer.vocabulary_

BLOCKED_WORDS = {
    "fuck", "fucking", "fucked", "fucks", "shit", "shitty", "bullshit",
    "bitch", "bitches", "ass", "asses", "asshole", "assholes", "damn",
    "hell", "crap", "dick", "dicks", "piss", "pissed",
}

MODEL_INFO = {
    "algorithm": "Logistic Regression (C=2.0)",
    "vectorizer": "TF-IDF (1-2 grams, max 10,000 features, min_df=2)",
    "dataset": "Sarcasm Headlines Dataset",
    "total_headlines": 26709,
    "sarcastic": 11724,
    "not_sarcastic": 14985,
    "accuracy": 84.91,
    "precision_sarcastic": 84,
    "precision_not_sarcastic": 86,
}

app = Flask(__name__)
CORS(app)


# Same cleaning as training (final_train.py) — must stay identical
def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def is_clean(feature: str) -> bool:
    return not any(w in BLOCKED_WORDS for w in feature.split())


def explain(cleaned: str, tfidf_row):
    """Contribution of each n-gram present = tfidf value * model coefficient."""
    row = tfidf_row.tocoo()
    features = []
    for idx, val in zip(row.col, row.data):
        features.append({
            "term": feature_names[idx],
            "contribution": round(float(val * coefficients[idx]), 4),
        })
    features.sort(key=lambda f: abs(f["contribution"]), reverse=True)

    # Per-word score for highlighting: unigram + half of each bigram it belongs to
    tokens = cleaned.split()
    word_scores = []
    for i, tok in enumerate(tokens):
        score = coefficients[vocab[tok]] if tok in vocab else 0.0
        for j in (i - 1, i):
            if 0 <= j < len(tokens) - 1:
                bigram = f"{tokens[j]} {tokens[j + 1]}"
                if bigram in vocab:
                    score += coefficients[vocab[bigram]] / 2
        word_scores.append({"word": tok, "score": round(float(score), 4),
                            "known": tok in vocab})
    return features[:10], word_scores


def save_prediction(headline, prediction, p_sarc, p_not, confidence):
    exists = RESULTS_CSV.exists()
    with open(RESULTS_CSV, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if not exists:
            w.writerow(["Headline", "Prediction", "Sarcastic probability",
                        "Not sarcastic probability", "Confidence"])
        w.writerow([headline, prediction, f"{p_sarc * 100:.2f}%",
                    f"{p_not * 100:.2f}%", f"{confidence * 100:.2f}%"])


@app.get("/api/health")
def health():
    return jsonify({"status": "ok"})


@app.post("/api/predict")
def predict():
    data = request.get_json(silent=True) or {}
    headline = (data.get("headline") or "").strip()

    if not headline:
        return jsonify({"error": "Please enter a headline."}), 400
    if len(headline) > 300:
        return jsonify({"error": "Headline too long (max 300 characters)."}), 400

    cleaned = clean_text(headline)
    if not cleaned:
        return jsonify({"error": "Headline has no usable words after cleaning."}), 400

    X = vectorizer.transform([cleaned])
    p_not, p_sarc = model.predict_proba(X)[0]
    is_sarcastic = bool(model.predict(X)[0] == 1)
    confidence = max(p_not, p_sarc)
    label = "SARCASTIC" if is_sarcastic else "NOT SARCASTIC"

    top_terms, word_scores = explain(cleaned, X)
    known = sum(1 for w in word_scores if w["known"])

    save_prediction(headline, label, p_sarc, p_not, confidence)

    return jsonify({
        "headline": headline,
        "cleaned": cleaned,
        "prediction": label,
        "is_sarcastic": is_sarcastic,
        "sarcastic_probability": round(float(p_sarc) * 100, 2),
        "not_sarcastic_probability": round(float(p_not) * 100, 2),
        "confidence": round(float(confidence) * 100, 2),
        "top_terms": top_terms,
        "word_scores": word_scores,
        "known_words": known,
        "total_words": len(word_scores),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    })


@app.get("/api/top-features")
def top_features():
    n = min(int(request.args.get("n", 15)), 50)
    order = np.argsort(coefficients)
    sarcastic, normal = [], []
    for idx in order[::-1]:
        if is_clean(feature_names[idx]):
            sarcastic.append({"term": feature_names[idx], "weight": round(float(coefficients[idx]), 3)})
        if len(sarcastic) == n:
            break
    for idx in order:
        if is_clean(feature_names[idx]):
            normal.append({"term": feature_names[idx], "weight": round(float(coefficients[idx]), 3)})
        if len(normal) == n:
            break
    return jsonify({"sarcastic": sarcastic, "not_sarcastic": normal})


@app.get("/api/history")
def history():
    if not RESULTS_CSV.exists():
        return jsonify([])
    with open(RESULTS_CSV, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return jsonify(rows[-15:][::-1])


@app.delete("/api/history")
def clear_history():
    if RESULTS_CSV.exists():
        RESULTS_CSV.unlink()
    return jsonify({"status": "cleared"})


@app.get("/api/model-info")
def model_info():
    return jsonify(MODEL_INFO)


if __name__ == "__main__":
    print(f"Model loaded from {MODEL_PATH}")
    app.run(host="0.0.0.0", port=5000, debug=True)
