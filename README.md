````markdown
# Sarcasm Detection in News Headlines

A Machine Learning project that detects whether a news headline is **sarcastic** or **non-sarcastic** using Natural Language Processing (NLP), TF-IDF vectorization, and Logistic Regression.

---

## Project Overview

Sarcasm can make automated text classification difficult because the literal meaning of a sentence may differ from its intended meaning.

This project builds a machine learning model that classifies news headlines into two categories:

- Sarcastic
- Not Sarcastic

The project follows a complete machine learning workflow:

```text
Dataset
   ↓
Data Loading
   ↓
Text Preprocessing
   ↓
Train/Test Split
   ↓
TF-IDF Vectorization
   ↓
Logistic Regression
   ↓
Hyperparameter Tuning
   ↓
Final Model Training
   ↓
Model Evaluation
   ↓
Prediction
````

---

## Dataset

The project uses the **Sarcasm Headlines Dataset**.

Dataset file:

```text
Sarcasm_Headlines_Dataset.json
```

### Dataset Statistics

| Category        |  Count |
| --------------- | -----: |
| Total Headlines | 26,709 |
| Non-Sarcastic   | 14,985 |
| Sarcastic       | 11,724 |

### Dataset Columns

Each record contains:

```text
article_link
headline
is_sarcastic
```

Where:

```text
is_sarcastic = 0 → Not Sarcastic
is_sarcastic = 1 → Sarcastic
```

---

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Seaborn
* Joblib
* Regular Expressions (`re`)
* TF-IDF
* Logistic Regression
* Git
* GitHub

---

## Project Structure

```text
Sarcasm Detector/
│
├── data/
│   └── Sarcasm_Headlines_Dataset.json
│
├── models/
│   ├── sarcasm_model.pkl
│   └── tfidf_vectorizer.pkl
│
├── src/
│   ├── evaluate.py
│   ├── final_train.py
│   ├── load_dataset.py
│   ├── predict.py
│   ├── preprocess.py
│   ├── test_dataset.py
│   ├── top_features.py
│   ├── train.py
│   └── tune_model.py
│
├── .gitignore
├── README.md
└── requirements.txt
```

The generated file:

```text
prediction_results.csv
```

is created when using the prediction program and is excluded from GitHub using `.gitignore`.

---

## Text Preprocessing

The headlines are cleaned before being passed to the machine learning model.

The preprocessing steps include:

1. Convert text to lowercase
2. Remove URLs
3. Remove punctuation and special characters
4. Normalize whitespace
5. Remove unnecessary spacing

### Example

Original:

```text
Local Man Says He Will Start Diet Tomorrow!!!
```

After preprocessing:

```text
local man says he will start diet tomorrow
```

---

## Feature Extraction

TF-IDF (Term Frequency-Inverse Document Frequency) is used to convert text into numerical features.

The final vectorizer uses:

```python
TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 2),
    min_df=2
)
```

The model uses both:

* **Unigrams** — individual words
* **Bigrams** — pairs of consecutive words

For example:

```text
local
man
local man
```

Using both unigrams and bigrams allows the model to learn individual word patterns as well as short phrase patterns.

---

## Machine Learning Model

The classification model used is **Logistic Regression**.

### Final Configuration

```python
LogisticRegression(
    C=2.0,
    max_iter=1000,
    random_state=42
)
```

The dataset was divided using an **80/20 train-test split**.

```text
80% → Training
20% → Testing
```

Stratified splitting was used to preserve the class distribution between the training and testing sets.

### Dataset Split

```text
Training samples: 21,367
Testing samples:   5,342
```

---

## Hyperparameter Tuning

Hyperparameter tuning was performed to determine a suitable Logistic Regression regularization value and TF-IDF n-gram configuration.

### Tested C Values

```text
0.1
0.5
1.0
2.0
5.0
```

### Tested TF-IDF N-Gram Ranges

```text
(1, 1)
(1, 2)
```

The selected configuration was:

```text
C = 2.0
ngram_range = (1, 2)
```

---

## Model Performance

### Baseline Model

The initial model achieved:

```text
Accuracy: 84.43%
```

### Final Tuned Model

The final tuned model achieved:

```text
Accuracy: 84.91%
```

### Improvement

```text
84.43% → 84.91%
```

This represents an improvement of:

```text
+0.48 percentage points
```

The final accuracy was measured on the held-out test set.

---

## Classification Report

The final model achieved the following results on the held-out test set:

```text
                  precision    recall    f1-score

Not Sarcastic        0.86       0.88       0.87
Sarcastic            0.84       0.81       0.83

Accuracy                                  0.85
```

Test set size:

```text
5,342 headlines
```

### Interpretation

The model achieved:

* 86% precision for non-sarcastic headlines
* 88% recall for non-sarcastic headlines
* 84% precision for sarcastic headlines
* 81% recall for sarcastic headlines
* 84.91% overall accuracy

---

## Confusion Matrix

The model can also be evaluated using a confusion matrix.

The confusion matrix helps identify:

* True Positives
* True Negatives
* False Positives
* False Negatives

This provides more information about the types of classification errors made by the model.

---

## Running the Project

### 1. Clone the Repository

Clone the GitHub repository:

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

Move into the project directory:

```bash
cd "Sarcasm Detector"
```

---

### 2. Create a Virtual Environment

On Windows:

```bash
python -m venv venv
```

Activate the virtual environment:

```bash
venv\Scripts\activate
```

---

### 3. Install Dependencies

Install the required Python packages:

```bash
pip install -r requirements.txt
```

---

## Training the Model

To train the final model:

```bash
python src/final_train.py
```

This creates the trained model and TF-IDF vectorizer:

```text
models/sarcasm_model.pkl
models/tfidf_vectorizer.pkl
```

---

## Evaluating the Model

Run:

```bash
python src/evaluate.py
```

This evaluates the trained model and displays metrics such as:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix

---

## Hyperparameter Tuning

To run the hyperparameter tuning process:

```bash
python src/tune_model.py
```

This tests different values of `C` and different TF-IDF n-gram configurations.

---

## Viewing Important Features

To view the features that have the strongest statistical association with the model's predictions:

```bash
python src/top_features.py
```

The script displays important words and phrases associated with both sarcastic and non-sarcastic classifications.

---

## Making Predictions

To use the trained model interactively:

```bash
python src/predict.py
```

The program allows the user to enter a news headline.

Example:

```text
Headline: Local man spends three hours looking for his phone while holding it
```

Example output:

```text
Prediction              : SARCASTIC

Sarcastic probability   : 98.63%
Not sarcastic probability: 1.37%

Confidence              : 98.63%
```

The program can continue accepting headlines until:

```text
exit
```

is entered.

---

## Prediction Results

The prediction program records each manual prediction in:

```text
prediction_results.csv
```

Each prediction contains:

```text
Headline
Prediction
Sarcastic probability
Not sarcastic probability
Confidence
```

Example:

| Headline                                   | Prediction    | Sarcastic Probability | Not Sarcastic Probability | Confidence |
| ------------------------------------------ | ------------- | --------------------: | ------------------------: | ---------: |
| Local man finds his phone while holding it | SARCASTIC     |                98.63% |                     1.37% |     98.63% |
| The company reported increased revenue     | NOT SARCASTIC |                42.54% |                    57.46% |     57.46% |

These examples are **manual demonstration predictions** and are not part of the official test-set evaluation.

The `prediction_results.csv` file is generated locally and is excluded from the GitHub repository through `.gitignore`.

---

## Model Output

For each headline, the prediction system provides:

```text
Headline
Prediction
Sarcastic probability
Not sarcastic probability
Confidence
```

The probabilities are the model's estimated class probabilities. They should not be interpreted as guaranteed human-level certainty.

---

## Important Source Files

### `load_dataset.py`

Loads and inspects the JSON dataset.

### `preprocess.py`

Contains the text preprocessing functionality.

### `train.py`

Contains the baseline model training workflow.

### `tune_model.py`

Tests different hyperparameter configurations and identifies a suitable configuration.

### `final_train.py`

Trains the final Logistic Regression model using the selected hyperparameters.

### `evaluate.py`

Evaluates the trained model using classification metrics and a confusion matrix.

### `top_features.py`

Displays important features learned by the Logistic Regression model.

### `predict.py`

Provides an interactive interface for making predictions on new headlines.

### `test_dataset.py`

Performs basic dataset loading and inspection.

---

## Limitations

This project has several limitations:

* Sarcasm is highly dependent on context.
* News headlines can be ambiguous.
* The model learns statistical patterns from the training dataset.
* Real-world sarcasm may differ from the patterns present in the dataset.
* A headline can be interpreted differently by different people.
* Model probabilities do not guarantee human-level certainty.
* The dataset may contain biases or source-specific patterns.
* TF-IDF and Logistic Regression do not understand language in the same way as modern language models.

Therefore, this model should be considered a machine learning text classification system rather than a perfect sarcasm detector.

---

## Future Improvements

Possible future improvements include:

* Word2Vec embeddings
* GloVe embeddings
* Transformer-based models
* BERT fine-tuning
* Context-aware sarcasm detection
* Sentiment analysis
* Probability calibration
* More advanced NLP preprocessing
* Web-based user interface
* REST API deployment
* Real-time sarcasm detection
* Model comparison with other machine learning algorithms

---

## Project Status

```text
Dataset                ✓
Data preprocessing     ✓
Baseline model         ✓
Hyperparameter tuning  ✓
Final model            ✓
Model evaluation       ✓
Prediction system      ✓
Prediction logging     ✓
GitHub preparation     ✓
```

---

## Author

**Soham Shimpi**

Computer Science Engineering

Machine Learning / NLP Project
