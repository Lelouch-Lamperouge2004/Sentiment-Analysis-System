import os
import re
import pandas as pd
import nltk
import joblib

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression


DATASET_PATH = "data/sentiment_dataset.csv"
MODEL_PATH = "saved_models/linear_svc_model.joblib"
VECTORIZER_PATH = "saved_models/count_vectorizer.joblib"


try:
    STOP_WORDS = set(stopwords.words("english"))
except LookupError:
    nltk.download("stopwords")
    STOP_WORDS = set(stopwords.words("english"))

stemmer = PorterStemmer()


def preprocess_text(text):
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    tokens = text.split()

    tokens = [
        stemmer.stem(word)
        for word in tokens
        if word not in STOP_WORDS
    ]

    return " ".join(tokens)


def load_dataset():
    df = pd.read_csv(DATASET_PATH)
    df = df[["verified_reviews", "feedback"]]
    df.dropna(inplace=True)
    df["clean_text"] = df["verified_reviews"].apply(preprocess_text)
    return df


def prepare_features():
    df = load_dataset()

    X = df["clean_text"]
    y = df["feedback"]

    vectorizer = CountVectorizer(max_features=5000)
    X_vec = vectorizer.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_vec,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    return X_train, X_test, y_train, y_test, vectorizer


def get_models():
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced"
        ),
        "Linear SVC": LinearSVC(
            class_weight="balanced",
            max_iter=5000
        )
    }


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    report = classification_report(
        y_test,
        y_pred,
        output_dict=True,
        zero_division=0
    )

    cm = confusion_matrix(y_test, y_pred)

    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1 Score": f1_score(y_test, y_pred, zero_division=0),
        "Negative F1": report["0"]["f1-score"],
        "Positive F1": report["1"]["f1-score"]
    }

    return metrics, report, cm


def train_production_model():
    X_train, X_test, y_train, y_test, vectorizer = prepare_features()

    model = LinearSVC(
        class_weight="balanced",
        max_iter=5000
    )

    model.fit(X_train, y_train)

    os.makedirs("saved_models", exist_ok=True)

    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)

    metrics, report, cm = evaluate_model(model, X_test, y_test)

    return metrics, report, cm


def load_saved_model():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
        train_production_model()

    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)

    return model, vectorizer


def predict_sentiment(user_text):
    model, vectorizer = load_saved_model()

    cleaned_text = preprocess_text(user_text)
    user_vec = vectorizer.transform([cleaned_text])

    prediction = model.predict(user_vec)[0]
    score = model.decision_function(user_vec)[0]

    sentiment = "Positive" if prediction == 1 else "Negative"

    confidence = min(abs(score), 1.0) * 100
    confidence = round(confidence, 2)

    return {
        "sentiment": sentiment,
        "confidence": confidence,
        "processed_text": cleaned_text
    }


def batch_predict(reviews):
    results = []

    for review in reviews:
        prediction = predict_sentiment(review)

        results.append({
            "review": review,
            "sentiment": prediction["sentiment"],
            "confidence": prediction["confidence"]
        })

    return pd.DataFrame(results)


def compare_models():
    X_train, X_test, y_train, y_test, vectorizer = prepare_features()

    models = get_models()
    results = []

    for model_name, model in models.items():
        model.fit(X_train, y_train)

        metrics, report, cm = evaluate_model(model, X_test, y_test)

        results.append({
            "Model": model_name,
            "Accuracy": round(metrics["Accuracy"], 4),
            "Precision": round(metrics["Precision"], 4),
            "Recall": round(metrics["Recall"], 4),
            "F1 Score": round(metrics["F1 Score"], 4),
            "Negative F1": round(metrics["Negative F1"], 4),
            "Positive F1": round(metrics["Positive F1"], 4)
        })

    return pd.DataFrame(results)


def get_production_metrics():
    X_train, X_test, y_train, y_test, vectorizer = prepare_features()

    model, saved_vectorizer = load_saved_model()

    metrics, report, cm = evaluate_model(model, X_test, y_test)

    return metrics, report, cm