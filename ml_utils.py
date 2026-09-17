from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile
from io import BytesIO

import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline

DATA_PATH = Path("data/raw/SMSSpamCollection")
MODEL_PATH = Path("models/spam_classifier.joblib")
METRICS_PATH = Path("reports/metrics.csv")
REPORT_PATH = Path("reports/classification_report.txt")

DATASET_URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"


def download_dataset():
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)

    if DATA_PATH.exists():
        return

    with urlopen(DATASET_URL, timeout=60) as response:
        archive = response.read()

    with ZipFile(BytesIO(archive)) as zf:
        DATA_PATH.write_bytes(zf.read("SMSSpamCollection"))


def load_data():
    download_dataset()

    df = pd.read_csv(
        DATA_PATH,
        sep="\t",
        header=None,
        names=["label", "message"],
        encoding="utf-8",
    )

    df = df.dropna()
    df["label"] = df["label"].str.strip().str.lower()
    df["message"] = df["message"].astype(str).str.strip()
    return df


def build_pipeline():
    features = FeatureUnion([
        (
            "word_tfidf",
            TfidfVectorizer(
                lowercase=True,
                strip_accents="unicode",
                ngram_range=(1, 2),
                min_df=2,
                sublinear_tf=True,
                max_features=50000,
            ),
        ),
        (
            "char_tfidf",
            TfidfVectorizer(
                analyzer="char",
                ngram_range=(3, 5),
                min_df=2,
                sublinear_tf=True,
                max_features=50000,
            ),
        ),
    ])

    return Pipeline([
        ("features", features),
        (
            "classifier",
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=42,
            ),
        ),
    ])


def train_and_save():
    df = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        df["message"],
        df["label"],
        test_size=0.20,
        random_state=42,
        stratify=df["label"],
    )

    model = build_pipeline()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, pos_label="spam"),
        "recall": recall_score(y_test, predictions, pos_label="spam"),
        "f1_score": f1_score(y_test, predictions, pos_label="spam"),
    }

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    Path("reports").mkdir(exist_ok=True)

    joblib.dump(model, MODEL_PATH)

    pd.DataFrame(
        [{"metric": key, "score": value} for key, value in metrics.items()]
    ).to_csv(METRICS_PATH, index=False)

    REPORT_PATH.write_text(
        classification_report(y_test, predictions),
        encoding="utf-8",
    )

    return model


def ensure_model():
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)

    return train_and_save()
