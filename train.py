from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

from ml_utils import load_data, build_pipeline, MODEL_PATH

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import train_test_split
import joblib

REPORT_DIR = Path("reports")
FIGURE_DIR = REPORT_DIR / "figures"


def main():
    REPORT_DIR.mkdir(exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data()
    print(f"Loaded {len(df):,} messages.")
    print(df["label"].value_counts())

    counts = df["label"].value_counts().reindex(["ham", "spam"]).fillna(0)
    ax = counts.plot(kind="bar", figsize=(7, 4))
    ax.set_title("SMS Dataset Class Distribution")
    ax.set_xlabel("Class")
    ax.set_ylabel("Number of messages")
    ax.tick_params(axis="x", rotation=0)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "class_distribution.png", dpi=160)
    plt.close()

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

    pd.DataFrame(
        [{"metric": k, "score": v} for k, v in metrics.items()]
    ).to_csv(REPORT_DIR / "metrics.csv", index=False)

    (REPORT_DIR / "classification_report.txt").write_text(
        classification_report(y_test, predictions),
        encoding="utf-8",
    )

    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(
        y_test,
        predictions,
        labels=["ham", "spam"],
        cmap="Blues",
        ax=ax,
    )
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "confusion_matrix.png", dpi=160)
    plt.close()

    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    print("\nEvaluation:")
    for name, score in metrics.items():
        print(f"{name:>10}: {score:.4f}")

    print(f"\nModel saved to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
