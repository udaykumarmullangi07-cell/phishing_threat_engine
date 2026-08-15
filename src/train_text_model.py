import os

import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


INPUT_FILE = "data/cleaned_text.csv"

MODEL_DIR = "models"
MODEL_FILE = os.path.join(MODEL_DIR, "text_logistic_regression.joblib")
VECTORIZER_FILE = os.path.join(MODEL_DIR, "text_tfidf_vectorizer.joblib")


def main():

    # ---------------------------------------------------------
    # 1. Load dataset
    # ---------------------------------------------------------

    df = pd.read_csv(INPUT_FILE)

    print("Dataset:", df.shape)

    # Handle missing processed text
    df["processed_text"] = df["processed_text"].fillna("")

    X = df["processed_text"]
    y = df["label"]

    # ---------------------------------------------------------
    # 2. Train/test split
    # ---------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    print("\nTraining samples:", len(X_train))
    print("Testing samples:", len(X_test))

    # ---------------------------------------------------------
    # 3. TF-IDF
    # ---------------------------------------------------------

    vectorizer = TfidfVectorizer(
        max_features=50000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
    )

    print("\nCreating TF-IDF matrices...")

    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    print("X_train TF-IDF:", X_train_tfidf.shape)
    print("X_test TF-IDF:", X_test_tfidf.shape)

    # ---------------------------------------------------------
    # 4. Logistic Regression
    # ---------------------------------------------------------

    print("\nTraining Logistic Regression...")

    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        solver="liblinear",
        random_state=42,
    )

    model.fit(X_train_tfidf, y_train)

    print("Training completed.")

    # ---------------------------------------------------------
    # 5. Predictions
    # ---------------------------------------------------------

    y_pred = model.predict(X_test_tfidf)

    # Probability of phishing class = label 1
    y_probability = model.predict_proba(X_test_tfidf)[:, 1]

    # ---------------------------------------------------------
    # 6. Evaluation
    # ---------------------------------------------------------

    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(
        y_test,
        y_pred,
        pos_label=1,
    )

    recall = recall_score(
        y_test,
        y_pred,
        pos_label=1,
    )

    f1 = f1_score(
        y_test,
        y_pred,
        pos_label=1,
    )

    roc_auc = roc_auc_score(
        y_test,
        y_probability,
    )

    print("\n" + "=" * 60)
    print("TEXT MODEL — LOGISTIC REGRESSION")
    print("=" * 60)

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"ROC-AUC  : {roc_auc:.4f}")

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            y_pred,
            target_names=[
                "Legitimate",
                "Phishing",
            ],
        )
    )

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # ---------------------------------------------------------
    # 7. Save model and vectorizer
    # ---------------------------------------------------------

    os.makedirs(MODEL_DIR, exist_ok=True)

    joblib.dump(model, MODEL_FILE)
    joblib.dump(vectorizer, VECTORIZER_FILE)

    print("\nSaved model:")
    print(MODEL_FILE)

    print("Saved vectorizer:")
    print(VECTORIZER_FILE)


if __name__ == "__main__":
    main()
