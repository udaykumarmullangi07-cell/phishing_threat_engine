import os
import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)

from xgboost import XGBClassifier


INPUT_FILE = "data/cleaned_text.csv"
MODEL_DIR = "models"

MODEL_FILE = os.path.join(
    MODEL_DIR,
    "text_xgboost.joblib"
)

VECTORIZER_FILE = os.path.join(
    MODEL_DIR,
    "text_xgb_tfidf_vectorizer.joblib"
)


def main():

    df = pd.read_csv(INPUT_FILE)

    df["processed_text"] = df["processed_text"].fillna("")

    X = df["processed_text"]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    print("Creating TF-IDF representation...")

    vectorizer = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
    )

    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    print("Train shape:", X_train_tfidf.shape)
    print("Test shape:", X_test_tfidf.shape)

    print("\nTraining XGBoost...")

    model = XGBClassifier(
        n_estimators=150,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        n_jobs=2,
        random_state=42,
    )

    model.fit(X_train_tfidf, y_train)

    print("Training completed.")

    y_pred = model.predict(X_test_tfidf)
    y_probability = model.predict_proba(X_test_tfidf)[:, 1]

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
    print("TEXT MODEL — XGBOOST")
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

    print(
        confusion_matrix(
            y_test,
            y_pred,
        )
    )

    os.makedirs(
        MODEL_DIR,
        exist_ok=True,
    )

    joblib.dump(
        model,
        MODEL_FILE,
    )

    joblib.dump(
        vectorizer,
        VECTORIZER_FILE,
    )

    print("\nSaved model:", MODEL_FILE)
    print("Saved vectorizer:", VECTORIZER_FILE)


if __name__ == "__main__":
    main()
