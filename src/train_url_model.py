import os
import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
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


INPUT_FILE = "data/url_features.csv"

MODEL_DIR = "models"

MODEL_FILE = os.path.join(
    MODEL_DIR,
    "url_random_forest.joblib"
)


FEATURE_COLUMNS = [
    "url_length",
    "dot_count",
    "has_https",
    "has_ip_address",
    "special_chars",
    "subdomain_count",
    "path_depth",
    "digit_ratio",
    "has_at_symbol",
    "domain_length",
]


def main():

    # Load dataset
    df = pd.read_csv(INPUT_FILE)

    print("Dataset shape:", df.shape)

    # Convert labels
    df["label"] = df["status"].map(
        {
            "legitimate": 0,
            "phishing": 1,
        }
    )

    if df["label"].isnull().any():
        raise ValueError(
            "Some status values could not be converted to labels."
        )

    # Features and target
    X = df[FEATURE_COLUMNS]
    y = df["label"]

    print("\nFeatures:", X.shape)

    print("\nLabel distribution:")
    print(y.value_counts())

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    print("\nTraining samples:", len(X_train))
    print("Testing samples:", len(X_test))

    # Random Forest
    print("\nTraining URL Random Forest...")

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_leaf=1,
        max_features="sqrt",
        class_weight="balanced",
        n_jobs=2,
        random_state=42,
    )

    model.fit(X_train, y_train)

    print("Training completed.")

    # Predictions
    y_pred = model.predict(X_test)

    y_probability = model.predict_proba(X_test)[:, 1]

    # Evaluation
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
    print("URL MODEL — RANDOM FOREST")
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

    # Feature importance
    importance_df = pd.DataFrame(
        {
            "feature": FEATURE_COLUMNS,
            "importance": model.feature_importances_,
        }
    ).sort_values(
        by="importance",
        ascending=False,
    )

    print("\nFeature Importance:")

    print(
        importance_df.to_string(
            index=False
        )
    )

    # Save model
    os.makedirs(
        MODEL_DIR,
        exist_ok=True,
    )

    joblib.dump(
        model,
        MODEL_FILE,
    )

    print("\nSaved model:")
    print(MODEL_FILE)


if __name__ == "__main__":
    main()
