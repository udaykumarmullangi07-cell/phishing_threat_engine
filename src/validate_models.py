import pandas as pd
import joblib

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)


# =========================================================
# TEXT MODEL CONFIGURATION
# =========================================================

TEXT_DATA = "data/cleaned_text.csv"

TEXT_MODEL = "models/text_logistic_regression.joblib"

TEXT_VECTORIZER = (
    "models/text_tfidf_vectorizer.joblib"
)


# =========================================================
# URL MODEL CONFIGURATION
# =========================================================

URL_DATA = "data/url_features.csv"

URL_MODEL = "models/url_random_forest.joblib"

URL_FEATURE_COLUMNS = [
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


# =========================================================
# TEXT MODEL VALIDATION
# =========================================================

def validate_text_model():

    print("\n" + "=" * 70)
    print("TEXT MODEL VALIDATION")
    print("=" * 70)

    # -----------------------------------------------------
    # Load dataset
    # -----------------------------------------------------

    df = pd.read_csv(
        TEXT_DATA
    )

    # Remove rows where the model input or label is missing
    df = df.dropna(
        subset=[
            "processed_text",
            "label",
        ]
    )

    print(
        f"Validation samples: {len(df)}"
    )

    # -----------------------------------------------------
    # Load trained model
    # -----------------------------------------------------

    model = joblib.load(
        TEXT_MODEL
    )

    # -----------------------------------------------------
    # Load TF-IDF vectorizer
    # -----------------------------------------------------

    vectorizer = joblib.load(
        TEXT_VECTORIZER
    )

    # -----------------------------------------------------
    # Create TF-IDF representation
    # -----------------------------------------------------

    X = vectorizer.transform(
        df["processed_text"]
    )

    y = df["label"]

    # -----------------------------------------------------
    # Generate predictions
    # -----------------------------------------------------

    predictions = model.predict(
        X
    )

    probabilities = model.predict_proba(
        X
    )[:, 1]

    # -----------------------------------------------------
    # Calculate metrics
    # -----------------------------------------------------

    accuracy = accuracy_score(
        y,
        predictions
    )

    precision = precision_score(
        y,
        predictions
    )

    recall = recall_score(
        y,
        predictions
    )

    f1 = f1_score(
        y,
        predictions
    )

    roc_auc = roc_auc_score(
        y,
        probabilities
    )

    # -----------------------------------------------------
    # Display metrics
    # -----------------------------------------------------

    print(
        f"\nAccuracy : {accuracy:.4f}"
    )

    print(
        f"Precision: {precision:.4f}"
    )

    print(
        f"Recall   : {recall:.4f}"
    )

    print(
        f"F1 Score : {f1:.4f}"
    )

    print(
        f"ROC-AUC  : {roc_auc:.4f}"
    )

    # -----------------------------------------------------
    # Confusion matrix
    # -----------------------------------------------------

    print("\nConfusion Matrix:")

    print(
        confusion_matrix(
            y,
            predictions
        )
    )

    # -----------------------------------------------------
    # Classification report
    # -----------------------------------------------------

    print("\nClassification Report:")

    print(
        classification_report(
            y,
            predictions,
            target_names=[
                "Legitimate",
                "Phishing",
            ],
        )
    )


# =========================================================
# URL MODEL VALIDATION
# =========================================================

def validate_url_model():

    print("\n" + "=" * 70)
    print("URL MODEL VALIDATION")
    print("=" * 70)

    # -----------------------------------------------------
    # Load URL dataset
    # -----------------------------------------------------

    df = pd.read_csv(
        URL_DATA
    )

    print(
        f"Validation samples: {len(df)}"
    )

    # -----------------------------------------------------
    # Convert status to numeric label
    #
    # legitimate = 0
    # phishing   = 1
    # -----------------------------------------------------

    df["label"] = (
        df["status"] == "phishing"
    ).astype(int)

    # -----------------------------------------------------
    # Select exactly the same features
    # used during URL model training
    # -----------------------------------------------------

    X = df[
        URL_FEATURE_COLUMNS
    ]

    y = df["label"]

    # -----------------------------------------------------
    # Load trained URL model
    # -----------------------------------------------------

    model = joblib.load(
        URL_MODEL
    )

    # -----------------------------------------------------
    # Generate predictions
    # -----------------------------------------------------

    predictions = model.predict(
        X
    )

    probabilities = model.predict_proba(
        X
    )[:, 1]

    # -----------------------------------------------------
    # Calculate metrics
    # -----------------------------------------------------

    accuracy = accuracy_score(
        y,
        predictions
    )

    precision = precision_score(
        y,
        predictions
    )

    recall = recall_score(
        y,
        predictions
    )

    f1 = f1_score(
        y,
        predictions
    )

    roc_auc = roc_auc_score(
        y,
        probabilities
    )

    # -----------------------------------------------------
    # Display metrics
    # -----------------------------------------------------

    print(
        f"\nAccuracy : {accuracy:.4f}"
    )

    print(
        f"Precision: {precision:.4f}"
    )

    print(
        f"Recall   : {recall:.4f}"
    )

    print(
        f"F1 Score : {f1:.4f}"
    )

    print(
        f"ROC-AUC  : {roc_auc:.4f}"
    )

    # -----------------------------------------------------
    # Confusion matrix
    # -----------------------------------------------------

    print("\nConfusion Matrix:")

    print(
        confusion_matrix(
            y,
            predictions
        )
    )

    # -----------------------------------------------------
    # Classification report
    # -----------------------------------------------------

    print("\nClassification Report:")

    print(
        classification_report(
            y,
            predictions,
            target_names=[
                "Legitimate",
                "Phishing",
            ],
        )
    )


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    # Validate text model
    validate_text_model()

    # Validate URL model
    validate_url_model()
