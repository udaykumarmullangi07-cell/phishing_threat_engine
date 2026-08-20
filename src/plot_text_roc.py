import os

import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score


# =========================================================
# FILE PATHS
# =========================================================

DATA_FILE = "data/cleaned_text.csv"

MODEL_FILE = (
    "models/text_logistic_regression.joblib"
)

VECTORIZER_FILE = (
    "models/text_tfidf_vectorizer.joblib"
)

REPORT_DIR = "reports"

ROC_CURVE_FILE = (
    "reports/text_roc_curve.png"
)


# =========================================================
# CONFIGURATION
# =========================================================

TEST_SIZE = 0.20
RANDOM_STATE = 42


# =========================================================
# MAIN
# =========================================================

def main():

    print("=" * 70)
    print("TEXT MODEL — ROC CURVE ANALYSIS")
    print("=" * 70)


    # =====================================================
    # LOAD DATASET
    # =====================================================

    print("\nLoading dataset...")

    df = pd.read_csv(
        DATA_FILE
    )

    print(
        f"Dataset shape: {df.shape}"
    )


    # =====================================================
    # VALIDATE REQUIRED COLUMNS
    # =====================================================

    required_columns = [
        "processed_text",
        "label",
    ]

    for column in required_columns:

        if column not in df.columns:

            raise ValueError(
                f"Required column '{column}' "
                f"was not found in dataset."
            )


    # =====================================================
    # REMOVE INVALID ROWS
    # =====================================================

    df = df[
        df["processed_text"].notna()
        & df["label"].notna()
    ].copy()


    print(
        f"Valid samples: {len(df)}"
    )


    # =====================================================
    # PREPARE DATA
    # =====================================================

    X = (
        df["processed_text"]
        .astype(str)
    )

    y = (
        df["label"]
        .astype(int)
    )


    # =====================================================
    # TRAIN / TEST SPLIT
    #
    # MUST MATCH train_text_model.py
    # =====================================================

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
        )
    )


    print(
        f"\nTraining samples: {len(X_train)}"
    )

    print(
        f"Testing samples:  {len(X_test)}"
    )


    # =====================================================
    # LOAD TRAINED TF-IDF VECTORIZER
    # =====================================================

    print(
        "\nLoading TF-IDF vectorizer..."
    )

    vectorizer = joblib.load(
        VECTORIZER_FILE
    )


    # =====================================================
    # TRANSFORM TEST DATA
    #
    # IMPORTANT:
    # We do NOT fit the vectorizer again.
    # The saved vectorizer already contains the vocabulary
    # learned during training.
    # =====================================================

    print(
        "Creating TF-IDF test representation..."
    )

    X_test_tfidf = vectorizer.transform(
        X_test
    )


    print(
        f"Test TF-IDF shape: "
        f"{X_test_tfidf.shape}"
    )


    # =====================================================
    # LOAD TRAINED MODEL
    # =====================================================

    print(
        "\nLoading Logistic Regression model..."
    )

    model = joblib.load(
        MODEL_FILE
    )


    # =====================================================
    # PREDICT TEST PROBABILITIES
    # =====================================================

    print(
        "\nCalculating phishing probabilities..."
    )

    y_probability = model.predict_proba(
        X_test_tfidf
    )[:, 1]


    # =====================================================
    # CALCULATE ROC-AUC
    # =====================================================

    roc_auc = roc_auc_score(
        y_test,
        y_probability,
    )


    print(
        f"\nROC-AUC: {roc_auc:.4f}"
    )


    # =====================================================
    # CALCULATE ROC CURVE
    # =====================================================

    false_positive_rate, true_positive_rate, thresholds = (
        roc_curve(
            y_test,
            y_probability,
        )
    )


    print(
        f"ROC points: {len(false_positive_rate)}"
    )


    # =====================================================
    # CREATE REPORT DIRECTORY
    # =====================================================

    os.makedirs(
        REPORT_DIR,
        exist_ok=True,
    )


    # =====================================================
    # CREATE ROC PLOT
    # =====================================================

    plt.figure(
        figsize=(8, 6)
    )


    # Model ROC curve

    plt.plot(
        false_positive_rate,
        true_positive_rate,
        linewidth=2,
        label=(
            "Logistic Regression "
            f"(AUC = {roc_auc:.4f})"
        ),
    )


    # Random classifier baseline

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        linewidth=1,
        label="Random Classifier",
    )


    # Labels

    plt.xlabel(
        "False Positive Rate"
    )

    plt.ylabel(
        "True Positive Rate"
    )

    plt.title(
        "ROC Curve — Text Phishing Detection"
    )


    # Legend

    plt.legend(
        loc="lower right"
    )


    # Grid

    plt.grid(
        True,
        alpha=0.3,
    )


    # Layout

    plt.tight_layout()


    # =====================================================
    # SAVE CHART
    # =====================================================

    plt.savefig(
        ROC_CURVE_FILE,
        dpi=300,
        bbox_inches="tight",
    )


    plt.close()


    print(
        f"\nROC curve saved to:"
    )

    print(
        ROC_CURVE_FILE
    )


    # =====================================================
    # FINAL SUMMARY
    # =====================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "ROC CURVE ANALYSIS COMPLETE"
    )

    print(
        "=" * 70
    )

    print(
        f"Dataset samples : {len(df)}"
    )

    print(
        f"Test samples    : {len(y_test)}"
    )

    print(
        f"ROC-AUC         : {roc_auc:.4f}"
    )

    print(
        f"ROC curve       : {ROC_CURVE_FILE}"
    )

    print(
        "=" * 70
    )


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":

    main()
