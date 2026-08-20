import os

import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


# =========================================================
# FILES
# =========================================================

INPUT_FILE = "data/cleaned_text.csv"

MODEL_FILE = (
    "models/final_text_model.joblib"
)

VECTORIZER_FILE = (
    "models/final_text_vectorizer.joblib"
)

REPORT_FILE = (
    "reports/text_threshold_analysis.csv"
)


# =========================================================
# CONFIGURATION
# =========================================================

TEST_SIZE = 0.20
RANDOM_STATE = 42


# =========================================================
# THRESHOLDS
# =========================================================

THRESHOLDS = [
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
    0.75,
    0.80,
    0.85,
    0.90,
]


# =========================================================
# MAIN
# =========================================================

def main():

    print("=" * 70)
    print("TEXT MODEL — DECISION THRESHOLD ANALYSIS")
    print("=" * 70)


    # =====================================================
    # LOAD DATASET
    # =====================================================

    print("\nLoading dataset...")

    df = pd.read_csv(
        INPUT_FILE
    )

    df["processed_text"] = (
        df["processed_text"]
        .fillna("")
    )

    X = df["processed_text"]
    y = df["label"]


    print(
        f"Dataset samples: {len(df)}"
    )


    # =====================================================
    # SAME TRAIN/TEST SPLIT
    # =====================================================

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=y,
        )
    )


    print(
        f"Training samples: {len(X_train)}"
    )

    print(
        f"Testing samples:  {len(X_test)}"
    )


    # =====================================================
    # LOAD FINAL MODEL
    # =====================================================

    print(
        "\nLoading final model..."
    )

    model = joblib.load(
        MODEL_FILE
    )

    vectorizer = joblib.load(
        VECTORIZER_FILE
    )


    print(
        "Model:",
        type(model).__name__,
    )

    print(
        "C:",
        model.C,
    )

    print(
        "TF-IDF features:",
        len(
            vectorizer
            .get_feature_names_out()
        ),
    )


    # =====================================================
    # CREATE TEST FEATURES
    # =====================================================

    print(
        "\nCreating TF-IDF test representation..."
    )

    X_test_tfidf = (
        vectorizer.transform(
            X_test
        )
    )


    print(
        "Test TF-IDF shape:",
        X_test_tfidf.shape,
    )


    # =====================================================
    # PHISHING PROBABILITIES
    # =====================================================

    print(
        "\nCalculating phishing probabilities..."
    )

    probabilities = (
        model
        .predict_proba(
            X_test_tfidf
        )[:, 1]
    )


    # =====================================================
    # ROC-AUC
    # =====================================================

    roc_auc = roc_auc_score(
        y_test,
        probabilities,
    )


    print(
        f"\nROC-AUC: {roc_auc:.4f}"
    )


    # =====================================================
    # THRESHOLD ANALYSIS
    # =====================================================

    results = []


    print(
        "\n" + "=" * 70
    )

    print(
        "THRESHOLD RESULTS"
    )

    print(
        "=" * 70
    )


    header = (
        f"{'Threshold':>10}"
        f"{'Accuracy':>12}"
        f"{'Precision':>12}"
        f"{'Recall':>12}"
        f"{'F1':>12}"
        f"{'FP':>8}"
        f"{'FN':>8}"
    )

    print(
        "\n" + header
    )

    print(
        "-" * len(header)
    )


    for threshold in THRESHOLDS:

        # -------------------------------------------------
        # Convert probabilities into classes
        # -------------------------------------------------

        predictions = (
            probabilities
            >= threshold
        ).astype(int)


        # -------------------------------------------------
        # Metrics
        # -------------------------------------------------

        accuracy = (
            accuracy_score(
                y_test,
                predictions,
            )
        )

        precision = (
            precision_score(
                y_test,
                predictions,
                zero_division=0,
            )
        )

        recall = (
            recall_score(
                y_test,
                predictions,
                zero_division=0,
            )
        )

        f1 = (
            f1_score(
                y_test,
                predictions,
                zero_division=0,
            )
        )


        # -------------------------------------------------
        # Confusion matrix
        #
        # [[TN FP]
        #  [FN TP]]
        # -------------------------------------------------

        tn, fp, fn, tp = (
            confusion_matrix(
                y_test,
                predictions,
                labels=[0, 1],
            ).ravel()
        )


        print(
            f"{threshold:10.2f}"
            f"{accuracy:12.4f}"
            f"{precision:12.4f}"
            f"{recall:12.4f}"
            f"{f1:12.4f}"
            f"{fp:8d}"
            f"{fn:8d}"
        )


        results.append({

            "threshold":
                threshold,

            "accuracy":
                accuracy,

            "precision":
                precision,

            "recall":
                recall,

            "f1":
                f1,

            "false_positives":
                int(fp),

            "false_negatives":
                int(fn),

            "true_negatives":
                int(tn),

            "true_positives":
                int(tp),

            "roc_auc":
                roc_auc,

        })


    # =====================================================
    # DATAFRAME
    # =====================================================

    results_df = pd.DataFrame(
        results
    )


    # =====================================================
    # BEST F1
    # =====================================================

    best_f1_row = (
        results_df
        .loc[
            results_df["f1"].idxmax()
        ]
    )


    # =====================================================
    # BEST PRECISION
    # =====================================================

    best_precision_row = (
        results_df
        .loc[
            results_df["precision"].idxmax()
        ]
    )


    # =====================================================
    # BEST RECALL
    # =====================================================

    best_recall_row = (
        results_df
        .loc[
            results_df["recall"].idxmax()
        ]
    )


    # =====================================================
    # PRINT INTERPRETATION
    # =====================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "BEST THRESHOLDS"
    )

    print(
        "=" * 70
    )


    print(
        "\nBest F1:"
    )

    print(
        f"Threshold : "
        f"{best_f1_row['threshold']:.2f}"
    )

    print(
        f"F1        : "
        f"{best_f1_row['f1']:.4f}"
    )

    print(
        f"Precision : "
        f"{best_f1_row['precision']:.4f}"
    )

    print(
        f"Recall    : "
        f"{best_f1_row['recall']:.4f}"
    )

    print(
        f"FP        : "
        f"{int(best_f1_row['false_positives'])}"
    )

    print(
        f"FN        : "
        f"{int(best_f1_row['false_negatives'])}"
    )


    print(
        "\nBest Precision:"
    )

    print(
        f"Threshold : "
        f"{best_precision_row['threshold']:.2f}"
    )

    print(
        f"Precision : "
        f"{best_precision_row['precision']:.4f}"
    )

    print(
        f"Recall    : "
        f"{best_precision_row['recall']:.4f}"
    )


    print(
        "\nBest Recall:"
    )

    print(
        f"Threshold : "
        f"{best_recall_row['threshold']:.2f}"
    )

    print(
        f"Recall    : "
        f"{best_recall_row['recall']:.4f}"
    )

    print(
        f"Precision : "
        f"{best_recall_row['precision']:.4f}"
    )


    # =====================================================
    # SAVE REPORT
    # =====================================================

    os.makedirs(
        "reports",
        exist_ok=True,
    )

    results_df.to_csv(
        REPORT_FILE,
        index=False,
    )


    print(
        "\n" + "=" * 70
    )

    print(
        "THRESHOLD ANALYSIS COMPLETE"
    )

    print(
        "=" * 70
    )

    print(
        "\nReport saved to:"
    )

    print(
        REPORT_FILE
    )


if __name__ == "__main__":

    main()
