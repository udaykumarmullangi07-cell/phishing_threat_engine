import os

import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

from sklearn.model_selection import train_test_split


# =========================================================
# CONFIGURATION
# =========================================================

INPUT_FILE = "data/cleaned_text.csv"

REPORT_FILE = (
    "reports/text_regularization_experiment.csv"
)

RANDOM_STATE = 42
TEST_SIZE = 0.20

C_VALUES = [
    30,
    50,
    75,
    100,
]


# =========================================================
# LOAD DATA
# =========================================================

def load_data():

    df = pd.read_csv(
        INPUT_FILE
    )

    df["processed_text"] = (
        df["processed_text"]
        .fillna("")
    )

    X = df["processed_text"]
    y = df["label"]

    return X, y


# =========================================================
# MAIN EXPERIMENT
# =========================================================

def main():

    print("=" * 70)
    print(
        "TEXT MODEL — REGULARIZATION EXPERIMENT"
    )
    print("=" * 70)

    print(
        "\nThis experiment tests alternative"
        " Logistic Regression C values."
    )

    print(
        "The existing final model will NOT be modified."
    )


    # =====================================================
    # DATA
    # =====================================================

    print(
        "\nLoading dataset..."
    )

    X, y = load_data()

    print(
        "Dataset samples:",
        len(X),
    )


    # =====================================================
    # SAME SPLIT AS PREVIOUS EXPERIMENTS
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
        "Training samples:",
        len(X_train),
    )

    print(
        "Testing samples: ",
        len(X_test),
    )


    # =====================================================
    # TF-IDF
    # =====================================================

    print(
        "\nCreating TF-IDF matrices..."
    )

    vectorizer = TfidfVectorizer(
        max_features=50000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
    )

    X_train_tfidf = (
        vectorizer.fit_transform(
            X_train
        )
    )

    X_test_tfidf = (
        vectorizer.transform(
            X_test
        )
    )

    print(
        "Training TF-IDF:",
        X_train_tfidf.shape,
    )

    print(
        "Testing TF-IDF: ",
        X_test_tfidf.shape,
    )


    # =====================================================
    # EXPERIMENTS
    # =====================================================

    results = []


    for C in C_VALUES:

        print(
            "\n" + "-" * 70
        )

        print(
            f"Training Logistic Regression with C={C}"
        )

        print(
            "-" * 70
        )


        model = LogisticRegression(

            C=C,

            max_iter=1000,

            class_weight="balanced",

            solver="liblinear",

            random_state=RANDOM_STATE,
        )


        model.fit(
            X_train_tfidf,
            y_train,
        )


        # -------------------------------------------------
        # Predictions
        # -------------------------------------------------

        y_pred = model.predict(
            X_test_tfidf
        )

        y_probability = (
            model
            .predict_proba(
                X_test_tfidf
            )[:, 1]
        )


        # -------------------------------------------------
        # Metrics
        # -------------------------------------------------

        accuracy = accuracy_score(
            y_test,
            y_pred,
        )

        precision = precision_score(
            y_test,
            y_pred,
            zero_division=0,
        )

        recall = recall_score(
            y_test,
            y_pred,
            zero_division=0,
        )

        f1 = f1_score(
            y_test,
            y_pred,
            zero_division=0,
        )

        roc_auc = roc_auc_score(
            y_test,
            y_probability,
        )


        # -------------------------------------------------
        # Confusion matrix
        # -------------------------------------------------

        tn, fp, fn, tp = (
            confusion_matrix(
                y_test,
                y_pred,
                labels=[0, 1],
            ).ravel()
        )


        # -------------------------------------------------
        # Print
        # -------------------------------------------------

        print(
            f"Accuracy : {accuracy:.4f}"
        )

        print(
            f"Precision: {precision:.4f}"
        )

        print(
            f"Recall   : {recall:.4f}"
        )

        print(
            f"F1       : {f1:.4f}"
        )

        print(
            f"ROC-AUC  : {roc_auc:.4f}"
        )

        print(
            f"False Positives: {fp}"
        )

        print(
            f"False Negatives: {fn}"
        )


        results.append({

            "C":
                C,

            "accuracy":
                accuracy,

            "precision":
                precision,

            "recall":
                recall,

            "f1":
                f1,

            "roc_auc":
                roc_auc,

            "false_positives":
                int(fp),

            "false_negatives":
                int(fn),

            "true_negatives":
                int(tn),

            "true_positives":
                int(tp),

        })


    # =====================================================
    # RESULTS TABLE
    # =====================================================

    results_df = pd.DataFrame(
        results
    )


    print(
        "\n" + "=" * 70
    )

    print(
        "REGULARIZATION EXPERIMENT RESULTS"
    )

    print(
        "=" * 70
    )

    print(
        results_df.to_string(
            index=False
        )
    )


    # =====================================================
    # BEST MODEL
    # =====================================================

    best = (
        results_df
        .loc[
            results_df["f1"].idxmax()
        ]
    )


    print(
        "\n" + "=" * 70
    )

    print(
        "BEST EXPERIMENTAL CONFIGURATION"
    )

    print(
        "=" * 70
    )

    print(
        f"C           : {best['C']}"
    )

    print(
        f"Accuracy    : {best['accuracy']:.4f}"
    )

    print(
        f"Precision   : {best['precision']:.4f}"
    )

    print(
        f"Recall      : {best['recall']:.4f}"
    )

    print(
        f"F1          : {best['f1']:.4f}"
    )

    print(
        f"ROC-AUC     : {best['roc_auc']:.4f}"
    )

    print(
        f"FP          : {int(best['false_positives'])}"
    )

    print(
        f"FN          : {int(best['false_negatives'])}"
    )


    # =====================================================
    # SAVE RESULTS
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
        "\nExperiment results saved to:"
    )

    print(
        REPORT_FILE
    )


    print(
        "\nIMPORTANT:"
    )

    print(
        "The final production model was NOT replaced."
    )


    print(
        "\n" + "=" * 70
    )


if __name__ == "__main__":

    main()
