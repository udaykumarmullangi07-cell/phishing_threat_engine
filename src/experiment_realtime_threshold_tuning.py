import json
import os

import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split

from realtime_url_features import (
    extract_realtime_url_features,
)


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = "data/url_dataset.csv"

REPORT_FILE = (
    "reports/realtime_threshold_tuning.json"
)

RANDOM_STATE = 123

TEST_SIZE = 0.20

N_ESTIMATORS = 300


# ============================================================
# TOP 25 FEATURES
# ============================================================
#
# These are the permutation-selected features validated
# in the previous experiment.
# ============================================================

TOP_25_FEATURES = [

    "nb_www",

    "phish_hints",

    "nb_slash",

    "nb_hyphens",

    "nb_subdomains",

    "ratio_digits_host",

    "char_repeat",

    "ratio_digits_url",

    "longest_words_raw",

    "length_hostname",

    "nb_underscore",

    "https_token",

    "nb_dots",

    "longest_word_host",

    "avg_word_host",

    "nb_com",

    "shortest_word_host",

    "length_words_raw",

    "path_extension",

    "avg_word_path",

    "prefix_suffix",

    "longest_word_path",

    "shortest_words_raw",

    "nb_qm",

    "nb_eq",
]


# ============================================================
# THRESHOLDS TO TEST
# ============================================================

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
]


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_features(urls):

    rows = []

    failures = []

    for index, url in enumerate(urls):

        try:

            extracted = (
                extract_realtime_url_features(
                    url
                )
            )

            row = {}

            for feature in TOP_25_FEATURES:

                if feature not in extracted:

                    raise ValueError(
                        f"Missing feature: {feature}"
                    )

                value = extracted[feature]

                if value is None:

                    raise ValueError(
                        f"Feature '{feature}' "
                        "returned None"
                    )

                row[feature] = value

            rows.append(row)

        except Exception as error:

            failures.append(
                {
                    "index": index,
                    "url": str(url),
                    "error": str(error),
                }
            )

    return (
        pd.DataFrame(
            rows,
            columns=TOP_25_FEATURES,
        ),
        failures,
    )


# ============================================================
# THRESHOLD EVALUATION
# ============================================================

def evaluate_threshold(
    y_true,
    probabilities,
    threshold,
):

    predictions = (
        probabilities >= threshold
    ).astype(int)

    tn, fp, fn, tp = (
        confusion_matrix(
            y_true,
            predictions,
            labels=[0, 1],
        ).ravel()
    )

    return {

        "threshold":
            float(threshold),

        "accuracy":
            float(
                accuracy_score(
                    y_true,
                    predictions,
                )
            ),

        "precision":
            float(
                precision_score(
                    y_true,
                    predictions,
                    zero_division=0,
                )
            ),

        "recall":
            float(
                recall_score(
                    y_true,
                    predictions,
                    zero_division=0,
                )
            ),

        "f1":
            float(
                f1_score(
                    y_true,
                    predictions,
                    zero_division=0,
                )
            ),

        "roc_auc":
            float(
                roc_auc_score(
                    y_true,
                    probabilities,
                )
            ),

        "false_positives":
            int(fp),

        "false_negatives":
            int(fn),

        "true_negatives":
            int(tn),

        "true_positives":
            int(tp),
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)

    print(
        "REAL-TIME URL MODEL — THRESHOLD TUNING"
    )

    print("=" * 70)

    print(
        """
Purpose:

Evaluate different classification thresholds for
the validated Top-25 real-time URL model.

The goal is to understand the trade-off between:

- Precision
- Recall
- F1
- False positives
- False negatives

The production URL model will NOT be modified.
"""
    )

    # ========================================================
    # LOAD DATASET
    # ========================================================

    print(
        "\nLoading dataset..."
    )

    df = pd.read_csv(
        INPUT_FILE
    )

    print(
        "Dataset samples:",
        len(df)
    )

    # ========================================================
    # LABEL ENCODING
    # ========================================================

    y = (
        df["status"]
        .map(
            {
                "legitimate": 0,
                "phishing": 1,
            }
        )
        .astype(int)
    )

    # ========================================================
    # FEATURE EXTRACTION
    # ========================================================

    print(
        "\nExtracting Top 25 features..."
    )

    X, failures = extract_features(
        df["url"]
    )

    if failures:

        print(
            "\nFeature extraction failures:",
            len(failures)
        )

        for failure in failures[:10]:

            print(
                failure
            )

        raise RuntimeError(
            "Feature extraction failed."
        )

    print(
        "Samples:",
        len(X)
    )

    print(
        "Features:",
        len(X.columns)
    )

    # ========================================================
    # FRESH STRATIFIED SPLIT
    # ========================================================

    print(
        "\nCreating fresh stratified split..."
    )

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
        len(X_train)
    )

    print(
        "Testing samples:",
        len(X_test)
    )

    # ========================================================
    # TRAIN MODEL
    # ========================================================

    print(
        "\nTraining Random Forest..."
    )

    model = RandomForestClassifier(

        n_estimators=N_ESTIMATORS,

        random_state=RANDOM_STATE,

        n_jobs=-1,

        class_weight="balanced",
    )

    model.fit(
        X_train,
        y_train,
    )

    print(
        "Training completed."
    )

    # ========================================================
    # PREDICT PROBABILITIES
    # ========================================================

    print(
        "\nGenerating phishing probabilities..."
    )

    probabilities = (
        model.predict_proba(
            X_test
        )[:, 1]
    )

    # ========================================================
    # ROC-AUC
    # ========================================================

    roc_auc = roc_auc_score(
        y_test,
        probabilities,
    )

    # ========================================================
    # THRESHOLD TESTING
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "THRESHOLD ANALYSIS"
    )

    print(
        "=" * 70
    )

    print(
        f"{'Threshold':<12}"
        f"{'Accuracy':<12}"
        f"{'Precision':<12}"
        f"{'Recall':<12}"
        f"{'F1':<12}"
        f"{'FP':<8}"
        f"{'FN':<8}"
    )

    print(
        "-" * 70
    )

    results = []

    for threshold in THRESHOLDS:

        result = evaluate_threshold(
            y_test,
            probabilities,
            threshold,
        )

        result["roc_auc"] = float(
            roc_auc
        )

        results.append(
            result
        )

        print(
            f"{threshold:<12.2f}"
            f"{result['accuracy']:<12.4f}"
            f"{result['precision']:<12.4f}"
            f"{result['recall']:<12.4f}"
            f"{result['f1']:<12.4f}"
            f"{result['false_positives']:<8}"
            f"{result['false_negatives']:<8}"
        )

    # ========================================================
    # FIND BEST THRESHOLDS
    # ========================================================

    best_f1 = max(
        results,
        key=lambda item: item["f1"],
    )

    best_recall = max(
        results,
        key=lambda item: item["recall"],
    )

    best_precision = max(
        results,
        key=lambda item: item["precision"],
    )

    # ========================================================
    # PRINT BEST CONFIGURATIONS
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "BEST THRESHOLD CONFIGURATIONS"
    )

    print(
        "=" * 70
    )

    print(
        "\nBest F1:"
    )

    print(
        f"Threshold: {best_f1['threshold']:.2f}"
    )

    print(
        f"F1: {best_f1['f1']:.4f}"
    )

    print(
        f"Precision: {best_f1['precision']:.4f}"
    )

    print(
        f"Recall: {best_f1['recall']:.4f}"
    )

    print(
        f"False Positives: "
        f"{best_f1['false_positives']}"
    )

    print(
        f"False Negatives: "
        f"{best_f1['false_negatives']}"
    )

    print(
        "\nBest Recall:"
    )

    print(
        f"Threshold: {best_recall['threshold']:.2f}"
    )

    print(
        f"Recall: {best_recall['recall']:.4f}"
    )

    print(
        f"Precision: {best_recall['precision']:.4f}"
    )

    print(
        f"F1: {best_recall['f1']:.4f}"
    )

    print(
        f"False Positives: "
        f"{best_recall['false_positives']}"
    )

    print(
        f"False Negatives: "
        f"{best_recall['false_negatives']}"
    )

    print(
        "\nBest Precision:"
    )

    print(
        f"Threshold: "
        f"{best_precision['threshold']:.2f}"
    )

    print(
        f"Precision: "
        f"{best_precision['precision']:.4f}"
    )

    print(
        f"Recall: "
        f"{best_precision['recall']:.4f}"
    )

    print(
        f"F1: "
        f"{best_precision['f1']:.4f}"
    )

    print(
        f"False Positives: "
        f"{best_precision['false_positives']}"
    )

    print(
        f"False Negatives: "
        f"{best_precision['false_negatives']}"
    )

    # ========================================================
    # SAVE REPORT
    # ========================================================

    os.makedirs(
        "reports",
        exist_ok=True,
    )

    report = {

        "experiment":
            "Real-time URL threshold tuning",

        "dataset_samples":
            int(len(df)),

        "feature_count":
            len(TOP_25_FEATURES),

        "features":
            TOP_25_FEATURES,

        "random_state":
            RANDOM_STATE,

        "test_size":
            TEST_SIZE,

        "model":
            {
                "type":
                    "RandomForestClassifier",

                "n_estimators":
                    N_ESTIMATORS,

                "class_weight":
                    "balanced",
            },

        "roc_auc":
            float(roc_auc),

        "thresholds":
            results,

        "best_f1":
            best_f1,

        "best_recall":
            best_recall,

        "best_precision":
            best_precision,
    }

    with open(
        REPORT_FILE,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            report,
            file,
            indent=4,
        )

    print(
        "\n" + "=" * 70
    )

    print(
        "THRESHOLD TUNING COMPLETE"
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

    print(
        "\nProduction URL model remains unchanged."
    )


if __name__ == "__main__":

    main()
