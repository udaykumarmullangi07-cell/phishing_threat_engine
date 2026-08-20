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
    "reports/realtime_top25_validation.json"
)

RANDOM_STATE = 123

TEST_SIZE = 0.20

N_ESTIMATORS = 300


# ============================================================
# TOP 25 FEATURES
# ============================================================
#
# These are taken directly from the permutation ranking
# produced in the previous experiment.
#
# IMPORTANT:
# Do not change this list before validation.
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
# MODEL EVALUATION
# ============================================================

def evaluate(
    model,
    X_test,
    y_test,
):

    predictions = model.predict(
        X_test
    )

    probabilities = (
        model.predict_proba(
            X_test
        )[:, 1]
    )

    tn, fp, fn, tp = (
        confusion_matrix(
            y_test,
            predictions,
        ).ravel()
    )

    return {

        "accuracy":
            float(
                accuracy_score(
                    y_test,
                    predictions,
                )
            ),

        "precision":
            float(
                precision_score(
                    y_test,
                    predictions,
                    zero_division=0,
                )
            ),

        "recall":
            float(
                recall_score(
                    y_test,
                    predictions,
                    zero_division=0,
                )
            ),

        "f1":
            float(
                f1_score(
                    y_test,
                    predictions,
                    zero_division=0,
                )
            ),

        "roc_auc":
            float(
                roc_auc_score(
                    y_test,
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
        "REAL-TIME URL MODEL — TOP 25 FRESH VALIDATION"
    )

    print("=" * 70)

    print(
        """
Purpose:

Validate the permutation-selected Top 25
real-time URL features using a completely
different stratified train/test split.

This experiment does NOT modify the
production URL model.
"""
    )

    # ========================================================
    # LOAD DATA
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
    # LABEL
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
    # EXTRACT TOP 25
    # ========================================================

    print(
        "\nExtracting Top 25 features..."
    )

    X, failures = extract_features(
        df["url"]
    )

    if failures:

        print(
            "\nExtraction failures:",
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

    print(
        "\nFeature list:"
    )

    for number, feature in enumerate(
        TOP_25_FEATURES,
        start=1,
    ):

        print(
            f"{number:02d}. {feature}"
        )

    # ========================================================
    # FRESH SPLIT
    # ========================================================

    print(
        "\n" + "-" * 70
    )

    print(
        "Creating FRESH stratified split..."
    )

    print(
        "Random state:",
        RANDOM_STATE
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
    # TRAIN
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
    # EVALUATE
    # ========================================================

    result = evaluate(
        model,
        X_test,
        y_test,
    )

    # ========================================================
    # PRINT RESULTS
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "FRESH TOP 25 VALIDATION RESULTS"
    )

    print(
        "=" * 70
    )

    print(
        "Features :",
        len(TOP_25_FEATURES)
    )

    print(
        f"Accuracy : {result['accuracy']:.4f}"
    )

    print(
        f"Precision: {result['precision']:.4f}"
    )

    print(
        f"Recall   : {result['recall']:.4f}"
    )

    print(
        f"F1       : {result['f1']:.4f}"
    )

    print(
        f"ROC-AUC  : {result['roc_auc']:.4f}"
    )

    print(
        "False Positives:",
        result["false_positives"]
    )

    print(
        "False Negatives:",
        result["false_negatives"]
    )

    print(
        "True Negatives:",
        result["true_negatives"]
    )

    print(
        "True Positives:",
        result["true_positives"]
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
            "Fresh validation of permutation "
            "Top 25 real-time URL features",

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

        "results":
            result,
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
        "\nReport saved to:"
    )

    print(
        REPORT_FILE
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "TOP 25 FRESH VALIDATION COMPLETE"
    )

    print(
        "=" * 70
    )

    print(
        "\nProduction URL model remains unchanged."
    )


if __name__ == "__main__":

    main()
