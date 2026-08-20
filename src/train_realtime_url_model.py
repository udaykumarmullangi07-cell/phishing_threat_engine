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
    "reports/realtime_url_model_experiment.json"
)

RANDOM_STATE = 42


# ============================================================
# FEATURES
# ============================================================

MODEL_FEATURES = [

    "length_url",
    "length_hostname",
    "ip",

    "nb_dots",
    "nb_hyphens",
    "nb_at",
    "nb_qm",
    "nb_and",
    "nb_eq",
    "nb_underscore",
    "nb_tilde",
    "nb_percent",
    "nb_slash",
    "nb_star",
    "nb_colon",
    "nb_comma",
    "nb_semicolumn",
    "nb_dollar",
    "nb_space",

    "nb_www",
    "nb_com",
    "nb_dslash",

    "http_in_path",
    "https_token",

    "ratio_digits_url",
    "ratio_digits_host",

    "punycode",
    "port",

    "tld_in_path",
    "tld_in_subdomain",
    "abnormal_subdomain",
    "nb_subdomains",

    "prefix_suffix",
    "random_domain",
    "shortening_service",

    "path_extension",

    "nb_redirection",
    "nb_external_redirection",

    "length_words_raw",
    "char_repeat",

    "shortest_words_raw",
    "shortest_word_host",
    "shortest_word_path",

    "longest_words_raw",
    "longest_word_host",
    "longest_word_path",

    "avg_words_raw",
    "avg_word_host",
    "avg_word_path",

    "phish_hints",
    "suspecious_tld",
]


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def build_feature_dataframe(urls):

    rows = []

    failed = []

    for index, url in enumerate(urls):

        try:

            features = (
                extract_realtime_url_features(
                    url
                )
            )

            row = {
                feature: features[feature]
                for feature in MODEL_FEATURES
            }

            rows.append(row)

        except Exception as error:

            failed.append(
                {
                    "index": index,
                    "url": str(url),
                    "error": str(error),
                }
            )

    dataframe = pd.DataFrame(
        rows,
        columns=MODEL_FEATURES,
    )

    return dataframe, failed


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print(
        "REAL-TIME URL MODEL — "
        "END-TO-END EXPERIMENT"
    )
    print("=" * 70)

    print(
        """
Purpose:
Train and evaluate a phishing URL classifier where
the SAME URL feature extractor is used for both
training and inference.

No external reputation services are used.

Excluded:
- WHOIS
- DNS reputation
- Google index
- PageRank
- web traffic
- webpage fetching
- brand database
"""
    )

    # --------------------------------------------------------
    # LOAD DATASET
    # --------------------------------------------------------

    print("\nLoading dataset...")

    df = pd.read_csv(
        INPUT_FILE
    )

    print(
        "Dataset samples:",
        len(df)
    )

    # --------------------------------------------------------
    # LABEL
    # --------------------------------------------------------

    y = (
        df["status"]
        .map(
            {
                "legitimate": 0,
                "phishing": 1,
            }
        )
    )

    if y.isnull().any():

        raise ValueError(
            "Unexpected labels found."
        )

    y = y.astype(int)

    # --------------------------------------------------------
    # EXTRACT FEATURES FROM RAW URLS
    # --------------------------------------------------------

    print(
        "\nExtracting features from raw URLs..."
    )

    X, failed = build_feature_dataframe(
        df["url"]
    )

    if failed:

        print(
            "\nWARNING:"
        )

        print(
            "Failed URL extractions:",
            len(failed)
        )

        for item in failed[:10]:

            print(
                item
            )

        # Keep only successfully extracted rows
        valid_indices = [
            i
            for i in range(len(df))
            if i not in {
                item["index"]
                for item in failed
            }
        ]

        y = y.iloc[
            valid_indices
        ].reset_index(
            drop=True
        )

    print(
        "\nExtracted feature matrix:"
    )

    print(
        "Samples:",
        len(X)
    )

    print(
        "Features:",
        X.shape[1]
    )

    # --------------------------------------------------------
    # CHECK MISSING VALUES
    # --------------------------------------------------------

    missing_values = (
        X.isnull().sum()
    )

    missing_features = (
        missing_values[
            missing_values > 0
        ]
    )

    if len(missing_features) > 0:

        print(
            "\nMissing values detected:"
        )

        print(
            missing_features
        )

        raise ValueError(
            "Feature matrix contains missing values."
        )

    print(
        "\nNo missing feature values."
    )

    # --------------------------------------------------------
    # REMOVE CONSTANT FEATURES
    # --------------------------------------------------------

    constant_features = [
        feature
        for feature in X.columns
        if X[feature].nunique() <= 1
    ]

    if constant_features:

        print(
            "\nConstant features removed:"
        )

        for feature in constant_features:

            print(
                "-",
                feature
            )

        X = X.drop(
            columns=constant_features
        )

    print(
        "\nFinal feature count:",
        X.shape[1]
    )

    # --------------------------------------------------------
    # TRAIN / TEST SPLIT
    # --------------------------------------------------------

    print(
        "\nCreating stratified train/test split..."
    )

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=0.20,
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

    # --------------------------------------------------------
    # RANDOM FOREST
    # --------------------------------------------------------

    print(
        "\nTraining Random Forest..."
    )

    model = RandomForestClassifier(

        n_estimators=300,

        random_state=RANDOM_STATE,

        n_jobs=-1,

        class_weight="balanced",
    )

    model.fit(
        X_train,
        y_train
    )

    print(
        "Training completed."
    )

    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    predictions = model.predict(
        X_test
    )

    probabilities = (
        model.predict_proba(
            X_test
        )[:, 1]
    )

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    tn, fp, fn, tp = (
        confusion_matrix(
            y_test,
            predictions,
        ).ravel()
    )

    accuracy = accuracy_score(
        y_test,
        predictions,
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0,
    )

    roc_auc = roc_auc_score(
        y_test,
        probabilities,
    )

    # --------------------------------------------------------
    # RESULTS
    # --------------------------------------------------------

    print("\n" + "=" * 70)

    print(
        "REAL-TIME URL MODEL RESULTS"
    )

    print("=" * 70)

    print(
        f"Features : {X.shape[1]}"
    )

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

    print(
        f"True Negatives: {tn}"
    )

    print(
        f"True Positives: {tp}"
    )

    # --------------------------------------------------------
    # FEATURE IMPORTANCE
    # --------------------------------------------------------

    importance = pd.DataFrame(
        {
            "feature": X.columns,
            "importance":
                model.feature_importances_,
        }
    )

    importance = (
        importance
        .sort_values(
            "importance",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "TOP 20 REAL-TIME FEATURES"
    )

    print(
        "=" * 70
    )

    print(
        importance.head(20)
        .to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # SAVE MODEL METADATA
    # --------------------------------------------------------

    os.makedirs(
        "reports",
        exist_ok=True,
    )

    report = {

        "experiment":
            "End-to-end real-time URL model",

        "dataset_samples":
            int(len(df)),

        "successful_extractions":
            int(len(X)),

        "failed_extractions":
            int(len(failed)),

        "random_state":
            RANDOM_STATE,

        "feature_count":
            int(X.shape[1]),

        "features":
            list(X.columns),

        "constant_features_removed":
            constant_features,

        "metrics": {

            "accuracy":
                float(accuracy),

            "precision":
                float(precision),

            "recall":
                float(recall),

            "f1":
                float(f1),

            "roc_auc":
                float(roc_auc),

            "false_positives":
                int(fp),

            "false_negatives":
                int(fn),

            "true_negatives":
                int(tn),

            "true_positives":
                int(tp),
        },

        "feature_importance":
            importance.to_dict(
                orient="records"
            ),
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
        "END-TO-END REAL-TIME "
        "URL EXPERIMENT COMPLETE"
    )

    print("=" * 70)

    print(
        "\nProduction URL model remains unchanged."
    )


if __name__ == "__main__":

    main()
