import os
import json

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)


INPUT_FILE = "data/url_dataset.csv"
REPORT_FILE = "reports/url_reduced_feature_experiment.json"


# =========================================================
# FEATURE GROUPS
# =========================================================

# Features that can be derived directly from the URL
# or its lexical / structural properties.

LEXICAL_STRUCTURAL_FEATURES = [

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

    "domain_in_brand",
    "brand_in_subdomain",
    "brand_in_path",

    "suspecious_tld",
]


def evaluate_model(
    model,
    X_train,
    X_test,
    y_train,
    y_test,
):

    model.fit(
        X_train,
        y_train
    )

    predictions = model.predict(
        X_test
    )

    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions
    )

    recall = recall_score(
        y_test,
        predictions
    )

    f1 = f1_score(
        y_test,
        predictions
    )

    roc_auc = roc_auc_score(
        y_test,
        probabilities
    )

    tn, fp, fn, tp = confusion_matrix(
        y_test,
        predictions
    ).ravel()

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": float(roc_auc),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_negatives": int(tn),
        "true_positives": int(tp),
    }


def main():

    print("=" * 70)
    print("URL MODEL — REDUCED FEATURE EXPERIMENT")
    print("=" * 70)

    print(
        """
Purpose:
Evaluate URL lexical/structural features without
external reputation-dependent features.

Production URL model will NOT be modified.
"""
    )

    # ---------------------------------------------------------
    # LOAD DATASET
    # ---------------------------------------------------------

    print("\nLoading dataset...")

    df = pd.read_csv(
        INPUT_FILE
    )

    print(
        "Dataset samples:",
        len(df)
    )

    # ---------------------------------------------------------
    # LABEL
    # ---------------------------------------------------------

    y = (
        df["status"]
        .map(
            {
                "legitimate": 0,
                "phishing": 1,
            }
        )
    )

    # ---------------------------------------------------------
    # CHECK FEATURES
    # ---------------------------------------------------------

    available_features = [
        feature
        for feature in LEXICAL_STRUCTURAL_FEATURES
        if feature in df.columns
    ]

    missing_features = [
        feature
        for feature in LEXICAL_STRUCTURAL_FEATURES
        if feature not in df.columns
    ]

    print(
        "\nRequested features:",
        len(LEXICAL_STRUCTURAL_FEATURES)
    )

    print(
        "Available features:",
        len(available_features)
    )

    if missing_features:

        print(
            "\nMissing features:"
        )

        for feature in missing_features:

            print(
                "-",
                feature
            )

    X = df[
        available_features
    ].copy()

    # ---------------------------------------------------------
    # REMOVE CONSTANT FEATURES
    # ---------------------------------------------------------

    constant_features = [
        feature
        for feature in X.columns
        if X[feature].nunique() <= 1
    ]

    if constant_features:

        print(
            "\nRemoving constant features:"
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

    # ---------------------------------------------------------
    # TRAIN TEST SPLIT
    # ---------------------------------------------------------

    print(
        "\nCreating stratified train/test split..."
    )

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42,
            stratify=y,
        )
    )

    print(
        "Training samples:",
        len(X_train)
    )

    print(
        "Testing samples :",
        len(X_test)
    )

    # ---------------------------------------------------------
    # RANDOM FOREST
    # ---------------------------------------------------------

    print(
        "\nTraining Random Forest..."
    )

    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
    )

    result = evaluate_model(
        model,
        X_train,
        X_test,
        y_train,
        y_test,
    )

    # ---------------------------------------------------------
    # RESULTS
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("REDUCED FEATURE RESULTS")
    print("=" * 70)

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
        f"False Positives: {result['false_positives']}"
    )

    print(
        f"False Negatives: {result['false_negatives']}"
    )

    # ---------------------------------------------------------
    # COMPARISON
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("REFERENCE: FULL FEATURE MODEL")
    print("=" * 70)

    print(
        "Accuracy : 0.9633"
    )

    print(
        "Precision: 0.9600"
    )

    print(
        "Recall   : 0.9668"
    )

    print(
        "F1       : 0.9634"
    )

    print(
        "ROC-AUC  : 0.9934"
    )

    print(
        "False Positives: 46"
    )

    print(
        "False Negatives: 38"
    )

    # ---------------------------------------------------------
    # SAVE
    # ---------------------------------------------------------

    os.makedirs(
        "reports",
        exist_ok=True
    )

    report = {
        "feature_group":
            "lexical_structural",
        "features":
            list(X.columns),
        "feature_count":
            int(X.shape[1]),
        "constant_features_removed":
            constant_features,
        "training_samples":
            int(len(X_train)),
        "testing_samples":
            int(len(X_test)),
        "results":
            result,
        "full_feature_reference": {
            "accuracy": 0.9633,
            "precision": 0.9600,
            "recall": 0.9668,
            "f1": 0.9634,
            "roc_auc": 0.9934,
            "false_positives": 46,
            "false_negatives": 38,
        },
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
        "\nExperiment report saved to:"
    )

    print(
        REPORT_FILE
    )

    print("\n" + "=" * 70)
    print("REDUCED FEATURE EXPERIMENT COMPLETE")
    print("=" * 70)

    print(
        "\nProduction URL model remains unchanged."
    )


if __name__ == "__main__":
    main()

