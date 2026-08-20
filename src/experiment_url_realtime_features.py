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


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = "data/url_dataset.csv"
REPORT_FILE = "reports/url_realtime_feature_experiment.json"

RANDOM_STATE = 42


# ============================================================
# REAL-TIME URL FEATURES
# ============================================================
#
# These features can be calculated directly from the URL.
#
# No:
# - WHOIS lookup
# - DNS lookup
# - Google index lookup
# - PageRank lookup
# - web traffic lookup
# - webpage fetching
#
# is required.
# ============================================================


ALL_REALTIME_FEATURES = [
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


# ============================================================
# CORE REAL-TIME FEATURES
# ============================================================

CORE_REALTIME_FEATURES = [
    "ip",
    "nb_dots",
    "nb_hyphens",
    "nb_at",
    "nb_qm",
    "nb_eq",
    "nb_slash",
    "nb_www",
    "https_token",
    "ratio_digits_url",
    "ratio_digits_host",
    "punycode",
    "port",
    "nb_subdomains",
    "prefix_suffix",
    "random_domain",
    "shortening_service",
    "char_repeat",
    "phish_hints",
    "suspecious_tld",
]


# ============================================================
# IMPORTANT
# ============================================================
#
# domain_in_brand
# brand_in_subdomain
# brand_in_path
#
# are present in the dataset, but their calculation requires
# a brand vocabulary / matching mechanism.
#
# To keep this experiment strictly URL-local and reproducible,
# they are excluded from the core model.
#
# ============================================================


REALTIME_CORE_WITH_LENGTH = [
    "length_url",
    "length_hostname",
    *CORE_REALTIME_FEATURES,
]


# Remove duplicates while preserving order
REALTIME_CORE_WITH_LENGTH = list(
    dict.fromkeys(
        REALTIME_CORE_WITH_LENGTH
    )
)


# ============================================================
# EVALUATION
# ============================================================

def evaluate_model(
    X,
    y,
    feature_group,
):

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    model = RandomForestClassifier(
        n_estimators=500,
        max_depth=20,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="sqrt",
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    model.fit(
        X_train,
        y_train,
    )

    predictions = model.predict(
        X_test
    )

    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    tn, fp, fn, tp = confusion_matrix(
        y_test,
        predictions,
    ).ravel()

    return {
        "feature_group": feature_group,
        "feature_count": int(X.shape[1]),
        "accuracy": float(
            accuracy_score(
                y_test,
                predictions,
            )
        ),
        "precision": float(
            precision_score(
                y_test,
                predictions,
            )
        ),
        "recall": float(
            recall_score(
                y_test,
                predictions,
            )
        ),
        "f1": float(
            f1_score(
                y_test,
                predictions,
            )
        ),
        "roc_auc": float(
            roc_auc_score(
                y_test,
                probabilities,
            )
        ),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_negatives": int(tn),
        "true_positives": int(tp),
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("URL MODEL — REAL-TIME FEATURE EXPERIMENT")
    print("=" * 70)

    print(
        """
Purpose:
Evaluate URL features that can be calculated directly
from the URL without external reputation services
or webpage fetching.

Production URL model will NOT be modified.
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
        f"Dataset samples: {len(df)}"
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
    # VERIFY FEATURES
    # --------------------------------------------------------

    missing = [
        feature
        for feature in ALL_REALTIME_FEATURES
        if feature not in df.columns
    ]

    if missing:

        raise ValueError(
            f"Missing real-time features: {missing}"
        )

    # --------------------------------------------------------
    # REMOVE CONSTANT FEATURES
    # --------------------------------------------------------

    usable_all = [
        feature
        for feature in ALL_REALTIME_FEATURES
        if df[feature].nunique() > 1
    ]

    removed_all = sorted(
        set(ALL_REALTIME_FEATURES)
        - set(usable_all)
    )

    print(
        "\nAll real-time features:"
    )

    print(
        f"Requested: {len(ALL_REALTIME_FEATURES)}"
    )

    print(
        f"Usable:    {len(usable_all)}"
    )

    if removed_all:

        print(
            "Removed constant features:",
            ", ".join(removed_all),
        )

    # --------------------------------------------------------
    # CORE FEATURES
    # --------------------------------------------------------

    usable_core = [
        feature
        for feature in REALTIME_CORE_WITH_LENGTH
        if df[feature].nunique() > 1
    ]

    removed_core = sorted(
        set(REALTIME_CORE_WITH_LENGTH)
        - set(usable_core)
    )

    print(
        "\nCore real-time features:"
    )

    print(
        f"Requested: {len(REALTIME_CORE_WITH_LENGTH)}"
    )

    print(
        f"Usable:    {len(usable_core)}"
    )

    if removed_core:

        print(
            "Removed constant features:",
            ", ".join(removed_core),
        )

    # --------------------------------------------------------
    # FEATURE CONFIGURATIONS
    # --------------------------------------------------------

    configurations = {

        "All Real-Time Features":
            usable_all,

        "Core Real-Time Features":
            usable_core,

    }

    # --------------------------------------------------------
    # RUN EXPERIMENTS
    # --------------------------------------------------------

    results = []

    for name, features in configurations.items():

        print("\n")
        print("-" * 70)
        print(name)
        print("-" * 70)

        X = df[
            features
        ]

        metrics = evaluate_model(
            X,
            y,
            name,
        )

        results.append(
            metrics
        )

        print(
            f"Features : {metrics['feature_count']}"
        )

        print(
            f"Accuracy : {metrics['accuracy']:.4f}"
        )

        print(
            f"Precision: {metrics['precision']:.4f}"
        )

        print(
            f"Recall   : {metrics['recall']:.4f}"
        )

        print(
            f"F1       : {metrics['f1']:.4f}"
        )

        print(
            f"ROC-AUC  : {metrics['roc_auc']:.4f}"
        )

        print(
            f"False Positives: "
            f"{metrics['false_positives']}"
        )

        print(
            f"False Negatives: "
            f"{metrics['false_negatives']}"
        )

    # --------------------------------------------------------
    # RESULTS TABLE
    # --------------------------------------------------------

    results_df = pd.DataFrame(
        results
    )

    print("\n")
    print("=" * 70)
    print("REAL-TIME FEATURE EXPERIMENT RESULTS")
    print("=" * 70)

    print(
        results_df[
            [
                "feature_group",
                "feature_count",
                "accuracy",
                "precision",
                "recall",
                "f1",
                "roc_auc",
                "false_positives",
                "false_negatives",
            ]
        ].to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # BEST CONFIGURATIONS
    # --------------------------------------------------------

    best_f1 = results_df.loc[
        results_df["f1"].idxmax()
    ]

    best_auc = results_df.loc[
        results_df["roc_auc"].idxmax()
    ]

    best_recall = results_df.loc[
        results_df["recall"].idxmax()
    ]

    print("\n")
    print("=" * 70)
    print("BEST REAL-TIME CONFIGURATIONS")
    print("=" * 70)

    print(
        "\nBest F1:"
    )

    print(
        best_f1["feature_group"]
    )

    print(
        f"F1: {best_f1['f1']:.4f}"
    )

    print(
        "\nBest ROC-AUC:"
    )

    print(
        best_auc["feature_group"]
    )

    print(
        f"ROC-AUC: {best_auc['roc_auc']:.4f}"
    )

    print(
        "\nBest Recall:"
    )

    print(
        best_recall["feature_group"]
    )

    print(
        f"Recall: {best_recall['recall']:.4f}"
    )

    # --------------------------------------------------------
    # SAVE REPORT
    # --------------------------------------------------------

    os.makedirs(
        os.path.dirname(
            REPORT_FILE
        ),
        exist_ok=True,
    )

    report = {

        "dataset_samples":
            int(len(df)),

        "random_state":
            RANDOM_STATE,

        "feature_definitions": {

            "all_realtime":
                usable_all,

            "core_realtime":
                usable_core,
        },

        "removed_constant_features": {

            "all_realtime":
                removed_all,

            "core_realtime":
                removed_core,
        },

        "results":
            results,

        "best_f1": {

            "feature_group":
                best_f1["feature_group"],

            "f1":
                float(
                    best_f1["f1"]
                ),
        },

        "best_roc_auc": {

            "feature_group":
                best_auc["feature_group"],

            "roc_auc":
                float(
                    best_auc["roc_auc"]
                ),
        },

        "best_recall": {

            "feature_group":
                best_recall["feature_group"],

            "recall":
                float(
                    best_recall["recall"]
                ),
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

    print("\n")
    print(
        "Report saved to:"
    )

    print(
        REPORT_FILE
    )

    print("\n")
    print("=" * 70)
    print(
        "REAL-TIME FEATURE EXPERIMENT COMPLETE"
    )
    print("=" * 70)

    print(
        "\nIMPORTANT:"
    )

    print(
        "Production URL model remains unchanged."
    )


if __name__ == "__main__":
    main()
