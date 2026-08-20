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
    "reports/realtime_feature_reduction.json"
)

RANDOM_STATE = 42


# ============================================================
# COMPLETE REAL-TIME FEATURE SET
# ============================================================

ALL_FEATURES = [

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

def extract_dataset_features(urls):

    rows = []
    failed_indices = []

    for index, url in enumerate(urls):

        try:

            extracted = (
                extract_realtime_url_features(
                    url
                )
            )

            row = {}

            for feature in ALL_FEATURES:

                if feature not in extracted:

                    raise ValueError(
                        f"Missing feature: {feature}"
                    )

                value = extracted[feature]

                if value is None:

                    raise ValueError(
                        f"Feature {feature} is None"
                    )

                row[feature] = value

            rows.append(row)

        except Exception as error:

            print(
                f"Feature extraction failed "
                f"at row {index}: {error}"
            )

            failed_indices.append(
                index
            )

    X = pd.DataFrame(
        rows,
        columns=ALL_FEATURES,
    )

    return X, failed_indices


# ============================================================
# RANK FEATURES
# ============================================================

def rank_features(X, y):

    print(
        "\nTraining ranking Random Forest..."
    )

    model = RandomForestClassifier(
        n_estimators=300,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight="balanced",
    )

    model.fit(
        X,
        y,
    )

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

    return importance


# ============================================================
# EVALUATE FEATURE SET
# ============================================================

def evaluate_feature_set(
    X,
    y,
    features,
    group_name,
):

    X_selected = X[
        features
    ]

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X_selected,
            y,
            test_size=0.20,
            random_state=RANDOM_STATE,
            stratify=y,
        )
    )

    model = RandomForestClassifier(
        n_estimators=300,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight="balanced",
    )

    model.fit(
        X_train,
        y_train,
    )

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

    result = {

        "feature_group":
            group_name,

        "feature_count":
            len(features),

        "features":
            features,

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

    print("\n" + "-" * 70)

    print(
        group_name
    )

    print("-" * 70)

    print(
        "Features :",
        len(features)
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
        fp
    )

    print(
        "False Negatives:",
        fn
    )

    return result


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)

    print(
        "REAL-TIME URL MODEL — "
        "FEATURE REDUCTION EXPERIMENT"
    )

    print("=" * 70)

    print(
        """
Purpose:
Determine the smallest useful real-time URL feature
set while maintaining strong phishing detection.

The SAME raw URL extractor is used for all experiments.

No:
- WHOIS
- DNS
- PageRank
- Google index
- Web traffic
- webpage fetching
- brand database

Production URL model will NOT be modified.
"""
    )

    # --------------------------------------------------------
    # LOAD DATASET
    # --------------------------------------------------------

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
    # EXTRACT FEATURES
    # --------------------------------------------------------

    print(
        "\nExtracting real-time features..."
    )

    X, failed_indices = (
        extract_dataset_features(
            df["url"]
        )
    )

    if failed_indices:

        print(
            "\nFailed extractions:",
            len(failed_indices)
        )

        raise ValueError(
            "Feature extraction failed. "
            "Fix extractor before continuing."
        )

    print(
        "\nFeature matrix:"
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
    # CHECK MISSING
    # --------------------------------------------------------

    if X.isnull().any().any():

        missing = (
            X.isnull()
            .sum()
        )

        print(
            missing[
                missing > 0
            ]
        )

        raise ValueError(
            "Missing values detected."
        )

    print(
        "Missing values: 0"
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
            "\nConstant features:"
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
        "\nUsable features:",
        X.shape[1]
    )

    # --------------------------------------------------------
    # RANK FEATURES
    # --------------------------------------------------------

    importance = rank_features(
        X,
        y,
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "FEATURE RANKING"
    )

    print(
        "=" * 70
    )

    print(
        importance.to_string(
            index=False
        )
    )

    ranked_features = (
        importance[
            "feature"
        ].tolist()
    )

    # --------------------------------------------------------
    # FEATURE GROUPS
    # --------------------------------------------------------

    feature_groups = {}

    # All features

    feature_groups[
        "All Features"
    ] = ranked_features

    # Top 40

    feature_groups[
        "Top 40"
    ] = ranked_features[:40]

    # Top 30

    feature_groups[
        "Top 30"
    ] = ranked_features[:30]

    # Top 25

    feature_groups[
        "Top 25"
    ] = ranked_features[:25]

    # Top 20

    feature_groups[
        "Top 20"
    ] = ranked_features[:20]

    # Top 15

    feature_groups[
        "Top 15"
    ] = ranked_features[:15]

    # Top 10

    feature_groups[
        "Top 10"
    ] = ranked_features[:10]

    # --------------------------------------------------------
    # EVALUATE
    # --------------------------------------------------------

    results = []

    print(
        "\n" + "=" * 70
    )

    print(
        "FEATURE REDUCTION RESULTS"
    )

    print(
        "=" * 70
    )

    for group_name, features in (
        feature_groups.items()
    ):

        result = evaluate_feature_set(

            X,

            y,

            features,

            group_name,
        )

        results.append(
            result
        )

    # --------------------------------------------------------
    # RESULTS TABLE
    # --------------------------------------------------------

    results_df = pd.DataFrame(
        results
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "FEATURE REDUCTION SUMMARY"
    )

    print(
        "=" * 70
    )

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
    # BEST F1
    # --------------------------------------------------------

    best_f1 = results_df.loc[
        results_df["f1"].idxmax()
    ]

    best_auc = results_df.loc[
        results_df["roc_auc"].idxmax()
    ]

    print(
        "\n" + "=" * 70
    )

    print(
        "BEST CONFIGURATIONS"
    )

    print(
        "=" * 70
    )

    print(
        "\nBest F1:"
    )

    print(
        "Feature group:",
        best_f1[
            "feature_group"
        ]
    )

    print(
        "Features:",
        int(
            best_f1[
                "feature_count"
            ]
        )
    )

    print(
        f"F1: {best_f1['f1']:.4f}"
    )

    print(
        "\nBest ROC-AUC:"
    )

    print(
        "Feature group:",
        best_auc[
            "feature_group"
        ]
    )

    print(
        "Features:",
        int(
            best_auc[
                "feature_count"
            ]
        )
    )

    print(
        f"ROC-AUC: {best_auc['roc_auc']:.4f}"
    )

    # --------------------------------------------------------
    # SAVE REPORT
    # --------------------------------------------------------

    os.makedirs(
        "reports",
        exist_ok=True,
    )

    report = {

        "experiment":
            "Real-time URL feature reduction",

        "dataset_samples":
            int(len(df)),

        "usable_feature_count":
            int(X.shape[1]),

        "constant_features_removed":
            constant_features,

        "feature_ranking":
            importance.to_dict(
                orient="records"
            ),

        "results":
            results,

        "best_f1":
            {
                "feature_group":
                    best_f1[
                        "feature_group"
                    ],

                "feature_count":
                    int(
                        best_f1[
                            "feature_count"
                        ]
                    ),

                "f1":
                    float(
                        best_f1["f1"]
                    ),
            },

        "best_roc_auc":
            {
                "feature_group":
                    best_auc[
                        "feature_group"
                    ],

                "feature_count":
                    int(
                        best_auc[
                            "feature_count"
                        ]
                    ),

                "roc_auc":
                    float(
                        best_auc[
                            "roc_auc"
                        ]
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
        "FEATURE REDUCTION EXPERIMENT COMPLETE"
    )

    print(
        "=" * 70
    )

    print(
        "\nProduction URL model remains unchanged."
    )


if __name__ == "__main__":

    main()
