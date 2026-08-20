import json
import os

import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
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
    "reports/realtime_permutation_validation.json"
)

RANDOM_STATE = 42

TEST_SIZE = 0.20

N_ESTIMATORS = 300

PERMUTATION_REPEATS = 10


# ============================================================
# ALL REAL-TIME FEATURES
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

    failed = []

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
                        f"Feature '{feature}' "
                        "returned None."
                    )

                row[feature] = value

            rows.append(row)

        except Exception as error:

            failed.append(
                {
                    "index": index,
                    "url": str(url),
                    "error": str(error),
                }
            )

    return (
        pd.DataFrame(
            rows,
            columns=ALL_FEATURES,
        ),
        failed,
    )


# ============================================================
# MODEL EVALUATION
# ============================================================

def evaluate_model(
    X,
    y,
    feature_group,
):

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=y,
        )
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
            feature_group,

        "feature_count":
            int(X.shape[1]),

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

    return (
        model,
        X_test,
        y_test,
        result,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)

    print(
        "REAL-TIME URL MODEL — "
        "PERMUTATION IMPORTANCE VALIDATION"
    )

    print("=" * 70)

    print(
        """
Purpose:

1. Extract all real-time URL features directly
   from raw URLs.

2. Train a Random Forest using all usable features.

3. Calculate permutation importance on the
   held-out test set.

4. Compare impurity importance against
   permutation importance.

5. Evaluate permutation-selected feature sets.

Production URL model will NOT be modified.
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
    )

    if y.isnull().any():

        raise ValueError(
            "Unexpected labels found."
        )

    y = y.astype(int)

    # ========================================================
    # EXTRACT FEATURES
    # ========================================================

    print(
        "\nExtracting features from raw URLs..."
    )

    X, failed = (
        extract_dataset_features(
            df["url"]
        )
    )

    if failed:

        print(
            "\nFeature extraction failures:",
            len(failed)
        )

        for item in failed[:10]:

            print(
                item
            )

        raise ValueError(
            "Feature extraction failed."
        )

    print(
        "\nSamples:",
        len(X)
    )

    print(
        "Features:",
        X.shape[1]
    )

    # ========================================================
    # CONSTANT FEATURES
    # ========================================================

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
        "\nUsable features:",
        X.shape[1]
    )

    # ========================================================
    # TRAIN / TEST SPLIT
    # ========================================================

    print(
        "\nCreating stratified train/test split..."
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
    # TRAIN FULL MODEL
    # ========================================================

    print(
        "\nTraining full Random Forest..."
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
    # BASELINE
    # ========================================================

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

    baseline = {

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

    print(
        "\n" + "=" * 70
    )

    print(
        "FULL MODEL BASELINE"
    )

    print(
        "=" * 70
    )

    print(
        f"Accuracy : {baseline['accuracy']:.4f}"
    )

    print(
        f"Precision: {baseline['precision']:.4f}"
    )

    print(
        f"Recall   : {baseline['recall']:.4f}"
    )

    print(
        f"F1       : {baseline['f1']:.4f}"
    )

    print(
        f"ROC-AUC  : {baseline['roc_auc']:.4f}"
    )

    # ========================================================
    # IMPURITY IMPORTANCE
    # ========================================================

    impurity_df = pd.DataFrame(
        {
            "feature":
                X.columns,

            "impurity_importance":
                model.feature_importances_,
        }
    )

    impurity_df = (
        impurity_df
        .sort_values(
            "impurity_importance",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )

    # ========================================================
    # PERMUTATION IMPORTANCE
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "CALCULATING PERMUTATION IMPORTANCE"
    )

    print(
        "=" * 70
    )

    print(
        f"""
Repeats: {PERMUTATION_REPEATS}

Scoring:
F1

This measures how much F1 changes when
one feature is randomly shuffled.
"""
    )

    permutation = permutation_importance(

        model,

        X_test,

        y_test,

        n_repeats=PERMUTATION_REPEATS,

        random_state=RANDOM_STATE,

        n_jobs=-1,

        scoring="f1",
    )

    permutation_df = pd.DataFrame(
        {
            "feature":
                X.columns,

            "permutation_mean":
                permutation.importances_mean,

            "permutation_std":
                permutation.importances_std,
        }
    )

    permutation_df = (
        permutation_df
        .sort_values(
            "permutation_mean",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )

    # ========================================================
    # COMBINE IMPORTANCE
    # ========================================================

    combined = impurity_df.merge(
        permutation_df,
        on="feature",
        how="left",
    )

    combined = (
        combined
        .sort_values(
            "permutation_mean",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )

    # ========================================================
    # PRINT PERMUTATION RANKING
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "PERMUTATION IMPORTANCE RANKING"
    )

    print(
        "=" * 70
    )

    print(
        combined[
            [
                "feature",
                "permutation_mean",
                "permutation_std",
                "impurity_importance",
            ]
        ].to_string(
            index=False
        )
    )

    # ========================================================
    # SELECT FEATURE GROUPS
    # ========================================================

    ranked_features = (
        permutation_df[
            "feature"
        ].tolist()
    )

    feature_groups = {

        "All Features":
            ranked_features,

        "Permutation Top 40":
            ranked_features[:40],

        "Permutation Top 30":
            ranked_features[:30],

        "Permutation Top 25":
            ranked_features[:25],

        "Permutation Top 20":
            ranked_features[:20],

        "Permutation Top 15":
            ranked_features[:15],

        "Permutation Top 10":
            ranked_features[:10],
    }

    # ========================================================
    # EVALUATE REDUCED MODELS
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "PERMUTATION-BASED FEATURE VALIDATION"
    )

    print(
        "=" * 70
    )

    results = []

    for group_name, features in (
        feature_groups.items()
    ):

        X_selected = X[
            features
        ]

        (
            selected_model,
            selected_X_test,
            selected_y_test,
            result,
        ) = evaluate_model(

            X_selected,

            y,

            group_name,
        )

        results.append(
            result
        )

        print(
            "\n" + "-" * 70
        )

        print(
            group_name
        )

        print(
            "-" * 70
        )

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
            result["false_positives"]
        )

        print(
            "False Negatives:",
            result["false_negatives"]
        )

    # ========================================================
    # RESULTS TABLE
    # ========================================================

    results_df = pd.DataFrame(
        results
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "PERMUTATION FEATURE RESULTS"
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

    # ========================================================
    # BEST CONFIGURATION
    # ========================================================

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
        "BEST PERMUTATION CONFIGURATIONS"
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
        "Feature count:",
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
        "Feature count:",
        int(
            best_auc[
                "feature_count"
            ]
        )
    )

    print(
        f"ROC-AUC: {best_auc['roc_auc']:.4f}"
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
            "Real-time URL permutation "
            "importance validation",

        "dataset_samples":
            int(len(df)),

        "feature_count":
            int(X.shape[1]),

        "constant_features_removed":
            constant_features,

        "baseline":
            baseline,

        "impurity_importance":
            impurity_df.to_dict(
                orient="records"
            ),

        "permutation_importance":
            permutation_df.to_dict(
                orient="records"
            ),

        "combined_importance":
            combined.to_dict(
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
        "PERMUTATION VALIDATION COMPLETE"
    )

    print(
        "=" * 70
    )

    print(
        "\nProduction URL model remains unchanged."
    )


if __name__ == "__main__":

    main()
