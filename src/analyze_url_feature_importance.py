import os
import json

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance


INPUT_FILE = "data/url_dataset.csv"
REPORT_FILE = "reports/url_feature_importance.json"


def main():

    print("=" * 70)
    print("URL MODEL — FEATURE IMPORTANCE ANALYSIS")
    print("=" * 70)

    print(
        """
Purpose:
Determine which URL features contribute most strongly
to Random Forest phishing classification.

The production URL model will NOT be modified.
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
    # FEATURES
    # ---------------------------------------------------------

    excluded = [
        "url",
        "status",
    ]

    feature_columns = [
        column
        for column in df.columns
        if column not in excluded
        and pd.api.types.is_numeric_dtype(
            df[column]
        )
    ]

    X = df[
        feature_columns
    ].copy()

    # Remove constant features

    constant_features = [
        column
        for column in X.columns
        if X[column].nunique() <= 1
    ]

    X = X.drop(
        columns=constant_features
    )

    print(
        "\nFeatures used:",
        X.shape[1]
    )

    # ---------------------------------------------------------
    # SPLIT
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

    model.fit(
        X_train,
        y_train
    )

    print(
        "Training completed."
    )

    # ---------------------------------------------------------
    # BUILT-IN FEATURE IMPORTANCE
    # ---------------------------------------------------------

    print(
        "\nCalculating built-in feature importance..."
    )

    importance = pd.DataFrame(
        {
            "feature": X.columns,
            "importance": model.feature_importances_,
        }
    )

    importance = importance.sort_values(
        "importance",
        ascending=False,
    ).reset_index(
        drop=True
    )

    # ---------------------------------------------------------
    # PRINT TOP FEATURES
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("TOP 30 FEATURES")
    print("=" * 70)

    print(
        importance.head(30).to_string(
            index=False
        )
    )

    # ---------------------------------------------------------
    # PERMUTATION IMPORTANCE
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("CALCULATING PERMUTATION IMPORTANCE")
    print("=" * 70)

    print(
        """
This measures how much model performance changes
when individual features are randomly shuffled.
"""
    )

    permutation = permutation_importance(
        model,
        X_test,
        y_test,
        n_repeats=5,
        random_state=42,
        n_jobs=-1,
        scoring="f1",
    )

    permutation_df = pd.DataFrame(
        {
            "feature": X.columns,
            "importance_mean":
                permutation.importances_mean,
            "importance_std":
                permutation.importances_std,
        }
    )

    permutation_df = permutation_df.sort_values(
        "importance_mean",
        ascending=False,
    ).reset_index(
        drop=True
    )

    print(
        "\nTOP 30 PERMUTATION FEATURES"
    )

    print(
        permutation_df.head(30).to_string(
            index=False
        )
    )

    # ---------------------------------------------------------
    # MERGE RESULTS
    # ---------------------------------------------------------

    combined = importance.merge(
        permutation_df,
        on="feature",
        how="left",
    )

    # ---------------------------------------------------------
    # FEATURE CATEGORIES
    # ---------------------------------------------------------

    real_time_features = [
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
        "nb_percent",
        "nb_slash",
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

    external_features = [
        "domain_registration_length",
        "domain_age",
        "web_traffic",
        "dns_record",
        "google_index",
        "page_rank",
        "whois_registered_domain",
    ]

    combined["feature_type"] = "other"

    combined.loc[
        combined["feature"].isin(
            real_time_features
        ),
        "feature_type"
    ] = "URL lexical / structural"

    combined.loc[
        combined["feature"].isin(
            external_features
        ),
        "feature_type"
    ] = "External / reputation"

    # ---------------------------------------------------------
    # SAVE REPORT
    # ---------------------------------------------------------

    os.makedirs(
        "reports",
        exist_ok=True
    )

    report = {
        "dataset_samples": int(len(df)),
        "feature_count": int(X.shape[1]),
        "constant_features_removed":
            constant_features,
        "built_in_importance":
            importance.to_dict(
                orient="records"
            ),
        "permutation_importance":
            permutation_df.to_dict(
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
        "\nDetailed report saved to:"
    )

    print(
        REPORT_FILE
    )

    print("\n" + "=" * 70)
    print("FEATURE IMPORTANCE ANALYSIS COMPLETE")
    print("=" * 70)

    print(
        "\nProduction URL model remains unchanged."
    )


if __name__ == "__main__":
    main()
