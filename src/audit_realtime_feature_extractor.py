import json
import os
import sys

import pandas as pd


# ============================================================
# PATH SETUP
# ============================================================

# Allow imports from the project root even when this file is
# executed using:
#
#     python src/audit_realtime_feature_extractor.py
#
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

SRC_DIR = os.path.join(
    PROJECT_ROOT,
    "src"
)

if SRC_DIR not in sys.path:
    sys.path.insert(
        0,
        SRC_DIR
    )


# ============================================================
# IMPORT EXPERIMENTAL EXTRACTOR
# ============================================================

from url_experimental_features import (
    extract_lexical_features,
    build_experimental_features,
)


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_FILE = os.path.join(
    PROJECT_ROOT,
    "data",
    "url_dataset.csv"
)

REPORT_FILE = os.path.join(
    PROJECT_ROOT,
    "reports",
    "realtime_feature_extractor_audit.json"
)


# ============================================================
# EXPECTED REAL-TIME FEATURES
# ============================================================

REALTIME_FEATURES = [
    "length_url",
    "length_hostname",
    "ip",
    "nb_dots",
    "nb_hyphens",
    "nb_at",
    "nb_qm",
    "nb_and",
    "nb_or",
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
]


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("REAL-TIME URL FEATURE EXTRACTOR AUDIT")
    print("=" * 70)

    print("""
Purpose:
Verify that the experimental URL feature extractor can
reproduce the locally computable URL features directly
from raw URLs.

This audit does NOT modify the production URL model.
""")

    # --------------------------------------------------------
    # LOAD DATASET
    # --------------------------------------------------------

    print("\nLoading dataset...")

    df = pd.read_csv(
        DATASET_FILE
    )

    print(
        f"Dataset samples: {len(df)}"
    )

    # --------------------------------------------------------
    # VERIFY URL COLUMN
    # --------------------------------------------------------

    if "url" not in df.columns:

        raise ValueError(
            "Dataset does not contain a 'url' column."
        )

    # --------------------------------------------------------
    # CHECK EXPECTED FEATURES EXIST IN DATASET
    # --------------------------------------------------------

    missing_dataset_features = [
        feature
        for feature in REALTIME_FEATURES
        if feature not in df.columns
    ]

    if missing_dataset_features:

        raise ValueError(
            "The following expected real-time "
            "features are missing from the dataset:\n"
            + "\n".join(
                missing_dataset_features
            )
        )

    # --------------------------------------------------------
    # TEST REPRESENTATIVE URLS
    # --------------------------------------------------------

    test_urls = [

        (
            "Legitimate Google",
            "https://www.google.com"
        ),

        (
            "Legitimate GitHub",
            "https://github.com/python/cpython/issues"
        ),

        (
            "IP Login",
            "http://192.168.1.25/login"
        ),

        (
            "Suspicious Login",
            "http://secure-login.example.com/account/verify"
        ),

        (
            "Long Suspicious URL",
            (
                "http://secure-login.example.com/"
                "account/verify/update"
                "?session=983472983472"
                "&token=928374928374"
            )
        ),
    ]

    # --------------------------------------------------------
    # EXTRACT FEATURES
    # --------------------------------------------------------

    results = []

    print("\n")
    print("=" * 70)
    print("EXTRACTOR TESTS")
    print("=" * 70)

    for name, url in test_urls:

        print("\n" + "-" * 70)
        print(name)
        print("-" * 70)

        print("URL:")
        print(url)

        try:

            features = extract_lexical_features(
                url
            )

        except Exception as error:

            print(
                "\nERROR:"
            )

            print(
                error
            )

            results.append({
                "name": name,
                "url": url,
                "success": False,
                "error": str(error),
            })

            continue

        # ----------------------------------------------------
        # Determine available features
        # ----------------------------------------------------

        available = [
            feature
            for feature in REALTIME_FEATURES
            if feature in features
        ]

        missing = [
            feature
            for feature in REALTIME_FEATURES
            if feature not in features
        ]

        # ----------------------------------------------------
        # Check for None values
        # ----------------------------------------------------

        none_features = [
            feature
            for feature in available
            if features[feature] is None
        ]

        # ----------------------------------------------------
        # Check numeric values
        # ----------------------------------------------------

        non_numeric = []

        for feature in available:

            value = features[feature]

            if value is None:
                continue

            if not isinstance(
                value,
                (int, float)
            ):

                non_numeric.append(
                    feature
                )

        # ----------------------------------------------------
        # Print result
        # ----------------------------------------------------

        print(
            f"\nExpected features : "
            f"{len(REALTIME_FEATURES)}"
        )

        print(
            f"Extracted features: "
            f"{len(available)}"
        )

        if missing:

            print(
                "\nMissing features:"
            )

            for feature in missing:

                print(
                    f"- {feature}"
                )

        else:

            print(
                "\nAll real-time features "
                "were extracted."
            )

        if none_features:

            print(
                "\nFeatures with None values:"
            )

            for feature in none_features:

                print(
                    f"- {feature}"
                )

        if non_numeric:

            print(
                "\nNon-numeric features:"
            )

            for feature in non_numeric:

                print(
                    f"- {feature}"
                )

        success = (
            len(missing) == 0
            and len(none_features) == 0
            and len(non_numeric) == 0
        )

        print(
            "\nResult:",
            "PASS" if success else "FAIL"
        )

        results.append({
            "name": name,
            "url": url,
            "success": success,
            "expected_feature_count":
                len(REALTIME_FEATURES),
            "extracted_feature_count":
                len(available),
            "missing_features":
                missing,
            "none_features":
                none_features,
            "non_numeric_features":
                non_numeric,
        })

    # ========================================================
    # DATASET-LEVEL REFERENCE COMPARISON
    # ========================================================

    print("\n")
    print("=" * 70)
    print("DATASET-LEVEL FEATURE AUDIT")
    print("=" * 70)

    print("""
The extractor is checked against the dataset's URL feature
columns to determine whether the locally computable feature
names and values can be reproduced from raw URLs.
""")

    comparison_rows = []

    # Use a small representative sample rather than all
    # 11,430 URLs to keep this audit lightweight.

    sample_size = min(
        100,
        len(df)
    )

    sample = df.sample(
        sample_size,
        random_state=42
    )

    for index, row in sample.iterrows():

        url = row["url"]

        try:

            extracted = extract_lexical_features(
                url
            )

        except Exception:

            continue

        for feature in REALTIME_FEATURES:

            if feature not in extracted:
                continue

            dataset_value = row[
                feature
            ]

            extractor_value = extracted[
                feature
            ]

            # Skip missing values
            if pd.isna(
                dataset_value
            ):

                continue

            if extractor_value is None:

                continue

            try:

                dataset_numeric = float(
                    dataset_value
                )

                extractor_numeric = float(
                    extractor_value
                )

            except (
                ValueError,
                TypeError
            ):

                continue

            comparison_rows.append({

                "feature":
                    feature,

                "dataset_value":
                    dataset_numeric,

                "extractor_value":
                    extractor_numeric,

                "exact_match":
                    dataset_numeric
                    == extractor_numeric,

            })

    # --------------------------------------------------------
    # Calculate comparison statistics
    # --------------------------------------------------------

    comparison_df = pd.DataFrame(
        comparison_rows
    )

    feature_comparison = []

    if not comparison_df.empty:

        for feature in REALTIME_FEATURES:

            feature_rows = comparison_df[
                comparison_df["feature"]
                == feature
            ]

            if feature_rows.empty:

                continue

            matches = int(
                feature_rows[
                    "exact_match"
                ].sum()
            )

            total = len(
                feature_rows
            )

            match_rate = (
                matches / total
                if total > 0
                else 0.0
            )

            feature_comparison.append({

                "feature":
                    feature,

                "samples_compared":
                    total,

                "exact_matches":
                    matches,

                "match_rate":
                    match_rate,
            })

    # --------------------------------------------------------
    # Print comparison
    # --------------------------------------------------------

    comparison_result_df = pd.DataFrame(
        feature_comparison
    )

    if not comparison_result_df.empty:

        print(
            "\nFeature reproduction results:"
        )

        print(
            comparison_result_df.to_string(
                index=False
            )
        )

    else:

        print(
            "\nNo comparable feature values "
            "were produced."
        )

    # ========================================================
    # SAVE REPORT
    # ========================================================

    os.makedirs(
        os.path.dirname(
            REPORT_FILE
        ),
        exist_ok=True
    )

    report = {

        "dataset_samples":
            int(len(df)),

        "sample_size":
            int(sample_size),

        "expected_realtime_features":
            REALTIME_FEATURES,

        "expected_feature_count":
            len(REALTIME_FEATURES),

        "url_tests":
            results,

        "feature_reproduction":
            feature_comparison,
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

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    passed_tests = sum(
        1
        for result in results
        if result.get(
            "success",
            False
        )
    )

    failed_tests = (
        len(results)
        - passed_tests
    )

    print("\n")
    print("=" * 70)
    print("AUDIT SUMMARY")
    print("=" * 70)

    print(
        f"URL tests : {len(results)}"
    )

    print(
        f"Passed    : {passed_tests}"
    )

    print(
        f"Failed    : {failed_tests}"
    )

    if failed_tests == 0:

        print(
            "\nAll representative URL "
            "extraction tests passed."
        )

    else:

        print(
            "\nSome extraction tests failed."
        )

    print(
        "\nAudit report saved to:"
    )

    print(
        REPORT_FILE
    )

    print("\n")
    print("=" * 70)
    print("REAL-TIME URL FEATURE AUDIT COMPLETE")
    print("=" * 70)

    print(
        "\nProduction URL model remains unchanged."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
