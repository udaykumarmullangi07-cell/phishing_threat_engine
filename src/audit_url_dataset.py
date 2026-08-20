import json
import os
import re

import pandas as pd


INPUT_FILE = "data/url_dataset.csv"
OUTPUT_FILE = "reports/url_dataset_audit.json"


def main():

    print("=" * 70)
    print("URL DATASET — QUALITY AUDIT")
    print("=" * 70)

    print(
        """
Purpose:
Investigate URL dataset quality before machine-learning training.

Checks:
- Dataset dimensions
- Missing values
- Class distribution
- Duplicate URLs
- Cross-label duplicates
- URL length statistics
- HTTPS distribution
- IP-address usage
- Special-character distribution
- Digit ratio
- Suspicious URL patterns
"""
    )

    # ---------------------------------------------------------
    # 1. LOAD DATASET
    # ---------------------------------------------------------

    print("\nLoading dataset...")

    df = pd.read_csv(INPUT_FILE)

    print(
        f"Dataset shape: {df.shape}"
    )

    print(
        f"Columns: {len(df.columns)}"
    )

    # ---------------------------------------------------------
    # 2. BASIC QUALITY
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("BASIC DATA QUALITY")
    print("=" * 70)

    null_counts = df.isnull().sum()

    print("\nNull values:")

    print(
        null_counts[
            null_counts > 0
        ]
        if null_counts.sum() > 0
        else "No null values"
    )

    duplicate_rows = int(
        df.duplicated().sum()
    )

    duplicate_urls = int(
        df["url"].duplicated().sum()
    )

    print(
        f"\nDuplicate rows : {duplicate_rows}"
    )

    print(
        f"Duplicate URLs : {duplicate_urls}"
    )

    # ---------------------------------------------------------
    # 3. LABEL DISTRIBUTION
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("CLASS DISTRIBUTION")
    print("=" * 70)

    label_counts = (
        df["status"]
        .value_counts()
    )

    label_percentages = (
        df["status"]
        .value_counts(
            normalize=True
        )
        * 100
    )

    print(label_counts)

    print("\nPercentages:")

    for label in label_counts.index:

        print(
            f"{label:15} "
            f"{label_counts[label]:6} "
            f"({label_percentages[label]:.2f}%)"
        )

    # ---------------------------------------------------------
    # 4. CROSS-LABEL DUPLICATES
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("CROSS-LABEL DUPLICATE ANALYSIS")
    print("=" * 70)

    label_per_url = (
        df.groupby("url")["status"]
        .nunique()
    )

    cross_label_urls = (
        label_per_url[
            label_per_url > 1
        ]
    )

    print(
        "URLs appearing under multiple labels:",
        len(cross_label_urls),
    )

    if len(cross_label_urls) > 0:

        print(
            "\nPotential conflicting URLs:"
        )

        for url in cross_label_urls.index[:20]:

            print(
                url
            )

    else:

        print(
            "No cross-label duplicate URLs detected."
        )

    # ---------------------------------------------------------
    # 5. URL LENGTH
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("URL LENGTH ANALYSIS")
    print("=" * 70)

    length_stats = (
        df.groupby("status")["length_url"]
        .agg(
            [
                "count",
                "mean",
                "median",
                "min",
                "max",
            ]
        )
    )

    print(
        length_stats
    )

    # ---------------------------------------------------------
    # 6. ORIGINAL FEATURE DISTRIBUTIONS
    # ---------------------------------------------------------

    feature_columns = [
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
        "domain_in_brand",
        "brand_in_subdomain",
        "brand_in_path",
        "suspecious_tld",
        "statistical_report",
        "nb_hyperlinks",
        "ratio_intHyperlinks",
        "ratio_extHyperlinks",
        "ratio_nullHyperlinks",
        "nb_extCSS",
        "ratio_intRedirection",
        "ratio_extRedirection",
        "ratio_intErrors",
        "ratio_extErrors",
        "login_form",
        "external_favicon",
        "links_in_tags",
        "submit_email",
        "ratio_intMedia",
        "ratio_extMedia",
        "sfh",
        "iframe",
        "popup_window",
        "safe_anchor",
        "onmouseover",
        "right_clic",
        "empty_title",
        "domain_in_title",
        "domain_with_copyright",
        "whois_registered_domain",
        "domain_registration_length",
        "domain_age",
        "web_traffic",
        "dns_record",
        "google_index",
        "page_rank",
    ]

    available_features = [
        col
        for col in feature_columns
        if col in df.columns
    ]

    print("\n" + "=" * 70)
    print("ORIGINAL DATASET FEATURE SUMMARY")
    print("=" * 70)

    feature_summary = []

    for feature in available_features:

        series = df[feature]

        feature_summary.append(
            {
                "feature": feature,
                "mean": float(series.mean()),
                "std": float(series.std()),
                "min": float(series.min()),
                "max": float(series.max()),
                "unique_values": int(
                    series.nunique()
                ),
            }
        )

    feature_summary_df = pd.DataFrame(
        feature_summary
    )

    print(
        feature_summary_df.to_string(
            index=False
        )
    )

    # ---------------------------------------------------------
    # 7. OUR TEN FEATURES
    # ---------------------------------------------------------

    engineered_file = (
        "data/url_features.csv"
    )

    engineered_df = pd.read_csv(
        engineered_file
    )

    engineered_features = [
        "url_length",
        "dot_count",
        "has_https",
        "has_ip_address",
        "special_chars",
        "subdomain_count",
        "path_depth",
        "digit_ratio",
        "has_at_symbol",
        "domain_length",
    ]

    print("\n" + "=" * 70)
    print("ENGINEERED 10-FEATURE DATASET")
    print("=" * 70)

    print(
        "Rows:",
        len(engineered_df),
    )

    print(
        "Features:",
        engineered_features,
    )

    # ---------------------------------------------------------
    # 8. ENGINEERED FEATURE CLASS STATISTICS
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("ENGINEERED FEATURE CLASS ANALYSIS")
    print("=" * 70)

    class_feature_stats = {}

    for label in [
        "legitimate",
        "phishing",
    ]:

        subset = engineered_df[
            engineered_df["status"]
            == label
        ]

        print(
            f"\n--- {label.upper()} ---"
        )

        class_feature_stats[label] = {}

        for feature in engineered_features:

            mean_value = float(
                subset[feature].mean()
            )

            median_value = float(
                subset[feature].median()
            )

            print(
                f"{feature:20} "
                f"mean={mean_value:.4f} "
                f"median={median_value:.4f}"
            )

            class_feature_stats[
                label
            ][feature] = {
                "mean": mean_value,
                "median": median_value,
            }

    # ---------------------------------------------------------
    # 9. SUSPICIOUS PATTERNS
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("SUSPICIOUS URL PATTERN ANALYSIS")
    print("=" * 70)

    url_series = (
        df["url"]
        .astype(str)
    )

    patterns = {

        "uses_https":
            r"^https://",

        "uses_ip_address":
            r"^\w+://\d+\.\d+\.\d+\.\d+",

        "contains_at_symbol":
            r"@",

        "contains_question_mark":
            r"\?",

        "contains_percent_encoding":
            r"%[0-9a-fA-F]{2}",

        "contains_many_digits":
            r"\d{5,}",

        "contains_double_slash_path":
            r"//",

    }

    pattern_results = {}

    for name, pattern in patterns.items():

        matches = (
            url_series
            .str.contains(
                pattern,
                regex=True,
                na=False,
            )
        )

        total = int(
            matches.sum()
        )

        legitimate = int(
            (
                matches
                & (
                    df["status"]
                    == "legitimate"
                )
            ).sum()
        )

        phishing = int(
            (
                matches
                & (
                    df["status"]
                    == "phishing"
                )
            ).sum()
        )

        print(
            f"{name:35}"
            f" total={total:5} "
            f"legitimate={legitimate:5} "
            f"phishing={phishing:5}"
        )

        pattern_results[name] = {
            "total": total,
            "legitimate": legitimate,
            "phishing": phishing,
        }

    # ---------------------------------------------------------
    # 10. SAVE REPORT
    # ---------------------------------------------------------

    os.makedirs(
        "reports",
        exist_ok=True,
    )

    report = {

        "dataset": {
            "rows": int(len(df)),
            "columns": int(len(df.columns)),
        },

        "null_values":
            null_counts.to_dict(),

        "duplicate_rows":
            duplicate_rows,

        "duplicate_urls":
            duplicate_urls,

        "label_distribution":
            label_counts.to_dict(),

        "cross_label_duplicate_urls":
            int(len(cross_label_urls)),

        "url_length_statistics":
            length_stats.to_dict(),

        "original_feature_summary":
            feature_summary,

        "engineered_features":
            engineered_features,

        "engineered_feature_class_statistics":
            class_feature_stats,

        "suspicious_patterns":
            pattern_results,
    }

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            report,
            f,
            indent=4,
        )

    print("\n" + "=" * 70)
    print("URL DATASET AUDIT COMPLETE")
    print("=" * 70)

    print(
        "\nDetailed report saved to:"
    )

    print(
        OUTPUT_FILE
    )


if __name__ == "__main__":
    main()

