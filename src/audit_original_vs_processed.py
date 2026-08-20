import os
import re
import json

import pandas as pd


# =========================================================
# CONFIGURATION
# =========================================================

INPUT_FILE = "data/cleaned_text.csv"

REPORT_FILE = "reports/original_vs_processed_audit.json"

SAMPLE_COUNT = 5


# =========================================================
# TEXT UTILITIES
# =========================================================

def safe_text(value):
    """
    Convert dataset values safely to strings.
    """

    if pd.isna(value):
        return ""

    return str(value)


def count_tokens(text):
    """
    Count whitespace-separated tokens.
    """

    text = safe_text(text)

    return len(
        text.split()
    )


def count_words(text):
    """
    Count word-like tokens.
    """

    text = safe_text(text)

    return len(
        re.findall(
            r"\b\w+\b",
            text
        )
    )


def count_urls(text):
    """
    Count URLs before/after preprocessing.
    """

    text = safe_text(text)

    matches = re.findall(
        r"(?:https?://|www\.)[^\s]+",
        text,
        flags=re.IGNORECASE,
    )

    return len(matches)


def count_email_addresses(text):
    """
    Count email addresses.
    """

    text = safe_text(text)

    matches = re.findall(
        r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b",
        text,
    )

    return len(matches)


def count_html_tags(text):
    """
    Count HTML tags.
    """

    text = safe_text(text)

    matches = re.findall(
        r"<[^>]+>",
        text,
    )

    return len(matches)


def count_special_characters(text):
    """
    Count non-alphanumeric characters.
    """

    text = safe_text(text)

    return len(
        re.findall(
            r"[^A-Za-z0-9\s]",
            text,
        )
    )


def count_digits(text):
    """
    Count numeric characters.
    """

    text = safe_text(text)

    return len(
        re.findall(
            r"\d",
            text,
        )
    )


def count_uppercase(text):
    """
    Count uppercase alphabetic characters.
    """

    text = safe_text(text)

    return sum(
        1
        for char in text
        if char.isupper()
    )


# =========================================================
# STATISTICS
# =========================================================

def calculate_statistics(series):

    lengths = (
        series
        .fillna("")
        .astype(str)
        .str.len()
    )

    token_counts = (
        series
        .fillna("")
        .astype(str)
        .apply(count_tokens)
    )

    word_counts = (
        series
        .fillna("")
        .astype(str)
        .apply(count_words)
    )

    url_counts = (
        series
        .fillna("")
        .astype(str)
        .apply(count_urls)
    )

    email_counts = (
        series
        .fillna("")
        .astype(str)
        .apply(count_email_addresses)
    )

    html_counts = (
        series
        .fillna("")
        .astype(str)
        .apply(count_html_tags)
    )

    special_counts = (
        series
        .fillna("")
        .astype(str)
        .apply(count_special_characters)
    )

    digit_counts = (
        series
        .fillna("")
        .astype(str)
        .apply(count_digits)
    )

    uppercase_counts = (
        series
        .fillna("")
        .astype(str)
        .apply(count_uppercase)
    )

    return {

        "samples":
            int(len(series)),

        "mean_character_length":
            float(lengths.mean()),

        "median_character_length":
            float(lengths.median()),

        "minimum_character_length":
            int(lengths.min()),

        "maximum_character_length":
            int(lengths.max()),

        "mean_token_count":
            float(token_counts.mean()),

        "median_token_count":
            float(token_counts.median()),

        "mean_word_count":
            float(word_counts.mean()),

        "urls_detected":
            int(
                (url_counts > 0).sum()
            ),

        "total_urls":
            int(url_counts.sum()),

        "emails_detected":
            int(
                (email_counts > 0).sum()
            ),

        "total_email_addresses":
            int(email_counts.sum()),

        "html_detected":
            int(
                (html_counts > 0).sum()
            ),

        "total_html_tags":
            int(html_counts.sum()),

        "total_special_characters":
            int(special_counts.sum()),

        "total_digits":
            int(digit_counts.sum()),

        "total_uppercase_characters":
            int(uppercase_counts.sum()),
    }


# =========================================================
# CLASS-SPECIFIC STATISTICS
# =========================================================

def calculate_class_statistics(df, column):

    results = {}

    for label in sorted(
        df["label"].unique()
    ):

        subset = df[
            df["label"] == label
        ][column]

        stats = calculate_statistics(
            subset
        )

        class_name = (
            "LEGITIMATE"
            if label == 0
            else "PHISHING"
        )

        results[class_name] = stats

    return results


# =========================================================
# INFORMATION LOSS
# =========================================================

def calculate_information_loss(df):

    original_lengths = (
        df["text_combined"]
        .fillna("")
        .astype(str)
        .str.len()
    )

    clean_lengths = (
        df["clean_text"]
        .fillna("")
        .astype(str)
        .str.len()
    )

    processed_lengths = (
        df["processed_text"]
        .fillna("")
        .astype(str)
        .str.len()
    )

    original_tokens = (
        df["text_combined"]
        .fillna("")
        .astype(str)
        .apply(count_tokens)
    )

    clean_tokens = (
        df["clean_text"]
        .fillna("")
        .astype(str)
        .apply(count_tokens)
    )

    processed_tokens = (
        df["processed_text"]
        .fillna("")
        .astype(str)
        .apply(count_tokens)
    )

    original_mean_length = (
        original_lengths.mean()
    )

    clean_mean_length = (
        clean_lengths.mean()
    )

    processed_mean_length = (
        processed_lengths.mean()
    )

    original_mean_tokens = (
        original_tokens.mean()
    )

    clean_mean_tokens = (
        clean_tokens.mean()
    )

    processed_mean_tokens = (
        processed_tokens.mean()
    )

    return {

        "mean_character_length": {

            "original":
                float(
                    original_mean_length
                ),

            "clean":
                float(
                    clean_mean_length
                ),

            "processed":
                float(
                    processed_mean_length
                ),
        },

        "mean_token_count": {

            "original":
                float(
                    original_mean_tokens
                ),

            "clean":
                float(
                    clean_mean_tokens
                ),

            "processed":
                float(
                    processed_mean_tokens
                ),
        },

        "average_character_reduction_percent": {

            "original_to_clean":
                float(
                    (
                        1
                        -
                        clean_mean_length
                        /
                        max(
                            original_mean_length,
                            1
                        )
                    )
                    * 100
                ),

            "clean_to_processed":
                float(
                    (
                        1
                        -
                        processed_mean_length
                        /
                        max(
                            clean_mean_length,
                            1
                        )
                    )
                    * 100
                ),

            "original_to_processed":
                float(
                    (
                        1
                        -
                        processed_mean_length
                        /
                        max(
                            original_mean_length,
                            1
                        )
                    )
                    * 100
                ),
        },

        "average_token_reduction_percent": {

            "original_to_clean":
                float(
                    (
                        1
                        -
                        clean_mean_tokens
                        /
                        max(
                            original_mean_tokens,
                            1
                        )
                    )
                    * 100
                ),

            "clean_to_processed":
                float(
                    (
                        1
                        -
                        processed_mean_tokens
                        /
                        max(
                            clean_mean_tokens,
                            1
                        )
                    )
                    * 100
                ),

            "original_to_processed":
                float(
                    (
                        1
                        -
                        processed_mean_tokens
                        /
                        max(
                            original_mean_tokens,
                            1
                        )
                    )
                    * 100
                ),
        },
    }


# =========================================================
# INFORMATION PRESERVATION
# =========================================================

def calculate_preservation(df):

    columns = [
        "text_combined",
        "clean_text",
        "processed_text",
    ]

    results = {}

    for column in columns:

        series = (
            df[column]
            .fillna("")
            .astype(str)
        )

        results[column] = {

            "rows_with_urls":
                int(
                    series.apply(
                        count_urls
                    ).gt(0).sum()
                ),

            "rows_with_email_addresses":
                int(
                    series.apply(
                        count_email_addresses
                    ).gt(0).sum()
                ),

            "rows_with_html":
                int(
                    series.apply(
                        count_html_tags
                    ).gt(0).sum()
                ),

            "total_urls":
                int(
                    series.apply(
                        count_urls
                    ).sum()
                ),

            "total_email_addresses":
                int(
                    series.apply(
                        count_email_addresses
                    ).sum()
                ),

            "total_html_tags":
                int(
                    series.apply(
                        count_html_tags
                    ).sum()
                ),
        }

    return results


# =========================================================
# SAMPLE TRANSFORMATION
# =========================================================

def generate_samples(df):

    samples = []

    sample_indices = (
        df.index[
            df["text_combined"]
            .fillna("")
            .astype(str)
            .str.len()
            .sort_values(
                ascending=False
            )
            .index
        ][:SAMPLE_COUNT]
    )

    for index in sample_indices:

        row = df.loc[index]

        samples.append({

            "index":
                int(index),

            "label":
                int(row["label"]),

            "class":
                (
                    "LEGITIMATE"
                    if row["label"] == 0
                    else "PHISHING"
                ),

            "original":
                safe_text(
                    row["text_combined"]
                )[:3000],

            "clean":
                safe_text(
                    row["clean_text"]
                )[:3000],

            "processed":
                safe_text(
                    row["processed_text"]
                )[:3000],
        })

    return samples


# =========================================================
# SPECIFIC FEATURE LOSS CHECK
# =========================================================

def feature_loss_analysis(df):

    results = {}

    original = (
        df["text_combined"]
        .fillna("")
        .astype(str)
    )

    clean = (
        df["clean_text"]
        .fillna("")
        .astype(str)
    )

    processed = (
        df["processed_text"]
        .fillna("")
        .astype(str)
    )

    feature_tests = {

        "URLs":

            count_urls,

        "Email addresses":

            count_email_addresses,

        "HTML tags":

            count_html_tags,

        "Special characters":

            count_special_characters,

        "Digits":

            count_digits,

        "Uppercase characters":

            count_uppercase,
    }

    for name, function in (
        feature_tests.items()
    ):

        original_count = int(
            original.apply(
                function
            ).sum()
        )

        clean_count = int(
            clean.apply(
                function
            ).sum()
        )

        processed_count = int(
            processed.apply(
                function
            ).sum()
        )

        results[name] = {

            "original":
                original_count,

            "clean":
                clean_count,

            "processed":
                processed_count,

            "clean_retention_percent":
                float(
                    clean_count
                    /
                    max(
                        original_count,
                        1
                    )
                    * 100
                ),

            "processed_retention_percent":
                float(
                    processed_count
                    /
                    max(
                        original_count,
                        1
                    )
                    * 100
                ),
        }

    return results


# =========================================================
# MAIN
# =========================================================

def main():

    print("=" * 70)

    print(
        "ORIGINAL vs CLEAN vs PROCESSED TEXT AUDIT"
    )

    print("=" * 70)

    print(
        "\nPurpose:"
    )

    print(
        "Determine how much information is removed"
        " during the text preprocessing pipeline."
    )

    # =====================================================
    # LOAD DATASET
    # =====================================================

    print(
        "\nLoading dataset..."
    )

    df = pd.read_csv(
        INPUT_FILE
    )

    required_columns = [
        "text_combined",
        "label",
        "clean_text",
        "processed_text",
    ]

    missing_columns = [
        column
        for column
        in required_columns
        if column not in df.columns
    ]

    if missing_columns:

        raise ValueError(
            "Missing required columns: "
            + str(
                missing_columns
            )
        )

    print(
        "Dataset shape:",
        df.shape
    )

    # =====================================================
    # BASIC STATISTICS
    # =====================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "PIPELINE STATISTICS"
    )

    print(
        "=" * 70
    )

    pipeline_stats = {}

    for column in [
        "text_combined",
        "clean_text",
        "processed_text",
    ]:

        print(
            f"\n{column}"
        )

        stats = calculate_statistics(
            df[column]
        )

        pipeline_stats[column] = stats

        print(
            "Samples:",
            stats["samples"]
        )

        print(
            "Mean characters:",
            f"{stats['mean_character_length']:.2f}"
        )

        print(
            "Median characters:",
            f"{stats['median_character_length']:.2f}"
        )

        print(
            "Minimum characters:",
            stats["minimum_character_length"]
        )

        print(
            "Maximum characters:",
            stats["maximum_character_length"]
        )

        print(
            "Mean tokens:",
            f"{stats['mean_token_count']:.2f}"
        )

        print(
            "Rows with URLs:",
            stats["urls_detected"]
        )

        print(
            "Rows with email addresses:",
            stats["emails_detected"]
        )

        print(
            "Rows with HTML:",
            stats["html_detected"]
        )

    # =====================================================
    # INFORMATION LOSS
    # =====================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "INFORMATION LOSS"
    )

    print(
        "=" * 70
    )

    information_loss = (
        calculate_information_loss(
            df
        )
    )

    for metric, values in (
        information_loss.items()
    ):

        print(
            f"\n{metric}"
        )

        for stage, value in (
            values.items()
        ):

            if isinstance(
                value,
                dict
            ):

                print(
                    f"  {stage}:"
                )

                for substage, percentage in (
                    value.items()
                ):

                    print(
                        f"    {substage}: "
                        f"{percentage:.2f}%"
                    )

            else:

                print(
                    f"  {stage}: "
                    f"{value:.2f}"
                )

    # =====================================================
    # FEATURE PRESERVATION
    # =====================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "FEATURE PRESERVATION"
    )

    print(
        "=" * 70
    )

    preservation = (
        calculate_preservation(
            df
        )
    )

    for column, values in (
        preservation.items()
    ):

        print(
            f"\n{column}"
        )

        print(
            "Rows with URLs:",
            values[
                "rows_with_urls"
            ]
        )

        print(
            "Rows with emails:",
            values[
                "rows_with_email_addresses"
            ]
        )

        print(
            "Rows with HTML:",
            values[
                "rows_with_html"
            ]
        )

        print(
            "Total URLs:",
            values[
                "total_urls"
            ]
        )

        print(
            "Total email addresses:",
            values[
                "total_email_addresses"
            ]
        )

        print(
            "Total HTML tags:",
            values[
                "total_html_tags"
            ]
        )

    # =====================================================
    # FEATURE LOSS
    # =====================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "SPECIFIC FEATURE RETENTION"
    )

    print(
        "=" * 70
    )

    feature_loss = (
        feature_loss_analysis(
            df
        )
    )

    for feature, values in (
        feature_loss.items()
    ):

        print(
            f"\n{feature}"
        )

        print(
            "Original:",
            values["original"]
        )

        print(
            "Clean:",
            values["clean"]
        )

        print(
            "Processed:",
            values["processed"]
        )

        print(
            "Processed retention:",
            f"{values['processed_retention_percent']:.2f}%"
        )

    # =====================================================
    # CLASS-SPECIFIC COMPARISON
    # =====================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "CLASS-SPECIFIC PIPELINE ANALYSIS"
    )

    print(
        "=" * 70
    )

    class_statistics = {}

    for column in [
        "text_combined",
        "clean_text",
        "processed_text",
    ]:

        class_statistics[column] = (
            calculate_class_statistics(
                df,
                column,
            )
        )

        print(
            f"\n--- {column} ---"
        )

        for class_name, stats in (
            class_statistics[
                column
            ].items()
        ):

            print(
                f"{class_name}: "
                f"mean length="
                f"{stats['mean_character_length']:.2f}, "
                f"mean tokens="
                f"{stats['mean_token_count']:.2f}"
            )

    # =====================================================
    # SAMPLE TRANSFORMATIONS
    # =====================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "EXTREME SAMPLE TRANSFORMATIONS"
    )

    print(
        "=" * 70
    )

    samples = generate_samples(
        df
    )

    for sample in samples:

        print(
            "\n" + "-" * 70
        )

        print(
            "Index:",
            sample["index"]
        )

        print(
            "Class:",
            sample["class"]
        )

        print(
            "\nORIGINAL:"
        )

        print(
            sample["original"][:1000]
        )

        print(
            "\nCLEAN:"
        )

        print(
            sample["clean"][:1000]
        )

        print(
            "\nPROCESSED:"
        )

        print(
            sample["processed"][:1000]
        )

    # =====================================================
    # FINAL REPORT
    # =====================================================

    report = {

        "dataset": {

            "shape":
                list(
                    df.shape
                ),

            "columns":
                df.columns.tolist(),
        },

        "pipeline_statistics":
            pipeline_stats,

        "information_loss":
            information_loss,

        "feature_preservation":
            preservation,

        "feature_loss":
            feature_loss,

        "class_statistics":
            class_statistics,

        "extreme_samples":
            samples,
    }

    os.makedirs(
        "reports",
        exist_ok=True,
    )

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

    # =====================================================
    # COMPLETE
    # =====================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "ORIGINAL vs PROCESSED AUDIT COMPLETE"
    )

    print(
        "=" * 70
    )

    print(
        "\nDetailed report saved to:"
    )

    print(
        REPORT_FILE
    )

    print(
        "\nProduction model remains unchanged."
    )

    print(
        "=" * 70
    )


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":

    main()
