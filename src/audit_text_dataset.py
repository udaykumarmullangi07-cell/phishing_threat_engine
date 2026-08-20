import os
import re
import json

import pandas as pd

from collections import Counter


# =========================================================
# CONFIGURATION
# =========================================================

INPUT_FILE = "data/cleaned_text.csv"

REPORT_FILE = (
    "reports/text_dataset_audit.json"
)

SHORT_TEXT_THRESHOLD = 10

VERY_SHORT_TEXT_THRESHOLD = 5

DUPLICATE_PREVIEW_COUNT = 10


# =========================================================
# TEXT NORMALIZATION
# =========================================================

def normalize_text(text):

    if not isinstance(text, str):
        return ""

    text = text.lower()

    # Normalize whitespace
    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


# =========================================================
# MALFORMED TOKEN DETECTION
# =========================================================

def is_suspicious_token(token):

    if len(token) < 6:
        return False

    # Extremely repetitive characters
    unique_chars = len(set(token))

    if unique_chars <= 2:
        return True

    # Same character repeated many times
    if re.search(
        r"(.)\1{5,}",
        token,
    ):
        return True

    return False


def find_malformed_tokens(text):

    if not isinstance(text, str):
        return []

    tokens = re.findall(
        r"\b[a-zA-Z0-9]+\b",
        text,
    )

    suspicious = []

    for token in tokens:

        if is_suspicious_token(token):

            suspicious.append(token)

    return suspicious


# =========================================================
# TEXT LENGTH STATISTICS
# =========================================================

def length_statistics(series):

    lengths = (
        series
        .fillna("")
        .astype(str)
        .str.len()
    )

    return {

        "count":
            int(len(lengths)),

        "minimum":
            int(lengths.min()),

        "maximum":
            int(lengths.max()),

        "mean":
            float(lengths.mean()),

        "median":
            float(lengths.median()),

        "std":
            float(lengths.std()),

        "short_count":
            int(
                (lengths < SHORT_TEXT_THRESHOLD)
                .sum()
            ),

        "very_short_count":
            int(
                (
                    lengths
                    <
                    VERY_SHORT_TEXT_THRESHOLD
                ).sum()
            ),
    }


# =========================================================
# CLASS STATISTICS
# =========================================================

def class_statistics(df):

    result = {}

    for label in sorted(
        df["label"].unique()
    ):

        subset = df[
            df["label"] == label
        ]

        lengths = (
            subset["processed_text"]
            .fillna("")
            .astype(str)
            .str.len()
        )

        result[str(label)] = {

            "samples":
                int(len(subset)),

            "percentage":
                float(
                    len(subset)
                    /
                    len(df)
                    *
                    100
                ),

            "mean_length":
                float(lengths.mean()),

            "median_length":
                float(lengths.median()),

            "minimum_length":
                int(lengths.min()),

            "maximum_length":
                int(lengths.max()),

            "short_messages":
                int(
                    (
                        lengths
                        <
                        SHORT_TEXT_THRESHOLD
                    ).sum()
                ),

            "very_short_messages":
                int(
                    (
                        lengths
                        <
                        VERY_SHORT_TEXT_THRESHOLD
                    ).sum()
                ),
        }

    return result


# =========================================================
# DUPLICATE ANALYSIS
# =========================================================

def duplicate_analysis(df):

    text_series = (
        df["processed_text"]
        .fillna("")
        .astype(str)
        .map(normalize_text)
    )

    duplicate_mask = (
        text_series
        .duplicated(
            keep=False
        )
    )

    duplicate_rows = (
        int(
            duplicate_mask.sum()
        )
    )

    unique_duplicate_texts = (
        int(
            text_series[
                duplicate_mask
            ].nunique()
        )
    )

    duplicate_groups = (
        text_series.value_counts()
    )

    duplicate_groups = (
        duplicate_groups[
            duplicate_groups > 1
        ]
    )

    top_duplicate_groups = []

    for text, count in (
        duplicate_groups
        .head(DUPLICATE_PREVIEW_COUNT)
        .items()
    ):

        top_duplicate_groups.append(
            {
                "count":
                    int(count),

                "text":
                    text[:500],
            }
        )

    return {

        "duplicate_rows":
            duplicate_rows,

        "unique_duplicate_texts":
            unique_duplicate_texts,

        "top_duplicate_groups":
            top_duplicate_groups,
    }


# =========================================================
# CROSS-CLASS DUPLICATE ANALYSIS
# =========================================================

def cross_class_duplicate_analysis(df):

    temp = df[
        [
            "processed_text",
            "label",
        ]
    ].copy()

    temp["normalized_text"] = (
        temp["processed_text"]
        .fillna("")
        .astype(str)
        .map(normalize_text)
    )

    grouped = (
        temp
        .groupby("normalized_text")[
            "label"
        ]
        .agg(
            lambda values:
                sorted(
                    set(
                        values.tolist()
                    )
                )
        )
    )

    cross_class = grouped[
        grouped.map(
            lambda labels:
                len(labels) > 1
        )
    ]

    examples = []

    for text, labels in (
        cross_class
        .head(DUPLICATE_PREVIEW_COUNT)
        .items()
    ):

        examples.append(
            {
                "labels":
                    labels,

                "text":
                    text[:500],
            }
        )

    return {

        "cross_class_duplicate_groups":
            int(len(cross_class)),

        "examples":
            examples,
    }


# =========================================================
# MALFORMED TEXT ANALYSIS
# =========================================================

def malformed_text_analysis(df):

    suspicious_rows = []

    token_counter = Counter()

    for index, text in (
        df["processed_text"]
        .fillna("")
        .astype(str)
        .items()
    ):

        tokens = (
            find_malformed_tokens(
                text
            )
        )

        if tokens:

            suspicious_rows.append(
                {
                    "index":
                        int(index),

                    "label":
                        int(
                            df.loc[
                                index,
                                "label"
                            ]
                        ),

                    "tokens":
                        tokens[:20],

                    "text":
                        text[:500],
                }
            )

            token_counter.update(
                tokens
            )

    label_counts = (
        Counter(
            row["label"]
            for row
            in suspicious_rows
        )
    )

    return {

        "rows_with_malformed_tokens":
            int(
                len(suspicious_rows)
            ),

        "legitimate_rows":
            int(
                label_counts.get(
                    0,
                    0
                )
            ),

        "phishing_rows":
            int(
                label_counts.get(
                    1,
                    0
                )
            ),

        "top_malformed_tokens": [

            {
                "token":
                    token,

                "count":
                    int(count),
            }

            for token, count
            in token_counter
                .most_common(30)
        ],

        "examples":
            suspicious_rows[:20],
    }


# =========================================================
# URL / EMAIL-LIKE CONTENT ANALYSIS
# =========================================================

def content_pattern_analysis(df):

    series = (
        df["processed_text"]
        .fillna("")
        .astype(str)
        .str.lower()
    )

    patterns = {

        "contains_url":
            r"https?://|www\.",

        "contains_email":
            r"\b[\w.+-]+@[\w.-]+\.[a-z]{2,}\b",

        "contains_html":
            r"<[^>]+>",

        "contains_password":
            r"\bpassword\b|\bpwd\b",

        "contains_otp":
            r"\botp\b|one time password",

        "contains_click":
            r"\bclick\b",

        "contains_verify":
            r"\bverify\b|\bverification\b",

        "contains_urgent":
            r"\burgent\b|\bimmediately\b",

        "contains_money":
            r"\bpayment\b|\bmoney\b|\btransfer\b",

        "contains_unsubscribe":
            r"\bunsubscribe\b",
    }

    results = {}

    for name, pattern in patterns.items():

        mask = series.str.contains(
            pattern,
            regex=True,
            na=False,
        )

        total = int(
            mask.sum()
        )

        legitimate = int(
            (
                mask
                &
                (
                    df["label"] == 0
                )
            ).sum()
        )

        phishing = int(
            (
                mask
                &
                (
                    df["label"] == 1
                )
            ).sum()
        )

        results[name] = {

            "total":
                total,

            "legitimate":
                legitimate,

            "phishing":
                phishing,

            "legitimate_percentage":
                float(
                    legitimate
                    /
                    max(
                        int(
                            (
                                df["label"] == 0
                            ).sum()
                        ),
                        1,
                    )
                    *
                    100
                ),

            "phishing_percentage":
                float(
                    phishing
                    /
                    max(
                        int(
                            (
                                df["label"] == 1
                            ).sum()
                        ),
                        1,
                    )
                    *
                    100
                ),
        }

    return results


# =========================================================
# MAIN
# =========================================================

def main():

    print("=" * 70)

    print(
        "TEXT DATASET — QUALITY AUDIT"
    )

    print("=" * 70)

    print(
        "\nPurpose:"
    )

    print(
        "Investigate duplicates, malformed samples,"
        " short messages, class distribution,"
        " and possible data leakage."
    )

    # =====================================================
    # LOAD
    # =====================================================

    print(
        "\nLoading dataset..."
    )

    df = pd.read_csv(
        INPUT_FILE
    )

    df["processed_text"] = (
        df["processed_text"]
        .fillna("")
    )

    print(
        "Dataset shape:",
        df.shape
    )

    print(
        "Columns:",
        df.columns.tolist()
    )

    # =====================================================
    # BASIC DATA QUALITY
    # =====================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "BASIC DATA QUALITY"
    )

    print(
        "=" * 70
    )

    null_counts = (
        df.isnull()
        .sum()
    )

    print(
        "\nNull values:"
    )

    print(
        null_counts
    )

    print(
        "\nDuplicate dataframe rows:",
        int(
            df.duplicated()
            .sum()
        )
    )

    print(
        "\nLabel distribution:"
    )

    print(
        df["label"]
        .value_counts()
        .sort_index()
    )

    # =====================================================
    # CLASS STATISTICS
    # =====================================================

    class_stats = (
        class_statistics(
            df
        )
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "CLASS-SPECIFIC TEXT STATISTICS"
    )

    print(
        "=" * 70
    )

    for label, stats in (
        class_stats.items()
    ):

        class_name = (
            "LEGITIMATE"
            if label == "0"
            else "PHISHING"
        )

        print(
            f"\n{class_name}"
        )

        print(
            "Samples:",
            stats["samples"]
        )

        print(
            "Percentage:",
            f"{stats['percentage']:.2f}%"
        )

        print(
            "Mean length:",
            f"{stats['mean_length']:.2f}"
        )

        print(
            "Median length:",
            f"{stats['median_length']:.2f}"
        )

        print(
            "Minimum length:",
            stats["minimum_length"]
        )

        print(
            "Maximum length:",
            stats["maximum_length"]
        )

        print(
            "Short messages:",
            stats["short_messages"]
        )

        print(
            "Very short messages:",
            stats["very_short_messages"]
        )

    # =====================================================
    # DUPLICATES
    # =====================================================

    duplicate_stats = (
        duplicate_analysis(
            df
        )
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "DUPLICATE ANALYSIS"
    )

    print(
        "=" * 70
    )

    print(
        "Duplicate rows:",
        duplicate_stats[
            "duplicate_rows"
        ]
    )

    print(
        "Unique duplicated texts:",
        duplicate_stats[
            "unique_duplicate_texts"
        ]
    )

    print(
        "\nTop duplicate groups:"
    )

    for item in (
        duplicate_stats[
            "top_duplicate_groups"
        ]
    ):

        print(
            f"\nCount: {item['count']}"
        )

        print(
            item["text"][:300]
        )

    # =====================================================
    # CROSS CLASS DUPLICATES
    # =====================================================

    cross_class_stats = (
        cross_class_duplicate_analysis(
            df
        )
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "CROSS-CLASS DUPLICATE ANALYSIS"
    )

    print(
        "=" * 70
    )

    print(
        "Texts appearing under both labels:",
        cross_class_stats[
            "cross_class_duplicate_groups"
        ]
    )

    if cross_class_stats[
        "examples"
    ]:

        print(
            "\nExamples:"
        )

        for item in (
            cross_class_stats[
                "examples"
            ]
        ):

            print(
                "\nLabels:",
                item["labels"]
            )

            print(
                "Text:",
                item["text"][:300]
            )

    # =====================================================
    # MALFORMED TEXT
    # =====================================================

    malformed_stats = (
        malformed_text_analysis(
            df
        )
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "MALFORMED TEXT ANALYSIS"
    )

    print(
        "=" * 70
    )

    print(
        "Rows containing suspicious tokens:",
        malformed_stats[
            "rows_with_malformed_tokens"
        ]
    )

    print(
        "Legitimate:",
        malformed_stats[
            "legitimate_rows"
        ]
    )

    print(
        "Phishing:",
        malformed_stats[
            "phishing_rows"
        ]
    )

    print(
        "\nTop suspicious tokens:"
    )

    for item in (
        malformed_stats[
            "top_malformed_tokens"
        ]
    ):

        print(
            f"{item['token']:40}"
            f"{item['count']}"
        )

    # =====================================================
    # CONTENT PATTERNS
    # =====================================================

    pattern_stats = (
        content_pattern_analysis(
            df
        )
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "CONTENT PATTERN DISTRIBUTION"
    )

    print(
        "=" * 70
    )

    for name, stats in (
        pattern_stats.items()
    ):

        print(
            f"\n{name}"
        )

        print(
            "Total:",
            stats["total"]
        )

        print(
            "Legitimate:",
            stats["legitimate"],
            f"({stats['legitimate_percentage']:.2f}%)"
        )

        print(
            "Phishing:",
            stats["phishing"],
            f"({stats['phishing_percentage']:.2f}%)"
        )

    # =====================================================
    # REPORT
    # =====================================================

    report = {

        "dataset": {

            "shape":
                list(df.shape),

            "columns":
                df.columns.tolist(),

            "null_counts":
                {
                    key:
                        int(value)

                    for key, value
                    in null_counts.items()
                },

            "duplicate_dataframe_rows":
                int(
                    df.duplicated()
                    .sum()
                ),
        },

        "class_statistics":
            class_stats,

        "duplicate_analysis":
            duplicate_stats,

        "cross_class_duplicates":
            cross_class_stats,

        "malformed_text_analysis":
            malformed_stats,

        "content_pattern_analysis":
            pattern_stats,
    }

    # =====================================================
    # SAVE REPORT
    # =====================================================

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

    print(
        "\n" + "=" * 70
    )

    print(
        "DATASET QUALITY AUDIT COMPLETE"
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
