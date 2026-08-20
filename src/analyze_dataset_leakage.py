import json
import os
import re
from collections import Counter

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = "data/cleaned_text.csv"
REPORT_DIR = "reports"
REPORT_FILE = os.path.join(
    REPORT_DIR,
    "text_dataset_leakage_analysis.json",
)

RANDOM_STATE = 42
TEST_SIZE = 0.20

# Similarity threshold used only for investigation.
# We are NOT modifying the production model.
NEAR_DUPLICATE_THRESHOLD = 0.90

# Maximum number of samples used for pairwise similarity.
# This protects the laptop from an enormous O(n²) calculation.
MAX_SIMILARITY_SAMPLES = 10000


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def normalize_for_template_analysis(text):
    """
    Normalize text aggressively so that messages which differ
    only by numbers, dates, IDs, URLs, or email-like tokens
    can be recognized as possible templates.
    """

    text = str(text).lower()

    # Replace URL-like structures.
    text = re.sub(
        r"(https?://\S+|www\.\S+)",
        " URL ",
        text,
    )

    # Replace email-like structures.
    text = re.sub(
        r"\b[\w.+-]+@[\w.-]+\.\w+\b",
        " EMAIL ",
        text,
    )

    # Replace long hexadecimal / identifier-like strings.
    text = re.sub(
        r"\b[a-f0-9]{8,}\b",
        " ID ",
        text,
    )

    # Replace numbers.
    text = re.sub(
        r"\b\d+\b",
        " NUM ",
        text,
    )

    # Collapse whitespace.
    text = re.sub(r"\s+", " ", text).strip()

    return text


def text_hash(text):
    """
    Simple deterministic normalized representation.
    """
    return normalize_for_template_analysis(text)


def print_section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# ============================================================
# MAIN ANALYSIS
# ============================================================

def main():

    print("=" * 70)
    print("TEXT DATASET — LEAKAGE & TEMPLATE ANALYSIS")
    print("=" * 70)

    print("\nPurpose:")
    print("Investigate duplicate leakage, repeated templates,")
    print("train/test contamination, and dataset-specific artifacts.")

    print("\nProduction model will NOT be modified.")

    # --------------------------------------------------------
    # 1. LOAD DATASET
    # --------------------------------------------------------

    print("\nLoading dataset...")

    df = pd.read_csv(INPUT_FILE)

    df["processed_text"] = df["processed_text"].fillna("")
    df["label"] = df["label"].astype(int)

    print(f"Dataset shape: {df.shape}")

    # --------------------------------------------------------
    # 2. BASIC STATISTICS
    # --------------------------------------------------------

    print_section("BASIC DATASET STATISTICS")

    total_samples = len(df)

    label_counts = df["label"].value_counts().sort_index()

    print(f"Total samples: {total_samples}")

    print("\nLabel distribution:")

    for label, count in label_counts.items():

        name = (
            "LEGITIMATE"
            if label == 0
            else "PHISHING"
        )

        percentage = count / total_samples * 100

        print(
            f"{name:12}: "
            f"{count:6} "
            f"({percentage:.2f}%)"
        )

    # --------------------------------------------------------
    # 3. EXACT DUPLICATE ANALYSIS
    # --------------------------------------------------------

    print_section("EXACT DUPLICATE ANALYSIS")

    text_series = df["processed_text"]

    duplicate_mask = text_series.duplicated(
        keep=False
    )

    duplicate_rows = int(
        duplicate_mask.sum()
    )

    duplicate_groups = int(
        text_series[duplicate_mask].nunique()
    )

    print(
        f"Rows belonging to duplicate groups: "
        f"{duplicate_rows}"
    )

    print(
        f"Unique duplicated texts: "
        f"{duplicate_groups}"
    )

    duplicate_group_sizes = (
        text_series[duplicate_mask]
        .value_counts()
        .head(20)
    )

    print("\nLargest duplicate groups:")

    for text, count in duplicate_group_sizes.items():

        preview = text[:150]

        print("\nCount:", count)
        print(preview)

    # --------------------------------------------------------
    # 4. CROSS-LABEL DUPLICATES
    # --------------------------------------------------------

    print_section("CROSS-LABEL DUPLICATE ANALYSIS")

    grouped_labels = (
        df.groupby("processed_text")["label"]
        .nunique()
    )

    cross_label_texts = grouped_labels[
        grouped_labels > 1
    ]

    cross_label_count = len(
        cross_label_texts
    )

    print(
        f"Texts appearing under both labels: "
        f"{cross_label_count}"
    )

    if cross_label_count == 0:

        print(
            "No exact cross-label duplicates detected."
        )

    else:

        print(
            "\nWARNING:"
            " exact duplicate text appears under"
            " multiple labels."
        )

        examples = (
            cross_label_texts
            .head(10)
            .index
        )

        for text in examples:
            print("\n", text[:300])

    # --------------------------------------------------------
    # 5. NORMALIZED TEMPLATE ANALYSIS
    # --------------------------------------------------------

    print_section("NORMALIZED TEMPLATE ANALYSIS")

    print(
        "Creating normalized template representations..."
    )

    df["template_text"] = (
        df["processed_text"]
        .apply(normalize_for_template_analysis)
    )

    template_counts = (
        df["template_text"]
        .value_counts()
    )

    repeated_templates = (
        template_counts[
            template_counts > 1
        ]
        .head(50)
    )

    print(
        f"Repeated normalized templates: "
        f"{len(repeated_templates)}"
    )

    print("\nLargest template groups:")

    for template, count in repeated_templates.head(20).items():

        print("\nCount:", count)
        print(template[:250])

    # --------------------------------------------------------
    # 6. TEMPLATE DISTRIBUTION BY LABEL
    # --------------------------------------------------------

    print_section(
        "TEMPLATE DISTRIBUTION BY LABEL"
    )

    template_label_table = (
        df.groupby(
            ["template_text", "label"]
        )
        .size()
        .unstack(fill_value=0)
    )

    if 0 not in template_label_table.columns:
        template_label_table[0] = 0

    if 1 not in template_label_table.columns:
        template_label_table[1] = 0

    template_label_table["total"] = (
        template_label_table[0]
        + template_label_table[1]
    )

    template_label_table = (
        template_label_table
        .sort_values(
            "total",
            ascending=False,
        )
    )

    print(
        "\nTop repeated templates with labels:"
    )

    for template, row in (
        template_label_table
        .head(20)
        .iterrows()
    ):

        print("\nTemplate:")
        print(template[:200])

        print(
            f"Legitimate: {int(row[0])}"
        )

        print(
            f"Phishing:   {int(row[1])}"
        )

        print(
            f"Total:      {int(row['total'])}"
        )

    # --------------------------------------------------------
    # 7. RANDOM TRAIN/TEST SPLIT
    # --------------------------------------------------------

    print_section(
        "RANDOM TRAIN/TEST SPLIT"
    )

    indices = df.index

    train_indices, test_indices = train_test_split(
        indices,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df["label"],
    )

    train_set = set(train_indices)
    test_set = set(test_indices)

    print(
        f"Training samples: {len(train_set)}"
    )

    print(
        f"Testing samples : {len(test_set)}"
    )

    # --------------------------------------------------------
    # 8. EXACT TRAIN/TEST CONTAMINATION
    # --------------------------------------------------------

    print_section(
        "EXACT TRAIN/TEST CONTAMINATION"
    )

    train_texts = set(
        df.loc[
            train_indices,
            "processed_text",
        ]
    )

    test_texts = set(
        df.loc[
            test_indices,
            "processed_text",
        ]
    )

    exact_overlap = (
        train_texts.intersection(
            test_texts
        )
    )

    print(
        f"Unique train texts: "
        f"{len(train_texts)}"
    )

    print(
        f"Unique test texts:  "
        f"{len(test_texts)}"
    )

    print(
        f"Exact text overlap: "
        f"{len(exact_overlap)}"
    )

    if exact_overlap:

        print(
            "\nWARNING:"
            " Exact text appears in both"
            " training and testing data."
        )

        for text in list(
            exact_overlap
        )[:10]:

            print("\n", text[:250])

    else:

        print(
            "No exact text contamination detected."
        )

    # --------------------------------------------------------
    # 9. TEMPLATE TRAIN/TEST CONTAMINATION
    # --------------------------------------------------------

    print_section(
        "TEMPLATE TRAIN/TEST CONTAMINATION"
    )

    train_templates = set(
        df.loc[
            train_indices,
            "template_text",
        ]
    )

    test_templates = set(
        df.loc[
            test_indices,
            "template_text",
        ]
    )

    template_overlap = (
        train_templates.intersection(
            test_templates
        )
    )

    print(
        f"Unique train templates: "
        f"{len(train_templates)}"
    )

    print(
        f"Unique test templates:  "
        f"{len(test_templates)}"
    )

    print(
        f"Template overlap: "
        f"{len(template_overlap)}"
    )

    template_overlap_percentage = (
        len(template_overlap)
        / max(len(test_templates), 1)
        * 100
    )

    print(
        f"Test templates also present in "
        f"training: "
        f"{template_overlap_percentage:.2f}%"
    )

    if template_overlap:

        print(
            "\nTop overlapping templates:"
        )

        overlap_counts = (
            df[
                df["template_text"].isin(
                    template_overlap
                )
            ]["template_text"]
            .value_counts()
            .head(20)
        )

        for template, count in (
            overlap_counts.items()
        ):

            print(
                f"\nCount: {count}"
            )

            print(
                template[:250]
            )

    # --------------------------------------------------------
    # 10. MESSAGE LENGTH ANALYSIS
    # --------------------------------------------------------

    print_section(
        "MESSAGE LENGTH ANALYSIS"
    )

    df["char_length"] = (
        df["processed_text"]
        .str.len()
    )

    df["token_length"] = (
        df["processed_text"]
        .str.split()
        .str.len()
    )

    length_summary = {}

    for label in [0, 1]:

        label_name = (
            "legitimate"
            if label == 0
            else "phishing"
        )

        subset = df[
            df["label"] == label
        ]

        print(
            f"\n{label_name.upper()}"
        )

        print(
            f"Mean characters: "
            f"{subset['char_length'].mean():.2f}"
        )

        print(
            f"Median characters: "
            f"{subset['char_length'].median():.2f}"
        )

        print(
            f"Mean tokens: "
            f"{subset['token_length'].mean():.2f}"
        )

        print(
            f"Median tokens: "
            f"{subset['token_length'].median():.2f}"
        )

        length_summary[
            label_name
        ] = {
            "mean_characters":
                float(
                    subset[
                        "char_length"
                    ].mean()
                ),

            "median_characters":
                float(
                    subset[
                        "char_length"
                    ].median()
                ),

            "mean_tokens":
                float(
                    subset[
                        "token_length"
                    ].mean()
                ),

            "median_tokens":
                float(
                    subset[
                        "token_length"
                    ].median()
                ),
        }

    # --------------------------------------------------------
    # 11. VOCABULARY DISTRIBUTION
    # --------------------------------------------------------

    print_section(
        "CLASS-SPECIFIC VOCABULARY"
    )

    print(
        "Calculating common words in each class..."
    )

    class_word_counts = {}

    for label in [0, 1]:

        label_name = (
            "legitimate"
            if label == 0
            else "phishing"
        )

        counter = Counter()

        subset = df[
            df["label"] == label
        ]["processed_text"]

        for text in subset:

            words = text.split()

            counter.update(words)

        class_word_counts[
            label_name
        ] = counter

    legitimate_words = (
        class_word_counts[
            "legitimate"
        ]
        .most_common(30)
    )

    phishing_words = (
        class_word_counts[
            "phishing"
        ]
        .most_common(30)
    )

    print("\nTop legitimate vocabulary:")

    for word, count in legitimate_words:

        print(
            f"{word:30} {count}"
        )

    print("\nTop phishing vocabulary:")

    for word, count in phishing_words:

        print(
            f"{word:30} {count}"
        )

    # --------------------------------------------------------
    # 12. NEAR DUPLICATE SAMPLE ANALYSIS
    # --------------------------------------------------------

    print_section(
        "NEAR-DUPLICATE ANALYSIS"
    )

    print(
        "This analysis uses a limited sample to avoid"
        " excessive memory usage."
    )

    similarity_sample_size = min(
        MAX_SIMILARITY_SAMPLES,
        len(df),
    )

    similarity_df = df.sample(
        n=similarity_sample_size,
        random_state=RANDOM_STATE,
    ).copy()

    print(
        f"Similarity sample size: "
        f"{len(similarity_df)}"
    )

    print(
        "Creating character-level TF-IDF..."
    )

    char_vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(3, 5),
        min_df=2,
        max_features=20000,
    )

    char_matrix = char_vectorizer.fit_transform(
        similarity_df[
            "processed_text"
        ]
    )

    print(
        f"Character TF-IDF shape: "
        f"{char_matrix.shape}"
    )

    # Rather than calculating a full 10k x 10k
    # matrix, compare batches against themselves.
    #
    # We identify highly similar pairs while
    # avoiding a huge dense matrix.

    near_duplicate_pairs = []

    batch_size = 500

    for start in range(
        0,
        len(similarity_df),
        batch_size,
    ):

        end = min(
            start + batch_size,
            len(similarity_df),
        )

        batch_matrix = char_matrix[
            start:end
        ]

        similarities = cosine_similarity(
            batch_matrix,
            char_matrix,
        )

        for local_i in range(
            similarities.shape[0]
        ):

            global_i = (
                start + local_i
            )

            # Only inspect j > i
            # to avoid duplicate pairs.
            candidate_indices = (
                similarities[
                    local_i
                ]
                .argsort()[-10:]
            )

            for global_j in candidate_indices:

                if global_j <= global_i:
                    continue

                score = float(
                    similarities[
                        local_i,
                        global_j,
                    ]
                )

                if (
                    score
                    >= NEAR_DUPLICATE_THRESHOLD
                ):

                    near_duplicate_pairs.append(
                        {
                            "index_a":
                                int(
                                    similarity_df.index[
                                        global_i
                                    ]
                                ),

                            "index_b":
                                int(
                                    similarity_df.index[
                                        global_j
                                    ]
                                ),

                            "similarity":
                                score,

                            "label_a":
                                int(
                                    similarity_df.iloc[
                                        global_i
                                    ]["label"]
                                ),

                            "label_b":
                                int(
                                    similarity_df.iloc[
                                        global_j
                                    ]["label"]
                                ),
                        }
                    )

    # Remove duplicates from pair list.
    unique_pairs = {}

    for pair in near_duplicate_pairs:

        key = (
            min(
                pair["index_a"],
                pair["index_b"],
            ),
            max(
                pair["index_a"],
                pair["index_b"],
            ),
        )

        unique_pairs[key] = pair

    near_duplicate_pairs = list(
        unique_pairs.values()
    )

    near_duplicate_pairs.sort(
        key=lambda x: x["similarity"],
        reverse=True,
    )

    print(
        f"Near-duplicate pairs found "
        f"(similarity >= "
        f"{NEAR_DUPLICATE_THRESHOLD}): "
        f"{len(near_duplicate_pairs)}"
    )

    print("\nTop near-duplicate pairs:")

    for pair in near_duplicate_pairs[:20]:

        print(
            f"\nSimilarity: "
            f"{pair['similarity']:.4f}"
        )

        print(
            f"Index A: {pair['index_a']} "
            f"Label: {pair['label_a']}"
        )

        print(
            f"Index B: {pair['index_b']} "
            f"Label: {pair['label_b']}"
        )

        text_a = df.loc[
            pair["index_a"],
            "processed_text",
        ]

        text_b = df.loc[
            pair["index_b"],
            "processed_text",
        ]

        print(
            "A:",
            text_a[:180],
        )

        print(
            "B:",
            text_b[:180],
        )

    # --------------------------------------------------------
    # 13. POTENTIAL CROSS-LABEL NEAR DUPLICATES
    # --------------------------------------------------------

    cross_label_near_duplicates = [
        pair
        for pair in near_duplicate_pairs
        if pair["label_a"]
        != pair["label_b"]
    ]

    print_section(
        "CROSS-LABEL NEAR-DUPLICATE ANALYSIS"
    )

    print(
        "Near-duplicate pairs with different labels:"
    )

    print(
        len(cross_label_near_duplicates)
    )

    if cross_label_near_duplicates:

        print(
            "\nWARNING:"
            " Similar messages appear under"
            " different labels."
        )

        for pair in (
            cross_label_near_duplicates[:20]
        ):

            print(
                f"\nSimilarity: "
                f"{pair['similarity']:.4f}"
            )

            print(
                f"Labels: "
                f"{pair['label_a']} vs "
                f"{pair['label_b']}"
            )

            print(
                "A:",
                df.loc[
                    pair["index_a"],
                    "processed_text",
                ][:200],
            )

            print(
                "B:",
                df.loc[
                    pair["index_b"],
                    "processed_text",
                ][:200],
            )

    # --------------------------------------------------------
    # 14. LEAKAGE ASSESSMENT
    # --------------------------------------------------------

    print_section(
        "LEAKAGE ASSESSMENT"
    )

    warnings = []

    if len(exact_overlap) > 0:

        warnings.append(
            "Exact text overlap exists "
            "between training and testing."
        )

    if len(template_overlap) > 0:

        warnings.append(
            "Normalized templates overlap "
            "between training and testing."
        )

    if len(cross_label_near_duplicates) > 0:

        warnings.append(
            "Near-duplicate messages exist "
            "across different labels."
        )

    if duplicate_rows > 0:

        warnings.append(
            "Dataset contains duplicated rows."
        )

    if warnings:

        print(
            "\nPotential issues detected:"
        )

        for warning in warnings:

            print(
                f"- {warning}"
            )

    else:

        print(
            "\nNo obvious leakage indicators "
            "were detected by these tests."
        )

    # --------------------------------------------------------
    # 15. SAVE REPORT
    # --------------------------------------------------------

    print_section(
        "SAVING ANALYSIS REPORT"
    )

    os.makedirs(
        REPORT_DIR,
        exist_ok=True,
    )

    report = {

        "dataset": {
            "samples":
                int(total_samples),

            "columns":
                df.columns.tolist(),

            "label_distribution": {
                "legitimate":
                    int(
                        label_counts.get(
                            0,
                            0,
                        )
                    ),

                "phishing":
                    int(
                        label_counts.get(
                            1,
                            0,
                        )
                    ),
            },
        },

        "exact_duplicates": {
            "duplicate_rows":
                duplicate_rows,

            "unique_duplicated_texts":
                duplicate_groups,
        },

        "cross_label_duplicates": {
            "count":
                cross_label_count,
        },

        "template_analysis": {

            "repeated_template_count":
                len(repeated_templates),

            "train_template_count":
                len(train_templates),

            "test_template_count":
                len(test_templates),

            "template_overlap_count":
                len(template_overlap),

            "test_template_overlap_percentage":
                template_overlap_percentage,
        },

        "train_test_analysis": {

            "training_samples":
                len(train_set),

            "testing_samples":
                len(test_set),

            "unique_train_texts":
                len(train_texts),

            "unique_test_texts":
                len(test_texts),

            "exact_text_overlap":
                len(exact_overlap),
        },

        "near_duplicate_analysis": {

            "sample_size":
                similarity_sample_size,

            "threshold":
                NEAR_DUPLICATE_THRESHOLD,

            "near_duplicate_pairs":
                len(
                    near_duplicate_pairs
                ),

            "cross_label_near_duplicates":
                len(
                    cross_label_near_duplicates
                ),

            "top_pairs":
                near_duplicate_pairs[:50],
        },

        "length_analysis":
            length_summary,

        "leakage_warnings":
            warnings,
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
        f"\nDetailed report saved to:"
    )

    print(
        REPORT_FILE
    )

    print("\n" + "=" * 70)
    print(
        "DATASET LEAKAGE ANALYSIS COMPLETE"
    )
    print("=" * 70)

    print(
        "\nProduction model remains unchanged."
    )


if __name__ == "__main__":
    main()
