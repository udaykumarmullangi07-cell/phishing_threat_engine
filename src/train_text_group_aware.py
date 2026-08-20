import os
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

from sklearn.model_selection import GroupShuffleSplit


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = "data/cleaned_text.csv"

REPORT_DIR = "reports"

REPORT_FILE = os.path.join(
    REPORT_DIR,
    "text_group_aware_evaluation.json",
)

RANDOM_STATE = 42

TEST_SIZE = 0.20


# ============================================================
# GROUP CREATION
# ============================================================

def create_message_group(text):

    """
    Create a normalized group identifier.

    Messages that are identical after normalization
    are placed into the same group.

    This prevents identical messages from appearing
    in both training and testing data.
    """

    text = str(text).lower().strip()

    return text


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("TEXT MODEL — GROUP-AWARE EVALUATION")
    print("=" * 70)

    print(
        """
Purpose:
Create a leakage-resistant train/test split where
identical messages cannot appear in both sets.

The existing production model will NOT be modified.
"""
    )

    # --------------------------------------------------------
    # 1. LOAD DATASET
    # --------------------------------------------------------

    print("\nLoading dataset...")

    df = pd.read_csv(INPUT_FILE)

    df["processed_text"] = (
        df["processed_text"]
        .fillna("")
        .astype(str)
    )

    df["label"] = df["label"].astype(int)

    print(
        f"Dataset samples: {len(df)}"
    )

    # --------------------------------------------------------
    # 2. CREATE GROUPS
    # --------------------------------------------------------

    print(
        "\nCreating message groups..."
    )

    df["group"] = (
        df["processed_text"]
        .apply(create_message_group)
    )

    unique_groups = (
        df["group"].nunique()
    )

    print(
        f"Unique message groups: "
        f"{unique_groups}"
    )

    print(
        f"Duplicate groups: "
        f"{len(df) - unique_groups}"
    )

    # --------------------------------------------------------
    # 3. GROUP DISTRIBUTION
    # --------------------------------------------------------

    print(
        "\nChecking group label consistency..."
    )

    group_label_counts = (
        df.groupby("group")["label"]
        .nunique()
    )

    mixed_groups = (
        group_label_counts[
            group_label_counts > 1
        ]
    )

    print(
        f"Groups containing multiple labels: "
        f"{len(mixed_groups)}"
    )

    if len(mixed_groups) > 0:

        print(
            "\nWARNING:"
            " Some identical messages have"
            " different labels."
        )

    else:

        print(
            "All identical-message groups"
            " have consistent labels."
        )

    # --------------------------------------------------------
    # 4. GROUP-AWARE SPLIT
    # --------------------------------------------------------

    print(
        "\nCreating group-aware train/test split..."
    )

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    train_idx, test_idx = next(
        splitter.split(
            df,
            df["label"],
            groups=df["group"],
        )
    )

    train_df = df.iloc[
        train_idx
    ].copy()

    test_df = df.iloc[
        test_idx
    ].copy()

    print(
        f"Training samples: "
        f"{len(train_df)}"
    )

    print(
        f"Testing samples:  "
        f"{len(test_df)}"
    )

    # --------------------------------------------------------
    # 5. VERIFY NO EXACT OVERLAP
    # --------------------------------------------------------

    train_groups = set(
        train_df["group"]
    )

    test_groups = set(
        test_df["group"]
    )

    overlap = (
        train_groups
        .intersection(test_groups)
    )

    print(
        "\nGroup overlap:"
    )

    print(
        len(overlap)
    )

    if overlap:

        raise RuntimeError(
            "Group leakage detected!"
        )

    print(
        "PASS — no message groups overlap."
    )

    # --------------------------------------------------------
    # 6. LABEL DISTRIBUTION
    # --------------------------------------------------------

    print(
        "\nTraining label distribution:"
    )

    print(
        train_df["label"]
        .value_counts()
        .sort_index()
    )

    print(
        "\nTesting label distribution:"
    )

    print(
        test_df["label"]
        .value_counts()
        .sort_index()
    )

    # --------------------------------------------------------
    # 7. PREPARE DATA
    # --------------------------------------------------------

    X_train = train_df[
        "processed_text"
    ]

    y_train = train_df[
        "label"
    ]

    X_test = test_df[
        "processed_text"
    ]

    y_test = test_df[
        "label"
    ]

    # --------------------------------------------------------
    # 8. TF-IDF
    # --------------------------------------------------------

    print(
        "\nCreating TF-IDF..."
    )

    vectorizer = TfidfVectorizer(
        max_features=50000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
    )

    X_train_tfidf = (
        vectorizer.fit_transform(
            X_train
        )
    )

    X_test_tfidf = (
        vectorizer.transform(
            X_test
        )
    )

    print(
        f"Training TF-IDF: "
        f"{X_train_tfidf.shape}"
    )

    print(
        f"Testing TF-IDF:  "
        f"{X_test_tfidf.shape}"
    )

    # --------------------------------------------------------
    # 9. TRAIN MODEL
    # --------------------------------------------------------

    print(
        "\nTraining Logistic Regression..."
    )

    model = LogisticRegression(
        C=30,
        max_iter=1000,
        class_weight="balanced",
        solver="liblinear",
        random_state=RANDOM_STATE,
    )

    model.fit(
        X_train_tfidf,
        y_train,
    )

    print(
        "Training completed."
    )

    # --------------------------------------------------------
    # 10. PREDICTIONS
    # --------------------------------------------------------

    print(
        "\nGenerating predictions..."
    )

    y_pred = model.predict(
        X_test_tfidf
    )

    y_probability = (
        model.predict_proba(
            X_test_tfidf
        )[:, 1]
    )

    # --------------------------------------------------------
    # 11. METRICS
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        y_pred,
    )

    precision = precision_score(
        y_test,
        y_pred,
        pos_label=1,
    )

    recall = recall_score(
        y_test,
        y_pred,
        pos_label=1,
    )

    f1 = f1_score(
        y_test,
        y_pred,
        pos_label=1,
    )

    roc_auc = roc_auc_score(
        y_test,
        y_probability,
    )

    cm = confusion_matrix(
        y_test,
        y_pred,
    )

    tn, fp, fn, tp = cm.ravel()

    # --------------------------------------------------------
    # 12. RESULTS
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print(
        "GROUP-AWARE MODEL RESULTS"
    )
    print("=" * 70)

    print(
        f"Accuracy : {accuracy:.4f}"
    )

    print(
        f"Precision: {precision:.4f}"
    )

    print(
        f"Recall   : {recall:.4f}"
    )

    print(
        f"F1 Score : {f1:.4f}"
    )

    print(
        f"ROC-AUC  : {roc_auc:.4f}"
    )

    print(
        f"\nTrue Negatives : {tn}"
    )

    print(
        f"False Positives: {fp}"
    )

    print(
        f"False Negatives: {fn}"
    )

    print(
        f"True Positives : {tp}"
    )

    print(
        "\nConfusion Matrix:"
    )

    print(cm)

    print(
        "\nClassification Report:"
    )

    print(
        classification_report(
            y_test,
            y_pred,
            target_names=[
                "Legitimate",
                "Phishing",
            ],
        )
    )

    # --------------------------------------------------------
    # 13. SAVE EXPERIMENT MODEL
    # --------------------------------------------------------

    os.makedirs(
        "models",
        exist_ok=True,
    )

    experimental_model_file = (
        "models/"
        "experimental_group_aware_text_model.joblib"
    )

    experimental_vectorizer_file = (
        "models/"
        "experimental_group_aware_text_vectorizer.joblib"
    )

    print(
        "\nSaving experimental model..."
    )

    joblib.dump(
        model,
        experimental_model_file,
    )

    joblib.dump(
        vectorizer,
        experimental_vectorizer_file,
    )

    print(
        experimental_model_file
    )

    print(
        experimental_vectorizer_file
    )

    # --------------------------------------------------------
    # 14. SAVE REPORT
    # --------------------------------------------------------

    report = {

        "experiment": {
            "name":
                "Group-aware text evaluation",

            "production_model_modified":
                False,

            "random_state":
                RANDOM_STATE,

            "test_size":
                TEST_SIZE,
        },

        "dataset": {

            "total_samples":
                len(df),

            "unique_groups":
                unique_groups,

            "mixed_label_groups":
                len(mixed_groups),
        },

        "split": {

            "training_samples":
                len(train_df),

            "testing_samples":
                len(test_df),

            "group_overlap":
                len(overlap),
        },

        "model": {

            "algorithm":
                "Logistic Regression",

            "C":
                30,

            "solver":
                "liblinear",

            "class_weight":
                "balanced",

            "tfidf_features":
                50000,

            "ngram_range":
                [1, 2],
        },

        "metrics": {

            "accuracy":
                float(accuracy),

            "precision":
                float(precision),

            "recall":
                float(recall),

            "f1":
                float(f1),

            "roc_auc":
                float(roc_auc),

            "true_negatives":
                int(tn),

            "false_positives":
                int(fp),

            "false_negatives":
                int(fn),

            "true_positives":
                int(tp),
        },
    }

    os.makedirs(
        REPORT_DIR,
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
        "\nReport saved to:"
    )

    print(
        REPORT_FILE
    )

    print("\n" + "=" * 70)
    print(
        "GROUP-AWARE EVALUATION COMPLETE"
    )
    print("=" * 70)

    print(
        "\nIMPORTANT:"
    )

    print(
        "This is an experimental evaluation."
    )

    print(
        "The existing production model remains unchanged."
    )


if __name__ == "__main__":
    main()
