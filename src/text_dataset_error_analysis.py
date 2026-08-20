import os
import re
import json
import joblib
import pandas as pd

from collections import Counter

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


# =========================================================
# CONFIGURATION
# =========================================================

INPUT_FILE = "data/cleaned_text.csv"

MODEL_FILE = (
    "models/final_text_model.joblib"
)

VECTORIZER_FILE = (
    "models/final_text_vectorizer.joblib"
)

REPORT_FILE = (
    "reports/text_dataset_error_analysis.json"
)

RANDOM_STATE = 42
TEST_SIZE = 0.20


# =========================================================
# TOKEN EXTRACTION
# =========================================================

def extract_words(text):

    if not isinstance(text, str):
        return []

    return re.findall(
        r"\b[a-zA-Z]{3,}\b",
        text.lower(),
    )


# =========================================================
# WORD STATISTICS
# =========================================================

def word_statistics(texts):

    counter = Counter()

    for text in texts:
        counter.update(
            extract_words(text)
        )

    return counter


# =========================================================
# MAIN
# =========================================================

def main():

    print("=" * 70)
    print(
        "TEXT MODEL — DATASET-LEVEL ERROR ANALYSIS"
    )
    print("=" * 70)

    print(
        "\nPurpose:"
    )

    print(
        "Analyze real validation-set false positives "
        "and false negatives."
    )

    print(
        "\nProduction model will NOT be modified."
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

    df["processed_text"] = (
        df["processed_text"]
        .fillna("")
    )

    X = df["processed_text"]
    y = df["label"]

    print(
        "Dataset samples:",
        len(df)
    )

    # =====================================================
    # SAME TRAIN / TEST SPLIT
    # =====================================================

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
        "\nTraining samples:",
        len(X_train)
    )

    print(
        "Testing samples: ",
        len(X_test)
    )

    # =====================================================
    # LOAD MODEL
    # =====================================================

    print(
        "\nLoading final production model..."
    )

    model = joblib.load(
        MODEL_FILE
    )

    vectorizer = joblib.load(
        VECTORIZER_FILE
    )

    print(
        "Model:",
        type(model).__name__
    )

    print(
        "C:",
        model.C
    )

    print(
        "TF-IDF features:",
        len(
            vectorizer.get_feature_names_out()
        )
    )

    # =====================================================
    # TRANSFORM TEST DATA
    # =====================================================

    print(
        "\nCreating TF-IDF test representation..."
    )

    X_test_tfidf = (
        vectorizer.transform(
            X_test
        )
    )

    print(
        "Test TF-IDF shape:",
        X_test_tfidf.shape
    )

    # =====================================================
    # PREDICTIONS
    # =====================================================

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

    # =====================================================
    # OVERALL METRICS
    # =====================================================

    accuracy = accuracy_score(
        y_test,
        y_pred,
    )

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0,
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0,
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0,
    )

    tn, fp, fn, tp = (
        confusion_matrix(
            y_test,
            y_pred,
            labels=[0, 1],
        ).ravel()
    )

    # =====================================================
    # BUILD ERROR DATAFRAME
    # =====================================================

    test_results = pd.DataFrame(
        {
            "text": X_test.values,
            "actual_label": y_test.values,
            "predicted_label": y_pred,
            "phishing_probability":
                y_probability,
        }
    )

    test_results[
        "prediction_correct"
    ] = (
        test_results["actual_label"]
        ==
        test_results["predicted_label"]
    )

    # =====================================================
    # FALSE POSITIVES
    # =====================================================

    false_positives = (
        test_results[
            (
                test_results["actual_label"] == 0
            )
            &
            (
                test_results["predicted_label"] == 1
            )
        ]
        .sort_values(
            "phishing_probability",
            ascending=False,
        )
    )

    # =====================================================
    # FALSE NEGATIVES
    # =====================================================

    false_negatives = (
        test_results[
            (
                test_results["actual_label"] == 1
            )
            &
            (
                test_results["predicted_label"] == 0
            )
        ]
        .sort_values(
            "phishing_probability",
            ascending=True,
        )
    )

    # =====================================================
    # CORRECT LEGITIMATE
    # =====================================================

    correct_legitimate = (
        test_results[
            (
                test_results["actual_label"] == 0
            )
            &
            (
                test_results["predicted_label"] == 0
            )
        ]
    )

    # =====================================================
    # CORRECT PHISHING
    # =====================================================

    correct_phishing = (
        test_results[
            (
                test_results["actual_label"] == 1
            )
            &
            (
                test_results["predicted_label"] == 1
            )
        ]
    )

    # =====================================================
    # CLASS-SPECIFIC ERROR RATES
    # =====================================================

    legitimate_total = (
        (y_test == 0).sum()
    )

    phishing_total = (
        (y_test == 1).sum()
    )

    false_positive_rate = (
        fp / legitimate_total
    )

    false_negative_rate = (
        fn / phishing_total
    )

    # =====================================================
    # WORD ANALYSIS
    # =====================================================

    fp_word_counts = (
        word_statistics(
            false_positives["text"]
        )
    )

    legitimate_word_counts = (
        word_statistics(
            correct_legitimate["text"]
        )
    )

    fn_word_counts = (
        word_statistics(
            false_negatives["text"]
        )
    )

    phishing_word_counts = (
        word_statistics(
            correct_phishing["text"]
        )
    )

    # =====================================================
    # PRINT OVERALL RESULTS
    # =====================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "OVERALL VALIDATION RESULTS"
    )

    print(
        "=" * 70
    )

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
        f"F1       : {f1:.4f}"
    )

    print(
        f"True Negatives : {tn}"
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

    # =====================================================
    # ERROR RATES
    # =====================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "CLASS-SPECIFIC ERROR ANALYSIS"
    )

    print(
        "=" * 70
    )

    print(
        f"Legitimate samples: {legitimate_total}"
    )

    print(
        f"Phishing samples  : {phishing_total}"
    )

    print(
        f"False Positive Rate: "
        f"{false_positive_rate * 100:.4f}%"
    )

    print(
        f"False Negative Rate: "
        f"{false_negative_rate * 100:.4f}%"
    )

    # =====================================================
    # FALSE POSITIVE SUMMARY
    # =====================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "FALSE POSITIVE ANALYSIS"
    )

    print(
        "=" * 70
    )

    print(
        f"False positives found: "
        f"{len(false_positives)}"
    )

    if len(false_positives) > 0:

        print(
            "\nHighest-risk legitimate messages:"
        )

        for index, row in (
            false_positives
            .head(10)
            .iterrows()
        ):

            text_preview = (
                str(row["text"])
                .replace("\n", " ")
                .strip()
            )

            if len(text_preview) > 250:

                text_preview = (
                    text_preview[:250]
                    + "..."
                )

            print(
                "\n--------------------------------------------------"
            )

            print(
                "Phishing probability:",
                f"{row['phishing_probability'] * 100:.2f}%"
            )

            print(
                "Message:"
            )

            print(
                text_preview
            )

    # =====================================================
    # TOP WORDS IN FALSE POSITIVES
    # =====================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "TOP WORDS IN FALSE POSITIVES"
    )

    print(
        "=" * 70
    )

    for word, count in (
        fp_word_counts.most_common(30)
    ):

        print(
            f"{word:25} {count}"
        )

    # =====================================================
    # TOP WORDS IN CORRECT LEGITIMATE
    # =====================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "TOP WORDS IN CORRECTLY CLASSIFIED LEGITIMATE EMAILS"
    )

    print(
        "=" * 70
    )

    for word, count in (
        legitimate_word_counts
        .most_common(30)
    ):

        print(
            f"{word:25} {count}"
        )

    # =====================================================
    # FALSE NEGATIVE ANALYSIS
    # =====================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "FALSE NEGATIVE ANALYSIS"
    )

    print(
        "=" * 70
    )

    print(
        f"False negatives found: "
        f"{len(false_negatives)}"
    )

    if len(false_negatives) > 0:

        print(
            "\nLowest-risk phishing messages:"
        )

        for index, row in (
            false_negatives
            .head(10)
            .iterrows()
        ):

            text_preview = (
                str(row["text"])
                .replace("\n", " ")
                .strip()
            )

            if len(text_preview) > 250:

                text_preview = (
                    text_preview[:250]
                    + "..."
                )

            print(
                "\n--------------------------------------------------"
            )

            print(
                "Phishing probability:",
                f"{row['phishing_probability'] * 100:.2f}%"
            )

            print(
                "Message:"
            )

            print(
                text_preview
            )

    # =====================================================
    # TOP WORDS IN FALSE NEGATIVES
    # =====================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "TOP WORDS IN FALSE NEGATIVES"
    )

    print(
        "=" * 70
    )

    for word, count in (
        fn_word_counts.most_common(30)
    ):

        print(
            f"{word:25} {count}"
        )

    # =====================================================
    # SAVE JSON REPORT
    # =====================================================

    os.makedirs(
        "reports",
        exist_ok=True,
    )

    report = {

        "model": {
            "type":
                type(model).__name__,

            "C":
                model.C,

            "tfidf_features":
                len(
                    vectorizer
                    .get_feature_names_out()
                ),
        },

        "dataset": {

            "total_samples":
                len(df),

            "training_samples":
                len(X_train),

            "testing_samples":
                len(X_test),

            "legitimate_test_samples":
                int(legitimate_total),

            "phishing_test_samples":
                int(phishing_total),
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

            "true_negatives":
                int(tn),

            "false_positives":
                int(fp),

            "false_negatives":
                int(fn),

            "true_positives":
                int(tp),

            "false_positive_rate":
                float(false_positive_rate),

            "false_negative_rate":
                float(false_negative_rate),
        },

        "false_positive_analysis": {

            "count":
                int(len(false_positives)),

            "top_words": [
                {
                    "word": word,
                    "count": int(count),
                }

                for word, count
                in fp_word_counts
                    .most_common(30)
            ],

            "highest_risk_examples": [

                {
                    "phishing_probability":
                        float(
                            row[
                                "phishing_probability"
                            ]
                        ),

                    "text":
                        str(
                            row["text"]
                        ),
                }

                for _, row
                in false_positives
                    .head(10)
                    .iterrows()
            ],
        },

        "false_negative_analysis": {

            "count":
                int(len(false_negatives)),

            "top_words": [
                {
                    "word": word,
                    "count": int(count),
                }

                for word, count
                in fn_word_counts
                    .most_common(30)
            ],

            "lowest_risk_examples": [

                {
                    "phishing_probability":
                        float(
                            row[
                                "phishing_probability"
                            ]
                        ),

                    "text":
                        str(
                            row["text"]
                        ),
                }

                for _, row
                in false_negatives
                    .head(10)
                    .iterrows()
            ],
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
        "\n" + "=" * 70
    )

    print(
        "ERROR ANALYSIS COMPLETE"
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


if __name__ == "__main__":

    main()
