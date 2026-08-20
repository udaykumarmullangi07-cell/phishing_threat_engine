import os

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
)

from sklearn.model_selection import train_test_split


# =========================================================
# CONFIGURATION
# =========================================================

INPUT_FILE = "data/cleaned_text.csv"

REPORT_FILE = (
    "reports/text_character_only_experiment.csv"
)

RANDOM_STATE = 42
TEST_SIZE = 0.20

MAX_CHAR_FEATURES = 30000


# =========================================================
# LOAD DATA
# =========================================================

def load_data():

    df = pd.read_csv(
        INPUT_FILE
    )

    df["processed_text"] = (
        df["processed_text"]
        .fillna("")
    )

    X = df["processed_text"]
    y = df["label"]

    return X, y


# =========================================================
# MAIN EXPERIMENT
# =========================================================

def main():

    print("=" * 70)

    print(
        "TEXT MODEL — CHARACTER-ONLY TF-IDF EXPERIMENT"
    )

    print("=" * 70)

    print(
        "\nIMPORTANT:"
    )

    print(
        "The production model will NOT be modified."
    )

    # =====================================================
    # LOAD DATA
    # =====================================================

    print(
        "\nLoading dataset..."
    )

    X, y = load_data()

    print(
        "Dataset samples:",
        len(X)
    )


    # =====================================================
    # TRAIN / TEST SPLIT
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
    # CHARACTER TF-IDF
    # =====================================================

    print(
        "\nCreating CHARACTER TF-IDF..."
    )

    vectorizer = TfidfVectorizer(

        analyzer="char",

        ngram_range=(3, 5),

        min_df=3,

        max_features=MAX_CHAR_FEATURES,

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
        "Training TF-IDF:",
        X_train_tfidf.shape
    )

    print(
        "Testing TF-IDF: ",
        X_test_tfidf.shape
    )


    # =====================================================
    # LOGISTIC REGRESSION
    # =====================================================

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


    # =====================================================
    # PREDICTIONS
    # =====================================================

    print(
        "\nCalculating predictions..."
    )

    y_pred = model.predict(
        X_test_tfidf
    )

    y_probability = (
        model
        .predict_proba(
            X_test_tfidf
        )[:, 1]
    )


    # =====================================================
    # METRICS
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

    roc_auc = roc_auc_score(
        y_test,
        y_probability,
    )


    # =====================================================
    # CONFUSION MATRIX
    # =====================================================

    tn, fp, fn, tp = (
        confusion_matrix(
            y_test,
            y_pred,
            labels=[0, 1],
        ).ravel()
    )


    # =====================================================
    # RESULTS
    # =====================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "CHARACTER-ONLY TF-IDF RESULTS"
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
        f"ROC-AUC  : {roc_auc:.4f}"
    )

    print(
        f"False Positives: {fp}"
    )

    print(
        f"False Negatives: {fn}"
    )


    # =====================================================
    # CURRENT WORD MODEL BASELINE
    # =====================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "CURRENT WORD TF-IDF BASELINE"
    )

    print(
        "=" * 70
    )

    print(
        "Model: Logistic Regression"
    )

    print(
        "Features: Word TF-IDF"
    )

    print(
        "Word features: 50,000"
    )

    print(
        "C: 30"
    )

    print(
        "Accuracy : 0.9911"
    )

    print(
        "Precision: 0.9911"
    )

    print(
        "Recall   : 0.9917"
    )

    print(
        "F1       : 0.9914"
    )

    print(
        "ROC-AUC  : 0.9994"
    )

    print(
        "FP       : 76"
    )

    print(
        "FN       : 71"
    )


    # =====================================================
    # WORD + CHARACTER BASELINE
    # =====================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "WORD + CHARACTER EXPERIMENT BASELINE"
    )

    print(
        "=" * 70
    )

    print(
        "Word features      : 50,000"
    )

    print(
        "Character features : 30,000"
    )

    print(
        "Total features     : 80,000"
    )

    print(
        "Accuracy           : 0.9915"
    )

    print(
        "Precision          : 0.9913"
    )

    print(
        "Recall             : 0.9923"
    )

    print(
        "F1                 : 0.9918"
    )

    print(
        "ROC-AUC            : 0.9995"
    )

    print(
        "FP                 : 75"
    )

    print(
        "FN                 : 66"
    )


    # =====================================================
    # EXPERIMENTAL MODEL SUMMARY
    # =====================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "CHARACTER-ONLY EXPERIMENT"
    )

    print(
        "=" * 70
    )

    print(
        "Representation: Character TF-IDF"
    )

    print(
        f"Character features: "
        f"{X_train_tfidf.shape[1]}"
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
        f"ROC-AUC  : {roc_auc:.4f}"
    )

    print(
        f"FP       : {fp}"
    )

    print(
        f"FN       : {fn}"
    )


    # =====================================================
    # SAVE RESULTS
    # =====================================================

    os.makedirs(
        "reports",
        exist_ok=True,
    )


    results = pd.DataFrame(
        [
            {
                "model":
                    "character_only_tfidf",

                "character_features":
                    X_train_tfidf.shape[1],

                "accuracy":
                    accuracy,

                "precision":
                    precision,

                "recall":
                    recall,

                "f1":
                    f1,

                "roc_auc":
                    roc_auc,

                "false_positives":
                    int(fp),

                "false_negatives":
                    int(fn),

                "true_negatives":
                    int(tn),

                "true_positives":
                    int(tp),
            }
        ]
    )


    results.to_csv(
        REPORT_FILE,
        index=False,
    )


    print(
        "\nExperiment report saved to:"
    )

    print(
        REPORT_FILE
    )


    print(
        "\nProduction model remains unchanged."
    )


    print(
        "\n" + "=" * 70
    )

    print(
        "CHARACTER-ONLY EXPERIMENT COMPLETE"
    )

    print(
        "=" * 70
    )


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":

    main()
