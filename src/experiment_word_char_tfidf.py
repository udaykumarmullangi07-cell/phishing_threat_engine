import os

import joblib
import pandas as pd

from scipy.sparse import hstack

from sklearn.feature_extraction.text import (
    TfidfVectorizer,
)

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
    "reports/text_word_char_experiment.csv"
)

RANDOM_STATE = 42
TEST_SIZE = 0.20

MAX_WORD_FEATURES = 50000
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
# MAIN
# =========================================================

def main():

    print("=" * 70)

    print(
        "TEXT MODEL — WORD + CHARACTER TF-IDF EXPERIMENT"
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
        len(X),
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
        len(X_train),
    )

    print(
        "Testing samples: ",
        len(X_test),
    )


    # =====================================================
    # WORD TF-IDF
    # =====================================================

    print(
        "\nCreating WORD TF-IDF..."
    )


    word_vectorizer = TfidfVectorizer(

        max_features=MAX_WORD_FEATURES,

        ngram_range=(1, 2),

        min_df=2,

        max_df=0.95,

        sublinear_tf=True,
    )


    X_train_word = (
        word_vectorizer.fit_transform(
            X_train
        )
    )

    X_test_word = (
        word_vectorizer.transform(
            X_test
        )
    )


    print(
        "Word training shape:",
        X_train_word.shape,
    )

    print(
        "Word testing shape: ",
        X_test_word.shape,
    )


    # =====================================================
    # CHARACTER TF-IDF
    # =====================================================

    print(
        "\nCreating CHARACTER TF-IDF..."
    )


    char_vectorizer = TfidfVectorizer(

        analyzer="char",

        ngram_range=(3, 5),

        min_df=3,

        max_features=MAX_CHAR_FEATURES,

        sublinear_tf=True,
    )


    X_train_char = (
        char_vectorizer.fit_transform(
            X_train
        )
    )

    X_test_char = (
        char_vectorizer.transform(
            X_test
        )
    )


    print(
        "Character training shape:",
        X_train_char.shape,
    )

    print(
        "Character testing shape: ",
        X_test_char.shape,
    )


    # =====================================================
    # COMBINE FEATURES
    # =====================================================

    print(
        "\nCombining word + character features..."
    )


    X_train_combined = hstack(
        [
            X_train_word,
            X_train_char,
        ]
    ).tocsr()


    X_test_combined = hstack(
        [
            X_test_word,
            X_test_char,
        ]
    ).tocsr()


    print(
        "Combined training shape:",
        X_train_combined.shape,
    )

    print(
        "Combined testing shape: ",
        X_test_combined.shape,
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
        X_train_combined,
        y_train,
    )


    print(
        "Training completed."
    )


    # =====================================================
    # PREDICTIONS
    # =====================================================

    y_pred = model.predict(
        X_test_combined
    )


    y_probability = (
        model
        .predict_proba(
            X_test_combined
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
        "WORD + CHARACTER TF-IDF RESULTS"
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
    # BASELINE COMPARISON
    # =====================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "CURRENT BASELINE"
    )

    print(
        "=" * 70
    )

    print(
        "Model: Logistic Regression"
    )

    print(
        "TF-IDF: Word n-grams"
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
    # EXPERIMENT
    # =====================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "EXPERIMENTAL MODEL"
    )

    print(
        "=" * 70
    )

    print(
        "Model: Logistic Regression"
    )

    print(
        "Features: Word + Character TF-IDF"
    )

    print(
        f"Total features: "
        f"{X_train_combined.shape[1]}"
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
    # SAVE EXPERIMENT RESULTS
    # =====================================================

    os.makedirs(
        "reports",
        exist_ok=True,
    )


    results = pd.DataFrame(
        [
            {
                "model":
                    "word_char_tfidf",

                "word_features":
                    X_train_word.shape[1],

                "char_features":
                    X_train_char.shape[1],

                "total_features":
                    X_train_combined.shape[1],

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
        "\n" + "=" * 70
    )

    print(
        "EXPERIMENT COMPLETE"
    )

    print(
        "=" * 70
    )


if __name__ == "__main__":

    main()
