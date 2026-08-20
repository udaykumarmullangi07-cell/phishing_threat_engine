import os
import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)


# =========================================================
# FILE PATHS
# =========================================================

INPUT_FILE = "data/cleaned_text.csv"

MODEL_DIR = "models"

TUNED_MODEL_FILE = os.path.join(
    MODEL_DIR,
    "text_logistic_regression_tuned.joblib",
)

VECTORIZER_FILE = os.path.join(
    MODEL_DIR,
    "text_tfidf_vectorizer_tuned.joblib",
)

RESULT_FILE = os.path.join(
    MODEL_DIR,
    "text_tuning_results.csv",
)


# =========================================================
# CONFIGURATION
# =========================================================

TEST_SIZE = 0.20
RANDOM_STATE = 42


# =========================================================
# MAIN
# =========================================================

def main():

    print("=" * 70)
    print("TEXT MODEL — LOGISTIC REGRESSION HYPERPARAMETER TUNING")
    print("=" * 70)


    # =====================================================
    # 1. LOAD DATASET
    # =====================================================

    print("\nLoading dataset...")

    df = pd.read_csv(
        INPUT_FILE
    )

    print(
        f"Dataset shape: {df.shape}"
    )


    # =====================================================
    # 2. HANDLE MISSING TEXT
    # =====================================================

    df["processed_text"] = (
        df["processed_text"]
        .fillna("")
    )


    X = df["processed_text"]
    y = df["label"]


    # =====================================================
    # 3. TRAIN / TEST SPLIT
    #
    # MUST MATCH BASELINE TRAINING
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
        f"\nTraining samples: {len(X_train)}"
    )

    print(
        f"Testing samples:  {len(X_test)}"
    )


    # =====================================================
    # 4. TF-IDF
    #
    # Keep this identical to the baseline model.
    # =====================================================

    print(
        "\nCreating TF-IDF matrices..."
    )

    vectorizer = TfidfVectorizer(
        max_features=50000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
    )


    X_train_tfidf = vectorizer.fit_transform(
        X_train
    )

    X_test_tfidf = vectorizer.transform(
        X_test
    )


    print(
        "X_train TF-IDF:",
        X_train_tfidf.shape,
    )

    print(
        "X_test TF-IDF:",
        X_test_tfidf.shape,
    )


    # =====================================================
    # 5. BASELINE MODEL
    # =====================================================

    print(
        "\nTraining baseline Logistic Regression..."
    )

    baseline_model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        solver="liblinear",
        random_state=42,
    )


    baseline_model.fit(
        X_train_tfidf,
        y_train,
    )


    # =====================================================
    # 6. BASELINE EVALUATION
    # =====================================================

    baseline_pred = baseline_model.predict(
        X_test_tfidf
    )

    baseline_probability = (
        baseline_model
        .predict_proba(X_test_tfidf)[:, 1]
    )


    baseline_accuracy = accuracy_score(
        y_test,
        baseline_pred,
    )

    baseline_precision = precision_score(
        y_test,
        baseline_pred,
        pos_label=1,
    )

    baseline_recall = recall_score(
        y_test,
        baseline_pred,
        pos_label=1,
    )

    baseline_f1 = f1_score(
        y_test,
        baseline_pred,
        pos_label=1,
    )

    baseline_roc_auc = roc_auc_score(
        y_test,
        baseline_probability,
    )


    print("\n" + "=" * 70)
    print("BASELINE RESULTS")
    print("=" * 70)

    print(
        f"Accuracy : {baseline_accuracy:.4f}"
    )

    print(
        f"Precision: {baseline_precision:.4f}"
    )

    print(
        f"Recall   : {baseline_recall:.4f}"
    )

    print(
        f"F1 Score : {baseline_f1:.4f}"
    )

    print(
        f"ROC-AUC  : {baseline_roc_auc:.4f}"
    )


    # =====================================================
    # 7. GRID SEARCH
    # =====================================================

    print("\n" + "=" * 70)
    print("STARTING GRID SEARCH")
    print("=" * 70)


    tuning_model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        solver="liblinear",
        random_state=42,
    )


    parameter_grid = {
        "C": [
            0.1,
            1,
            3,
            10,
            30,
        ],
    }


    grid_search = GridSearchCV(
        estimator=tuning_model,
        param_grid=parameter_grid,
        scoring="f1",
        cv=3,
        n_jobs=-1,
        verbose=1,
        return_train_score=True,
    )


    print(
        "\nParameter grid:"
    )

    print(
        parameter_grid
    )


    print(
        "\nScoring metric: F1"
    )

    print(
        "Cross-validation folds: 3"
    )


    grid_search.fit(
        X_train_tfidf,
        y_train,
    )


    # =====================================================
    # 8. BEST PARAMETERS
    # =====================================================

    print("\n" + "=" * 70)
    print("GRID SEARCH COMPLETE")
    print("=" * 70)


    print(
        "\nBest parameters:"
    )

    print(
        grid_search.best_params_
    )


    print(
        "\nBest cross-validation F1:"
    )

    print(
        f"{grid_search.best_score_:.4f}"
    )


    # =====================================================
    # 9. BEST MODEL
    # =====================================================

    tuned_model = (
        grid_search.best_estimator_
    )


    # =====================================================
    # 10. TEST SET EVALUATION
    # =====================================================

    tuned_pred = tuned_model.predict(
        X_test_tfidf
    )

    tuned_probability = (
        tuned_model
        .predict_proba(X_test_tfidf)[:, 1]
    )


    tuned_accuracy = accuracy_score(
        y_test,
        tuned_pred,
    )

    tuned_precision = precision_score(
        y_test,
        tuned_pred,
        pos_label=1,
    )

    tuned_recall = recall_score(
        y_test,
        tuned_pred,
        pos_label=1,
    )

    tuned_f1 = f1_score(
        y_test,
        tuned_pred,
        pos_label=1,
    )

    tuned_roc_auc = roc_auc_score(
        y_test,
        tuned_probability,
    )


    # =====================================================
    # 11. TUNED RESULTS
    # =====================================================

    print("\n" + "=" * 70)
    print("TUNED MODEL RESULTS")
    print("=" * 70)

    print(
        f"Accuracy : {tuned_accuracy:.4f}"
    )

    print(
        f"Precision: {tuned_precision:.4f}"
    )

    print(
        f"Recall   : {tuned_recall:.4f}"
    )

    print(
        f"F1 Score : {tuned_f1:.4f}"
    )

    print(
        f"ROC-AUC  : {tuned_roc_auc:.4f}"
    )


    # =====================================================
    # 12. COMPARISON
    # =====================================================

    print("\n" + "=" * 70)
    print("BASELINE vs TUNED")
    print("=" * 70)


    comparison = pd.DataFrame({

        "Metric": [
            "Accuracy",
            "Precision",
            "Recall",
            "F1",
            "ROC-AUC",
        ],

        "Baseline": [
            baseline_accuracy,
            baseline_precision,
            baseline_recall,
            baseline_f1,
            baseline_roc_auc,
        ],

        "Tuned": [
            tuned_accuracy,
            tuned_precision,
            tuned_recall,
            tuned_f1,
            tuned_roc_auc,
        ],

    })


    comparison["Improvement"] = (
        comparison["Tuned"]
        - comparison["Baseline"]
    )


    print(
        comparison.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )


    # =====================================================
    # 13. CLASSIFICATION REPORT
    # =====================================================

    print("\n" + "=" * 70)
    print("TUNED MODEL — CLASSIFICATION REPORT")
    print("=" * 70)

    print(
        classification_report(
            y_test,
            tuned_pred,
            target_names=[
                "Legitimate",
                "Phishing",
            ],
        )
    )


    # =====================================================
    # 14. CONFUSION MATRIX
    # =====================================================

    print("Confusion Matrix:")

    print(
        confusion_matrix(
            y_test,
            tuned_pred,
        )
    )


    # =====================================================
    # 15. SAVE TUNED MODEL
    # =====================================================

    os.makedirs(
        MODEL_DIR,
        exist_ok=True,
    )


    joblib.dump(
        tuned_model,
        TUNED_MODEL_FILE,
    )


    joblib.dump(
        vectorizer,
        VECTORIZER_FILE,
    )


    # =====================================================
    # 16. SAVE TUNING RESULTS
    # =====================================================

    comparison.to_csv(
        RESULT_FILE,
        index=False,
    )


    print("\n" + "=" * 70)
    print("FILES SAVED")
    print("=" * 70)


    print(
        f"Tuned model:"
    )

    print(
        TUNED_MODEL_FILE
    )


    print(
        "\nTuned vectorizer:"
    )

    print(
        VECTORIZER_FILE
    )


    print(
        "\nComparison:"
    )

    print(
        RESULT_FILE
    )


    # =====================================================
    # 17. FINAL DECISION
    # =====================================================

    print("\n" + "=" * 70)
    print("MODEL SELECTION DECISION")
    print("=" * 70)


    if tuned_f1 > baseline_f1:

        print(
            "\nTuned model improved F1."
        )

        print(
            "Tuned model is the candidate for the final model."
        )

    elif tuned_f1 == baseline_f1:

        print(
            "\nTuned and baseline models have equal F1."
        )

        print(
            "Further comparison of other metrics is required."
        )

    else:

        print(
            "\nTuned model did not improve F1."
        )

        print(
            "Baseline model remains the preferred model."
        )


    print("\n" + "=" * 70)


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":

    main()
