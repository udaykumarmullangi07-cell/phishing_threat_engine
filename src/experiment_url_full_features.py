import os
import json

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)


INPUT_FILE = "data/url_dataset.csv"
REPORT_FILE = "reports/url_full_feature_experiment.json"


def evaluate_model(name, model, X_train, X_test, y_train, y_test):

    print("\n" + "-" * 70)
    print(f"TRAINING {name}")
    print("-" * 70)

    model.fit(
        X_train,
        y_train
    )

    predictions = model.predict(
        X_test
    )

    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions
    )

    recall = recall_score(
        y_test,
        predictions
    )

    f1 = f1_score(
        y_test,
        predictions
    )

    roc_auc = roc_auc_score(
        y_test,
        probabilities
    )

    tn, fp, fn, tp = confusion_matrix(
        y_test,
        predictions
    ).ravel()

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

    return {
        "model": name,
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": float(roc_auc),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_negatives": int(tn),
        "true_positives": int(tp),
    }


def main():

    print("=" * 70)
    print("URL MODEL — FULL FEATURE EXPERIMENT")
    print("=" * 70)

    print(
        """
IMPORTANT:
The production URL model will NOT be modified.

This experiment compares the original dataset
features against the current 10-feature representation.
"""
    )

    # ---------------------------------------------------------
    # LOAD DATASET
    # ---------------------------------------------------------

    print("\nLoading dataset...")

    df = pd.read_csv(
        INPUT_FILE
    )

    print(
        "Dataset samples:",
        len(df)
    )

    # ---------------------------------------------------------
    # LABEL
    # ---------------------------------------------------------

    y = (
        df["status"]
        .map(
            {
                "legitimate": 0,
                "phishing": 1,
            }
        )
    )

    # ---------------------------------------------------------
    # REMOVE NON-NUMERIC / TARGET COLUMNS
    # ---------------------------------------------------------

    excluded_columns = [
        "url",
        "status",
    ]

    feature_columns = [
        column
        for column in df.columns
        if column not in excluded_columns
        and pd.api.types.is_numeric_dtype(
            df[column]
        )
    ]

    X = df[
        feature_columns
    ].copy()

    print(
        "\nFeature count:",
        len(feature_columns)
    )

    print(
        "\nFeatures:"
    )

    for feature in feature_columns:

        print(
            f"- {feature}"
        )

    # ---------------------------------------------------------
    # REMOVE CONSTANT FEATURES
    # ---------------------------------------------------------

    constant_features = [
        column
        for column in X.columns
        if X[column].nunique() <= 1
    ]

    if constant_features:

        print(
            "\nRemoving constant features:"
        )

        for feature in constant_features:

            print(
                f"- {feature}"
            )

        X = X.drop(
            columns=constant_features
        )

    print(
        "\nFinal feature count:",
        X.shape[1]
    )

    # ---------------------------------------------------------
    # TRAIN TEST SPLIT
    # ---------------------------------------------------------

    print(
        "\nCreating stratified train/test split..."
    )

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42,
            stratify=y,
        )
    )

    print(
        "Training samples:",
        len(X_train)
    )

    print(
        "Testing samples :",
        len(X_test)
    )

    # ---------------------------------------------------------
    # MODELS
    # ---------------------------------------------------------

    models = [

        (
            "Logistic Regression",
            Pipeline(
                [
                    (
                        "scaler",
                        StandardScaler()
                    ),
                    (
                        "classifier",
                        LogisticRegression(
                            max_iter=2000,
                            random_state=42,
                        ),
                    ),
                ]
            ),
        ),

        (
            "Random Forest",
            RandomForestClassifier(
                n_estimators=300,
                random_state=42,
                n_jobs=-1,
            ),
        ),
    ]

    # ---------------------------------------------------------
    # TRAIN
    # ---------------------------------------------------------

    results = []

    for name, model in models:

        result = evaluate_model(
            name,
            model,
            X_train,
            X_test,
            y_train,
            y_test,
        )

        results.append(
            result
        )

    # ---------------------------------------------------------
    # RESULTS
    # ---------------------------------------------------------

    results_df = pd.DataFrame(
        results
    )

    print("\n" + "=" * 70)
    print("FULL FEATURE EXPERIMENT RESULTS")
    print("=" * 70)

    print(
        results_df.to_string(
            index=False
        )
    )

    # ---------------------------------------------------------
    # SAVE
    # ---------------------------------------------------------

    os.makedirs(
        "reports",
        exist_ok=True
    )

    report = {
        "dataset_samples": int(len(df)),
        "feature_count": int(X.shape[1]),
        "features": list(X.columns),
        "constant_features_removed": constant_features,
        "training_samples": int(len(X_train)),
        "testing_samples": int(len(X_test)),
        "results": results,
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
        "\nExperiment report saved to:"
    )

    print(
        REPORT_FILE
    )

    print("\n" + "=" * 70)
    print("FULL FEATURE EXPERIMENT COMPLETE")
    print("=" * 70)

    print(
        "\nProduction URL model remains unchanged."
    )


if __name__ == "__main__":
    main()
