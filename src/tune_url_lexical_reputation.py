import json
import os

import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = "data/url_dataset.csv"
REPORT_FILE = "reports/url_lexical_reputation_tuning.json"

RANDOM_STATE = 42


# ============================================================
# 59-FEATURE MODEL
# ============================================================

LEXICAL_FEATURES = [
    "length_url",
    "length_hostname",
    "ip",
    "nb_dots",
    "nb_hyphens",
    "nb_at",
    "nb_qm",
    "nb_and",
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
]

REPUTATION_FEATURES = [
    "whois_registered_domain",
    "domain_registration_length",
    "domain_age",
    "web_traffic",
    "dns_record",
    "google_index",
    "page_rank",
]

FEATURES = list(
    dict.fromkeys(
        LEXICAL_FEATURES + REPUTATION_FEATURES
    )
)


# ============================================================
# EVALUATION FUNCTION
# ============================================================

def evaluate_model(model, X_test, y_test):

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    tn, fp, fn, tp = confusion_matrix(
        y_test,
        predictions
    ).ravel()

    return {
        "accuracy": float(
            accuracy_score(y_test, predictions)
        ),
        "precision": float(
            precision_score(y_test, predictions)
        ),
        "recall": float(
            recall_score(y_test, predictions)
        ),
        "f1": float(
            f1_score(y_test, predictions)
        ),
        "roc_auc": float(
            roc_auc_score(y_test, probabilities)
        ),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_negatives": int(tn),
        "true_positives": int(tp),
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("URL MODEL — LEXICAL + REPUTATION RANDOM FOREST TUNING")
    print("=" * 70)

    print("""
Purpose:
Optimize the 59-feature URL Random Forest model.

Production URL model will NOT be modified.

Baseline:
Lexical + Reputation
59 features
""")

    # --------------------------------------------------------
    # LOAD DATASET
    # --------------------------------------------------------

    print("\nLoading dataset...")

    df = pd.read_csv(INPUT_FILE)

    print(
        f"Dataset samples: {len(df)}"
    )

    # --------------------------------------------------------
    # VERIFY FEATURES
    # --------------------------------------------------------

    missing = [
        feature
        for feature in FEATURES
        if feature not in df.columns
    ]

    if missing:

        raise ValueError(
            f"Missing features: {missing}"
        )

    # Remove constant features
    usable_features = [
        feature
        for feature in FEATURES
        if df[feature].nunique() > 1
    ]

    print(
        f"Requested features: {len(FEATURES)}"
    )

    print(
        f"Usable features: {len(usable_features)}"
    )

    # --------------------------------------------------------
    # LABEL
    # --------------------------------------------------------

    y = (
        df["status"]
        .map({
            "legitimate": 0,
            "phishing": 1,
        })
        .astype(int)
    )

    X = df[usable_features]

    # --------------------------------------------------------
    # TRAIN / TEST SPLIT
    # --------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print(
        f"\nTraining samples: {len(X_train)}"
    )

    print(
        f"Testing samples : {len(X_test)}"
    )

    # --------------------------------------------------------
    # BASELINE
    # --------------------------------------------------------

    print("\n")
    print("-" * 70)
    print("BASELINE RANDOM FOREST")
    print("-" * 70)

    baseline = RandomForestClassifier(
        n_estimators=300,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight="balanced",
    )

    baseline.fit(
        X_train,
        y_train
    )

    baseline_results = evaluate_model(
        baseline,
        X_test,
        y_test,
    )

    print(
        f"Accuracy : {baseline_results['accuracy']:.4f}"
    )

    print(
        f"Precision: {baseline_results['precision']:.4f}"
    )

    print(
        f"Recall   : {baseline_results['recall']:.4f}"
    )

    print(
        f"F1       : {baseline_results['f1']:.4f}"
    )

    print(
        f"ROC-AUC  : {baseline_results['roc_auc']:.4f}"
    )

    print(
        f"False Positives: "
        f"{baseline_results['false_positives']}"
    )

    print(
        f"False Negatives: "
        f"{baseline_results['false_negatives']}"
    )

    # --------------------------------------------------------
    # HYPERPARAMETER CONFIGURATIONS
    # --------------------------------------------------------

    configurations = [

        {
            "name": "RF-500",
            "n_estimators": 500,
            "max_depth": None,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "max_features": "sqrt",
            "class_weight": "balanced",
        },

        {
            "name": "RF-500-depth20",
            "n_estimators": 500,
            "max_depth": 20,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "max_features": "sqrt",
            "class_weight": "balanced",
        },

        {
            "name": "RF-500-depth30",
            "n_estimators": 500,
            "max_depth": 30,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "max_features": "sqrt",
            "class_weight": "balanced",
        },

        {
            "name": "RF-leaf2",
            "n_estimators": 500,
            "max_depth": None,
            "min_samples_split": 2,
            "min_samples_leaf": 2,
            "max_features": "sqrt",
            "class_weight": "balanced",
        },

        {
            "name": "RF-leaf4",
            "n_estimators": 500,
            "max_depth": None,
            "min_samples_split": 2,
            "min_samples_leaf": 4,
            "max_features": "sqrt",
            "class_weight": "balanced",
        },

        {
            "name": "RF-split5",
            "n_estimators": 500,
            "max_depth": None,
            "min_samples_split": 5,
            "min_samples_leaf": 1,
            "max_features": "sqrt",
            "class_weight": "balanced",
        },

        {
            "name": "RF-split10",
            "n_estimators": 500,
            "max_depth": None,
            "min_samples_split": 10,
            "min_samples_leaf": 1,
            "max_features": "sqrt",
            "class_weight": "balanced",
        },

        {
            "name": "RF-maxfeatures-half",
            "n_estimators": 500,
            "max_depth": None,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "max_features": 0.5,
            "class_weight": "balanced",
        },

        {
            "name": "RF-maxfeatures-log2",
            "n_estimators": 500,
            "max_depth": None,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "max_features": "log2",
            "class_weight": "balanced",
        },

        {
            "name": "RF-balanced_subsample",
            "n_estimators": 500,
            "max_depth": None,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "max_features": "sqrt",
            "class_weight": "balanced_subsample",
        },

    ]

    # --------------------------------------------------------
    # RUN EXPERIMENTS
    # --------------------------------------------------------

    results = []

    for config in configurations:

        print("\n")
        print("-" * 70)
        print(
            f"TRAINING {config['name']}"
        )
        print("-" * 70)

        model = RandomForestClassifier(
            n_estimators=config["n_estimators"],
            max_depth=config["max_depth"],
            min_samples_split=config["min_samples_split"],
            min_samples_leaf=config["min_samples_leaf"],
            max_features=config["max_features"],
            class_weight=config["class_weight"],
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

        model.fit(
            X_train,
            y_train
        )

        metrics = evaluate_model(
            model,
            X_test,
            y_test,
        )

        result = {
            "configuration": config["name"],
            **config,
            **metrics,
        }

        results.append(result)

        print(
            f"Accuracy : {metrics['accuracy']:.4f}"
        )

        print(
            f"Precision: {metrics['precision']:.4f}"
        )

        print(
            f"Recall   : {metrics['recall']:.4f}"
        )

        print(
            f"F1       : {metrics['f1']:.4f}"
        )

        print(
            f"ROC-AUC  : {metrics['roc_auc']:.4f}"
        )

        print(
            f"False Positives: "
            f"{metrics['false_positives']}"
        )

        print(
            f"False Negatives: "
            f"{metrics['false_negatives']}"
        )

    # --------------------------------------------------------
    # RESULTS
    # --------------------------------------------------------

    results_df = pd.DataFrame(results)

    print("\n")
    print("=" * 70)
    print("URL RANDOM FOREST TUNING RESULTS")
    print("=" * 70)

    print(
        results_df[
            [
                "configuration",
                "accuracy",
                "precision",
                "recall",
                "f1",
                "roc_auc",
                "false_positives",
                "false_negatives",
            ]
        ].to_string(index=False)
    )

    # --------------------------------------------------------
    # BEST CONFIGURATIONS
    # --------------------------------------------------------

    best_f1 = results_df.loc[
        results_df["f1"].idxmax()
    ]

    best_auc = results_df.loc[
        results_df["roc_auc"].idxmax()
    ]

    best_recall = results_df.loc[
        results_df["recall"].idxmax()
    ]

    print("\n")
    print("=" * 70)
    print("BEST CONFIGURATIONS")
    print("=" * 70)

    print("\nBest F1:")
    print(
        f"{best_f1['configuration']}"
    )
    print(
        f"F1: {best_f1['f1']:.4f}"
    )

    print("\nBest ROC-AUC:")
    print(
        f"{best_auc['configuration']}"
    )
    print(
        f"ROC-AUC: {best_auc['roc_auc']:.4f}"
    )

    print("\nBest Recall:")
    print(
        f"{best_recall['configuration']}"
    )
    print(
        f"Recall: {best_recall['recall']:.4f}"
    )

    # --------------------------------------------------------
    # SAVE REPORT
    # --------------------------------------------------------

    os.makedirs(
        os.path.dirname(REPORT_FILE),
        exist_ok=True,
    )

    report = {
        "dataset_samples": int(len(df)),
        "feature_count": len(usable_features),
        "features": usable_features,
        "random_state": RANDOM_STATE,
        "baseline": {
            "configuration": "baseline",
            **baseline_results,
        },
        "experiments": results,
        "best_f1": {
            "configuration":
                best_f1["configuration"],
            "f1":
                float(best_f1["f1"]),
        },
        "best_roc_auc": {
            "configuration":
                best_auc["configuration"],
            "roc_auc":
                float(best_auc["roc_auc"]),
        },
        "best_recall": {
            "configuration":
                best_recall["configuration"],
            "recall":
                float(best_recall["recall"]),
        },
    }

    with open(
        REPORT_FILE,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            report,
            f,
            indent=4,
        )

    print("\n")
    print(
        "Tuning report saved to:"
    )

    print(
        REPORT_FILE
    )

    print("\n")
    print("=" * 70)
    print("URL RANDOM FOREST TUNING COMPLETE")
    print("=" * 70)

    print(
        "\nIMPORTANT:"
    )
    print(
        "Production URL model was NOT modified."
    )


if __name__ == "__main__":
    main()
