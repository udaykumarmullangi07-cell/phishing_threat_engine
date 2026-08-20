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
REPORT_FILE = "reports/url_feature_group_ablation.json"

RANDOM_STATE = 42


# ============================================================
# FEATURE GROUPS
# ============================================================

# These are URL lexical / structural characteristics.
LEXICAL_STRUCTURAL = [
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


# Features describing webpage/content behaviour.
PAGE_CONTENT = [
    "nb_redirection",
    "nb_external_redirection",
    "nb_hyperlinks",
    "ratio_intHyperlinks",
    "ratio_extHyperlinks",
    "ratio_nullHyperlinks",
    "nb_extCSS",
    "ratio_intRedirection",
    "ratio_extRedirection",
    "ratio_intErrors",
    "ratio_extErrors",
    "login_form",
    "external_favicon",
    "links_in_tags",
    "submit_email",
    "ratio_intMedia",
    "ratio_extMedia",
    "sfh",
    "iframe",
    "popup_window",
    "safe_anchor",
    "onmouseover",
    "right_clic",
    "empty_title",
    "domain_in_title",
    "domain_with_copyright",
]


# Features that depend more strongly on domain / reputation
# / external information.
REPUTATION_DOMAIN = [
    "whois_registered_domain",
    "domain_registration_length",
    "domain_age",
    "web_traffic",
    "dns_record",
    "google_index",
    "page_rank",
]


# ============================================================
# HELPER
# ============================================================

def evaluate_model(X, y, group_name):
    """
    Train and evaluate Random Forest for one feature group.
    """

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    model = RandomForestClassifier(
        n_estimators=300,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight="balanced",
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    tn, fp, fn, tp = confusion_matrix(
        y_test,
        predictions,
    ).ravel()

    results = {
        "feature_group": group_name,
        "feature_count": X.shape[1],
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

    print("\n" + "-" * 70)
    print(group_name)
    print("-" * 70)

    print(f"Features : {X.shape[1]}")
    print(f"Accuracy : {results['accuracy']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall   : {results['recall']:.4f}")
    print(f"F1       : {results['f1']:.4f}")
    print(f"ROC-AUC  : {results['roc_auc']:.4f}")
    print(f"False Positives: {fp}")
    print(f"False Negatives: {fn}")

    return results


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("URL MODEL — FEATURE GROUP ABLATION EXPERIMENT")
    print("=" * 70)

    print("""
Purpose:
Determine which URL feature groups contribute most strongly
to phishing detection.

Production URL model will NOT be modified.

Feature groups:
1. Lexical / Structural
2. Page / Content
3. Reputation / Domain
4. Full feature set
""")

    # --------------------------------------------------------
    # LOAD DATASET
    # --------------------------------------------------------

    print("\nLoading dataset...")

    df = pd.read_csv(INPUT_FILE)

    print(f"Dataset samples: {len(df)}")

    # --------------------------------------------------------
    # LABEL
    # --------------------------------------------------------

    y = (
        df["status"]
        .map({
            "legitimate": 0,
            "phishing": 1,
        })
    )

    if y.isnull().any():
        raise ValueError(
            "Unexpected labels found in status column."
        )

    y = y.astype(int)

    # --------------------------------------------------------
    # VERIFY FEATURES
    # --------------------------------------------------------

    all_groups = {
        "Lexical / Structural": LEXICAL_STRUCTURAL,
        "Page / Content": PAGE_CONTENT,
        "Reputation / Domain": REPUTATION_DOMAIN,
    }

    print("\nFeature group sizes:")

    for name, features in all_groups.items():

        missing = [
            f for f in features
            if f not in df.columns
        ]

        if missing:
            raise ValueError(
                f"{name} missing features: {missing}"
            )

        print(
            f"{name}: {len(features)} features"
        )

    # --------------------------------------------------------
    # REMOVE CONSTANT FEATURES FROM EACH GROUP
    # --------------------------------------------------------

    usable_groups = {}

    print("\nChecking constant features...")

    for name, features in all_groups.items():

        usable = []

        for feature in features:

            if df[feature].nunique() > 1:
                usable.append(feature)

        removed = sorted(
            set(features) - set(usable)
        )

        print(f"\n{name}")

        print(
            f"Usable features: {len(usable)}"
        )

        if removed:
            print(
                "Removed constant features:",
                ", ".join(removed)
            )

        usable_groups[name] = usable

    # --------------------------------------------------------
    # FULL FEATURE SET
    # --------------------------------------------------------

    full_features = (
        usable_groups["Lexical / Structural"]
        + usable_groups["Page / Content"]
        + usable_groups["Reputation / Domain"]
    )

    # Remove duplicates while preserving order
    full_features = list(
        dict.fromkeys(full_features)
    )

    print(
        f"\nFull usable feature count: "
        f"{len(full_features)}"
    )

    # --------------------------------------------------------
    # RUN EXPERIMENTS
    # --------------------------------------------------------

    results = []

    for name, features in usable_groups.items():

        X = df[features]

        result = evaluate_model(
            X,
            y,
            name,
        )

        results.append(result)

    # --------------------------------------------------------
    # COMBINATIONS
    # --------------------------------------------------------

    combinations = {

        "Lexical + Page":
            usable_groups["Lexical / Structural"]
            + usable_groups["Page / Content"],

        "Lexical + Reputation":
            usable_groups["Lexical / Structural"]
            + usable_groups["Reputation / Domain"],

        "Page + Reputation":
            usable_groups["Page / Content"]
            + usable_groups["Reputation / Domain"],

        "All Feature Groups":
            full_features,
    }

    for name, features in combinations.items():

        features = list(
            dict.fromkeys(features)
        )

        X = df[features]

        result = evaluate_model(
            X,
            y,
            name,
        )

        results.append(result)

    # --------------------------------------------------------
    # RESULTS TABLE
    # --------------------------------------------------------

    results_df = pd.DataFrame(results)

    print("\n")
    print("=" * 70)
    print("FEATURE GROUP ABLATION RESULTS")
    print("=" * 70)

    print(
        results_df[
            [
                "feature_group",
                "feature_count",
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
    # BEST MODELS
    # --------------------------------------------------------

    best_f1 = results_df.loc[
        results_df["f1"].idxmax()
    ]

    best_auc = results_df.loc[
        results_df["roc_auc"].idxmax()
    ]

    print("\n")
    print("=" * 70)
    print("BEST CONFIGURATIONS")
    print("=" * 70)

    print("\nBest F1:")
    print(
        f"Feature group: {best_f1['feature_group']}"
    )
    print(
        f"F1: {best_f1['f1']:.4f}"
    )

    print("\nBest ROC-AUC:")
    print(
        f"Feature group: {best_auc['feature_group']}"
    )
    print(
        f"ROC-AUC: {best_auc['roc_auc']:.4f}"
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
        "random_state": RANDOM_STATE,
        "feature_groups": {
            name: features
            for name, features
            in usable_groups.items()
        },
        "full_feature_count": len(full_features),
        "results": results,
        "best_f1": {
            "feature_group":
                best_f1["feature_group"],
            "f1":
                float(best_f1["f1"]),
        },
        "best_roc_auc": {
            "feature_group":
                best_auc["feature_group"],
            "roc_auc":
                float(best_auc["roc_auc"]),
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
        "Report saved to:"
    )
    print(
        REPORT_FILE
    )

    print("\n")
    print("=" * 70)
    print("FEATURE GROUP ABLATION COMPLETE")
    print("=" * 70)

    print(
        "\nProduction URL model remains unchanged."
    )


if __name__ == "__main__":
    main()
