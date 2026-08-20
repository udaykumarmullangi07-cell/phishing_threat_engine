import os
import sys
import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

SRC_DIR = os.path.join(
    PROJECT_ROOT,
    "src"
)

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)


from url_experimental_features import (
    extract_lexical_features
)


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = os.path.join(
    PROJECT_ROOT,
    "data",
    "url_dataset.csv"
)

MODEL_FILE = os.path.join(
    PROJECT_ROOT,
    "models",
    "realtime_url_top25_model.joblib"
)

RANDOM_STATE = 123

TEST_SIZE = 0.20

THRESHOLD = 0.45


# ============================================================
# VALIDATED TOP-25 FEATURE SET
# ============================================================

TOP25_FEATURES = [
    "nb_www",
    "phish_hints",
    "nb_slash",
    "nb_hyphens",
    "nb_subdomains",
    "ratio_digits_host",
    "char_repeat",
    "ratio_digits_url",
    "longest_words_raw",
    "length_hostname",
    "nb_underscore",
    "https_token",
    "nb_dots",
    "longest_word_host",
    "avg_word_host",
    "nb_com",
    "shortest_word_host",
    "length_words_raw",
    "path_extension",
    "avg_word_path",
    "prefix_suffix",
    "longest_word_path",
    "shortest_words_raw",
    "nb_qm",
    "nb_eq",
]


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_dataset_features(df):

    rows = []

    print("Extracting features from raw URLs...")

    for index, url in enumerate(df["url"]):

        if index % 1000 == 0:
            print(
                f"Processed: {index}/{len(df)}"
            )

        features = extract_lexical_features(url)

        row = []

        for feature in TOP25_FEATURES:

            if feature not in features:
                raise ValueError(
                    f"Missing feature: {feature}"
                )

            value = features[feature]

            if value is None:
                raise ValueError(
                    f"Feature '{feature}' "
                    f"is unavailable for URL: {url}"
                )

            row.append(value)

        rows.append(row)

    return pd.DataFrame(
        rows,
        columns=TOP25_FEATURES
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("FINAL REAL-TIME URL MODEL — TOP 25")
    print("=" * 70)

    print()
    print("Purpose:")
    print(
        "Train and save the validated Top-25 "
        "real-time URL Random Forest."
    )

    print()
    print("Configuration:")
    print("Features  :", len(TOP25_FEATURES))
    print("Threshold :", THRESHOLD)
    print("Random state:", RANDOM_STATE)

    print()
    print("External services:")
    print("- WHOIS: NO")
    print("- DNS reputation: NO")
    print("- Google index: NO")
    print("- PageRank: NO")
    print("- Web traffic: NO")
    print("- Webpage fetching: NO")
    print("- Brand database: NO")

    # --------------------------------------------------------
    # LOAD DATASET
    # --------------------------------------------------------

    print()
    print("-" * 70)
    print("LOADING DATASET")
    print("-" * 70)

    df = pd.read_csv(
        INPUT_FILE
    )

    print(
        f"Dataset samples: {len(df)}"
    )

    # --------------------------------------------------------
    # LABEL VALIDATION
    # --------------------------------------------------------

    if "status" not in df.columns:
        raise ValueError(
            "Dataset does not contain 'status' column."
        )

    print()
    print("Class distribution:")

    print(
        df["status"]
        .value_counts()
        .to_string()
    )

    # Convert labels
    y = (
        df["status"]
        .map(
            {
                "legitimate": 0,
                "phishing": 1,
            }
        )
    )

    if y.isna().any():
        raise ValueError(
            "Unknown values found in status column."
        )

    y = y.astype(int)

    # --------------------------------------------------------
    # EXTRACT TOP-25 FEATURES
    # --------------------------------------------------------

    print()
    print("-" * 70)
    print("EXTRACTING TOP-25 FEATURES")
    print("-" * 70)

    X = extract_dataset_features(
        df
    )

    print()
    print(
        f"Samples : {len(X)}"
    )

    print(
        f"Features: {len(X.columns)}"
    )

    print(
        f"Missing : {X.isna().sum().sum()}"
    )

    # --------------------------------------------------------
    # TRAIN / TEST SPLIT
    # --------------------------------------------------------

    print()
    print("-" * 70)
    print("CREATING STRATIFIED SPLIT")
    print("-" * 70)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print(
        f"Training samples: {len(X_train)}"
    )

    print(
        f"Testing samples : {len(X_test)}"
    )

    # --------------------------------------------------------
    # TRAIN MODEL
    # --------------------------------------------------------

    print()
    print("-" * 70)
    print("TRAINING FINAL RANDOM FOREST")
    print("-" * 70)

    model = RandomForestClassifier(
        n_estimators=300,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight="balanced",
    )

    model.fit(
        X_train,
        y_train
    )

    print(
        "Training completed."
    )

    # --------------------------------------------------------
    # PROBABILITY
    # --------------------------------------------------------

    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    # Use validated threshold
    predictions = (
        probabilities >= THRESHOLD
    ).astype(int)

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_test,
        probabilities
    )

    tn, fp, fn, tp = confusion_matrix(
        y_test,
        predictions
    ).ravel()

    # --------------------------------------------------------
    # RESULTS
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("FINAL TOP-25 MODEL RESULTS")
    print("=" * 70)

    print(
        f"Features : {len(TOP25_FEATURES)}"
    )

    print(
        f"Threshold: {THRESHOLD}"
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

    print(
        f"True Negatives : {tn}"
    )

    print(
        f"True Positives : {tp}"
    )

    # --------------------------------------------------------
    # FEATURE IMPORTANCE
    # --------------------------------------------------------

    importance_df = pd.DataFrame(
        {
            "feature": TOP25_FEATURES,
            "importance": model.feature_importances_,
        }
    ).sort_values(
        "importance",
        ascending=False
    )

    print()
    print("=" * 70)
    print("TOP-25 MODEL FEATURE IMPORTANCE")
    print("=" * 70)

    print(
        importance_df.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # SAVE MODEL
    # --------------------------------------------------------

    os.makedirs(
        os.path.dirname(MODEL_FILE),
        exist_ok=True
    )

    model_package = {
        "model": model,
        "features": TOP25_FEATURES,
        "threshold": THRESHOLD,
        "random_state": RANDOM_STATE,
        "model_type": "RandomForestClassifier",
        "feature_count": len(TOP25_FEATURES),
        "external_reputation": False,
    }

    joblib.dump(
        model_package,
        MODEL_FILE
    )

    print()
    print("=" * 70)
    print("MODEL SAVED")
    print("=" * 70)

    print()
    print(
        MODEL_FILE
    )

    print()
    print("Saved configuration:")
    print(
        "Feature count       :",
        len(TOP25_FEATURES)
    )

    print(
        "Decision threshold   :",
        THRESHOLD
    )

    print(
        "External reputation  :",
        "Disabled"
    )

    print()
    print("=" * 70)
    print("FINAL TOP-25 MODEL TRAINING COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
