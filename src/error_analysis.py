import pandas as pd
import joblib

from sklearn.metrics import confusion_matrix


# =========================================================
# CONFIGURATION
# =========================================================

TEXT_DATA = "data/cleaned_text.csv"

TEXT_MODEL = "models/text_logistic_regression.joblib"

TEXT_VECTORIZER = (
    "models/text_tfidf_vectorizer.joblib"
)


URL_DATA = "data/url_features.csv"

URL_MODEL = "models/url_random_forest.joblib"


URL_FEATURE_COLUMNS = [
    "url_length",
    "dot_count",
    "has_https",
    "has_ip_address",
    "special_chars",
    "subdomain_count",
    "path_depth",
    "digit_ratio",
    "has_at_symbol",
    "domain_length",
]


# =========================================================
# TEXT ERROR ANALYSIS
# =========================================================

def analyze_text_errors():

    print("\n" + "=" * 70)
    print("TEXT MODEL ERROR ANALYSIS")
    print("=" * 70)

    df = pd.read_csv(
        TEXT_DATA
    )

    df = df.dropna(
        subset=[
            "processed_text",
            "label",
        ]
    ).copy()

    model = joblib.load(
        TEXT_MODEL
    )

    vectorizer = joblib.load(
        TEXT_VECTORIZER
    )

    X = vectorizer.transform(
        df["processed_text"]
    )

    y = df["label"]

    predictions = model.predict(
        X
    )

    probabilities = model.predict_proba(
        X
    )[:, 1]

    # -----------------------------------------------------
    # Add predictions to dataframe
    # -----------------------------------------------------

    df["prediction"] = predictions

    df["probability"] = probabilities

    # -----------------------------------------------------
    # False negatives
    #
    # Actual = phishing (1)
    # Predicted = legitimate (0)
    # -----------------------------------------------------

    false_negatives = df[
        (df["label"] == 1)
        & (df["prediction"] == 0)
    ]

    # -----------------------------------------------------
    # False positives
    #
    # Actual = legitimate (0)
    # Predicted = phishing (1)
    # -----------------------------------------------------

    false_positives = df[
        (df["label"] == 0)
        & (df["prediction"] == 1)
    ]

    print(
        f"\nFalse negatives: "
        f"{len(false_negatives)}"
    )

    print(
        f"False positives: "
        f"{len(false_positives)}"
    )

    # -----------------------------------------------------
    # Show most confident false negatives
    # -----------------------------------------------------

    print(
        "\nTop 5 false negatives:"
    )

    if len(false_negatives) > 0:

        samples = false_negatives.sort_values(
            "probability"
        ).head(5)

        for index, row in samples.iterrows():

            print("\n---")
            print(
                f"Probability: "
                f"{row['probability']:.4f}"
            )

            print(
                str(row["text_combined"])[:500]
            )

    # -----------------------------------------------------
    # Show most confident false positives
    # -----------------------------------------------------

    print(
        "\nTop 5 false positives:"
    )

    if len(false_positives) > 0:

        samples = false_positives.sort_values(
            "probability",
            ascending=False
        ).head(5)

        for index, row in samples.iterrows():

            print("\n---")
            print(
                f"Probability: "
                f"{row['probability']:.4f}"
            )

            print(
                str(row["text_combined"])[:500]
            )


# =========================================================
# URL ERROR ANALYSIS
# =========================================================

def analyze_url_errors():

    print("\n" + "=" * 70)
    print("URL MODEL ERROR ANALYSIS")
    print("=" * 70)

    df = pd.read_csv(
        URL_DATA
    )

    # Convert status into labels
    df["label"] = (
        df["status"] == "phishing"
    ).astype(int)

    model = joblib.load(
        URL_MODEL
    )

    X = df[
        URL_FEATURE_COLUMNS
    ]

    y = df["label"]

    predictions = model.predict(
        X
    )

    probabilities = model.predict_proba(
        X
    )[:, 1]

    # -----------------------------------------------------
    # Add predictions
    # -----------------------------------------------------

    df["prediction"] = predictions

    df["probability"] = probabilities

    # -----------------------------------------------------
    # False negatives
    # -----------------------------------------------------

    false_negatives = df[
        (df["label"] == 1)
        & (df["prediction"] == 0)
    ]

    # -----------------------------------------------------
    # False positives
    # -----------------------------------------------------

    false_positives = df[
        (df["label"] == 0)
        & (df["prediction"] == 1)
    ]

    print(
        f"\nFalse negatives: "
        f"{len(false_negatives)}"
    )

    print(
        f"False positives: "
        f"{len(false_positives)}"
    )

    # -----------------------------------------------------
    # URL false negatives
    # -----------------------------------------------------

    print(
        "\nTop 5 URL false negatives:"
    )

    if len(false_negatives) > 0:

        samples = false_negatives.sort_values(
            "probability"
        ).head(5)

        for index, row in samples.iterrows():

            print("\n---")

            print(
                f"Probability: "
                f"{row['probability']:.4f}"
            )

            print(
                f"URL: {row['url']}"
            )

    # -----------------------------------------------------
    # URL false positives
    # -----------------------------------------------------

    print(
        "\nTop 5 URL false positives:"
    )

    if len(false_positives) > 0:

        samples = false_positives.sort_values(
            "probability",
            ascending=False
        ).head(5)

        for index, row in samples.iterrows():

            print("\n---")

            print(
                f"Probability: "
                f"{row['probability']:.4f}"
            )

            print(
                f"URL: {row['url']}"
            )


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    analyze_text_errors()

    analyze_url_errors()
