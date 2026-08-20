import os
import sys
import joblib
import pandas as pd


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

MODEL_FILE = os.path.join(
    PROJECT_ROOT,
    "models",
    "realtime_url_top25_model.joblib"
)


# ============================================================
# VALIDATED TOP-25 FEATURES
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
# MODEL LOADING
# ============================================================

def load_model():
    """
    Load the saved Top-25 model package.

    The saved file contains:

        {
            "model": RandomForestClassifier,
            "features": [...],
            "threshold": 0.45,
            ...
        }
    """

    if not os.path.exists(MODEL_FILE):

        raise FileNotFoundError(
            f"Model file not found:\n{MODEL_FILE}\n\n"
            "Run train_realtime_top25_model.py first."
        )

    model_package = joblib.load(
        MODEL_FILE
    )

    # --------------------------------------------------------
    # Validate package
    # --------------------------------------------------------

    if not isinstance(
        model_package,
        dict
    ):

        raise TypeError(
            "Saved model is not a valid model package."
        )

    required_keys = [
        "model",
        "features",
        "threshold",
    ]

    missing_keys = [
        key
        for key in required_keys
        if key not in model_package
    ]

    if missing_keys:

        raise ValueError(
            "Saved model package is missing: "
            + ", ".join(missing_keys)
        )

    # --------------------------------------------------------
    # Validate feature configuration
    # --------------------------------------------------------

    saved_features = model_package[
        "features"
    ]

    if saved_features != TOP25_FEATURES:

        raise ValueError(
            "Saved model feature configuration "
            "does not match the validated Top-25 list."
        )

    # --------------------------------------------------------
    # Validate threshold
    # --------------------------------------------------------

    saved_threshold = float(
        model_package[
            "threshold"
        ]
    )

    if saved_threshold != 0.45:

        raise ValueError(
            f"Unexpected model threshold: "
            f"{saved_threshold}. "
            f"Expected 0.45."
        )

    return model_package


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_top25_features(url):
    """
    Extract the validated Top-25 feature vector
    directly from the raw URL.

    No external services are used.
    """

    url = str(url).strip()

    if not url:

        raise ValueError(
            "URL cannot be empty."
        )

    # --------------------------------------------------------
    # Extract all locally available lexical features
    # --------------------------------------------------------

    lexical = extract_lexical_features(
        url
    )

    # --------------------------------------------------------
    # Check missing feature names
    # --------------------------------------------------------

    missing = [
        feature
        for feature in TOP25_FEATURES
        if feature not in lexical
    ]

    if missing:

        raise ValueError(
            "Extractor is missing required features: "
            + ", ".join(missing)
        )

    # --------------------------------------------------------
    # Check unavailable values
    # --------------------------------------------------------

    unavailable = [
        feature
        for feature in TOP25_FEATURES
        if lexical[feature] is None
    ]

    if unavailable:

        raise ValueError(
            "Required Top-25 features are unavailable: "
            + ", ".join(unavailable)
        )

    # --------------------------------------------------------
    # Return ordered feature dictionary
    # --------------------------------------------------------

    return {
        feature: lexical[feature]
        for feature in TOP25_FEATURES
    }


# ============================================================
# PREDICTION
# ============================================================

def predict_url(
    url,
    model_package=None
):
    """
    Predict whether a URL is phishing or legitimate.

    Parameters
    ----------
    url : str
        Raw URL.

    model_package : dict, optional
        Previously loaded model package.

    Returns
    -------
    dict
        Prediction result containing:

        - url
        - phishing_probability
        - threshold
        - prediction
        - risk
    """

    url = str(url).strip()

    if not url:

        raise ValueError(
            "URL cannot be empty."
        )

    # --------------------------------------------------------
    # Load model package
    # --------------------------------------------------------

    if model_package is None:

        model_package = load_model()

    # --------------------------------------------------------
    # Validate model package
    # --------------------------------------------------------

    if not isinstance(
        model_package,
        dict
    ):

        raise TypeError(
            "Model package must be a dictionary."
        )

    if "model" not in model_package:

        raise ValueError(
            "Model package does not contain 'model'."
        )

    if "features" not in model_package:

        raise ValueError(
            "Model package does not contain 'features'."
        )

    if "threshold" not in model_package:

        raise ValueError(
            "Model package does not contain 'threshold'."
        )

    # --------------------------------------------------------
    # Get actual Random Forest classifier
    # --------------------------------------------------------

    classifier = model_package[
        "model"
    ]

    features = model_package[
        "features"
    ]

    threshold = float(
        model_package[
            "threshold"
        ]
    )

    # --------------------------------------------------------
    # Extract features
    # --------------------------------------------------------

    extracted_features = (
        extract_top25_features(url)
    )

    # --------------------------------------------------------
    # Create DataFrame
    # --------------------------------------------------------

    X = pd.DataFrame(
        [
            [
                extracted_features[
                    feature
                ]
                for feature in features
            ]
        ],
        columns=features
    )

    # --------------------------------------------------------
    # Generate phishing probability
    # --------------------------------------------------------

    probability = float(
        classifier.predict_proba(
            X
        )[0][1]
    )

    # --------------------------------------------------------
    # Apply validated threshold
    # --------------------------------------------------------

    if probability >= threshold:

        prediction = "phishing"

    else:

        prediction = "legitimate"

    # --------------------------------------------------------
    # Risk classification
    # --------------------------------------------------------

    if probability >= 0.75:

        risk = "HIGH"

    elif probability >= threshold:

        risk = "MEDIUM"

    else:

        risk = "LOW"

    # --------------------------------------------------------
    # Return result
    # --------------------------------------------------------

    return {
        "url": url,

        "phishing_probability": round(
            probability,
            4
        ),

        "threshold": threshold,

        "prediction": prediction,

        "risk": risk,
    }


# ============================================================
# DISPLAY RESULT
# ============================================================

def print_prediction_result(
    result,
    test_number=None
):
    """
    Print a clean prediction result.
    """

    if test_number is not None:

        print("-" * 70)

        print(
            f"TEST {test_number}"
        )

        print("-" * 70)

    print()

    print(
        "URL:"
    )

    print(
        result["url"]
    )

    print()

    print(
        "Phishing probability:",
        result[
            "phishing_probability"
        ]
    )

    print(
        "Decision threshold  :",
        result[
            "threshold"
        ]
    )

    print(
        "Prediction          :",
        result[
            "prediction"
        ]
    )

    print(
        "Risk                :",
        result[
            "risk"
        ]
    )

    print()


# ============================================================
# TEST URLS
# ============================================================

TEST_URLS = [

    # Legitimate
    "https://www.google.com",

    # Legitimate
    "https://github.com/python/cpython/issues",

    # Suspicious IP-based URL
    "http://192.168.1.25/login",

    # Suspicious login URL
    "http://secure-login.example.com/account/verify",

    # Suspicious long URL
    "http://secure-login.example.com/account/verify/update?session=983472983472&token=928374928374",
]


# ============================================================
# MAIN TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 70)

    print(
        "REAL-TIME URL PREDICTION TEST"
    )

    print("=" * 70)

    print()

    try:

        # ----------------------------------------------------
        # Load saved model package
        # ----------------------------------------------------

        model_package = load_model()

        print(
            "Loaded model successfully."
        )

        print(
            "Model type :",
            type(
                model_package[
                    "model"
                ]
            ).__name__
        )

        print(
            "Features   :",
            len(
                model_package[
                    "features"
                ]
            )
        )

        print(
            "Threshold  :",
            model_package[
                "threshold"
            ]
        )

        print(
            "External reputation:",
            model_package.get(
                "external_reputation",
                "unknown"
            )
        )

        print()

        # ----------------------------------------------------
        # Test URLs
        # ----------------------------------------------------

        for number, url in enumerate(
            TEST_URLS,
            start=1
        ):

            try:

                result = predict_url(
                    url,
                    model_package
                )

                print_prediction_result(
                    result,
                    number
                )

            except Exception as error:

                print("-" * 70)

                print(
                    f"TEST {number}"
                )

                print("-" * 70)

                print(
                    "URL:"
                )

                print(
                    url
                )

                print()

                print(
                    "ERROR:",
                    error
                )

                print()

    except Exception as error:

        print(
            "MODEL LOADING ERROR:"
        )

        print(
            error
        )

    print("=" * 70)

    print(
        "REAL-TIME URL PREDICTION TEST COMPLETE"
    )

    print("=" * 70)
