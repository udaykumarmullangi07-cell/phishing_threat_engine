import joblib
import pandas as pd

from src.url_features import (
    extract_url_features,
    explain_url_features,
)

from src.risk_fusion import (
    calculate_risk_score,
    classify_risk,
    generate_explanation,
)


# =========================================================
# MODEL PATHS
# =========================================================

TEXT_MODEL_FILE = (
    "models/text_logistic_regression.joblib"
)

TEXT_VECTORIZER_FILE = (
    "models/text_tfidf_vectorizer.joblib"
)

URL_MODEL_FILE = (
    "models/url_random_forest.joblib"
)


# =========================================================
# URL FEATURE ORDER
#
# IMPORTANT:
# This order MUST match the order used during
# URL model training.
# =========================================================

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
# LOAD TRAINED MODELS
# =========================================================

def load_models():

    text_model = joblib.load(
        TEXT_MODEL_FILE
    )

    text_vectorizer = joblib.load(
        TEXT_VECTORIZER_FILE
    )

    url_model = joblib.load(
        URL_MODEL_FILE
    )

    return (
        text_model,
        text_vectorizer,
        url_model,
    )


# =========================================================
# TEXT PREDICTION
# =========================================================

def predict_text(
    text,
    text_model,
    text_vectorizer,
):
    """
    Predict phishing probability for an email/message.

    Returns:
        float:
            Probability that the message is phishing.
    """

    text = str(text)

    # Convert text using the trained TF-IDF vectorizer
    text_vector = text_vectorizer.transform(
        [text]
    )

    # Get phishing probability
    probability = text_model.predict_proba(
        text_vector
    )[0][1]

    return float(probability)


# =========================================================
# URL PREDICTION
# =========================================================

def predict_url(
    url,
    url_model,
):
    """
    Predict phishing probability for a URL.
    """

    # -----------------------------------------------------
    # Extract URL features
    # -----------------------------------------------------

    features = extract_url_features(
        url
    )

    # -----------------------------------------------------
    # Create DataFrame
    #
    # The feature names and order must match training.
    # -----------------------------------------------------

    feature_values = pd.DataFrame(
        [[
            features["url_length"],
            features["dot_count"],
            features["has_https"],
            features["has_ip_address"],
            features["special_chars"],
            features["subdomain_count"],
            features["path_depth"],
            features["digit_ratio"],
            features["has_at_symbol"],
            features["domain_length"],
        ]],
        columns=URL_FEATURE_COLUMNS,
    )

    # -----------------------------------------------------
    # Predict phishing probability
    # -----------------------------------------------------

    probability = url_model.predict_proba(
        feature_values
    )[0][1]

    return float(probability)


# =========================================================
# COMPLETE THREAT ANALYSIS
# =========================================================

def analyze_threat(
    text=None,
    url=None,
):
    """
    Analyze an email, a URL, or both.

    At least one of text or URL must be provided.

    Returns:
        Dictionary containing:

        text_probability
        url_probability
        risk_score
        threat_level
        explanation
        indicators
        url_features
    """

    # =====================================================
    # INPUT VALIDATION
    # =====================================================

    # Treat empty strings as missing input too.
    text_is_empty = (
        text is None
        or not str(text).strip()
    )

    url_is_empty = (
        url is None
        or not str(url).strip()
    )

    if text_is_empty and url_is_empty:

        raise ValueError(
            "Provide at least an email message or a URL."
        )

    # =====================================================
    # LOAD MODELS
    # =====================================================

    (
        text_model,
        text_vectorizer,
        url_model,
    ) = load_models()

    # =====================================================
    # TEXT MODEL
    # =====================================================

    text_probability = None

    if not text_is_empty:

        text_probability = predict_text(
            text,
            text_model,
            text_vectorizer,
        )

    # =====================================================
    # URL MODEL
    # =====================================================

    url_probability = None

    url_indicators = []

    url_features = None

    if not url_is_empty:

        # -------------------------------------------------
        # URL prediction
        # -------------------------------------------------

        url_probability = predict_url(
            url,
            url_model,
        )

        # -------------------------------------------------
        # URL explanation
        # -------------------------------------------------

        url_analysis = explain_url_features(
            url
        )

        url_features = (
            url_analysis["features"]
        )

        url_indicators = (
            url_analysis["indicators"]
        )

    # =====================================================
    # RISK FUSION
    # =====================================================

    risk_score = calculate_risk_score(
        text_probability,
        url_probability,
    )

    # Convert NumPy float to normal Python float
    risk_score = float(risk_score)

    # =====================================================
    # THREAT LEVEL
    # =====================================================

    threat_level = classify_risk(
        risk_score
    )

    # =====================================================
    # GENERAL EXPLANATION
    # =====================================================

    explanation = generate_explanation(
        text_probability,
        url_probability,
        risk_score,
        threat_level,
    )

    # =====================================================
    # ADD URL INDICATORS TO EXPLANATION
    # =====================================================

    if url_indicators:

        explanation += (
            " Indicators: "
            + "; ".join(
                url_indicators
            )
            + "."
        )

    # =====================================================
    # FINAL STRUCTURED RESULT
    # =====================================================

    return {

        # Model probabilities
        "text_probability":
            text_probability,

        "url_probability":
            url_probability,

        # Final risk assessment
        "risk_score":
            risk_score,

        "threat_level":
            threat_level,

        # Explainability
        "explanation":
            explanation,

        "indicators":
            url_indicators,

        # Detailed URL information
        "url_features":
            url_features,
    }


# =========================================================
# DISPLAY HELPER
# =========================================================

def format_probability(probability):
    """
    Format a probability safely.

    None -> "None"
    Number -> "0.xxxx"
    """

    if probability is None:

        return "None"

    return f"{float(probability):.4f}"


# =========================================================
# PRINT ANALYSIS RESULT
# =========================================================

def print_result(
    result,
):

    print(
        "\nText phishing probability:",
        format_probability(
            result["text_probability"]
        ),
    )

    print(
        "URL phishing probability: ",
        format_probability(
            result["url_probability"]
        ),
    )

    print(
        "Combined risk score:        ",
        f"{result['risk_score']:.4f}",
    )

    print(
        "Threat level:               ",
        result["threat_level"],
    )

    # -----------------------------------------------------
    # Explanation
    # -----------------------------------------------------

    print(
        "\nExplanation:"
    )

    print(
        result["explanation"]
    )

    # -----------------------------------------------------
    # URL indicators
    # -----------------------------------------------------

    print(
        "\nURL Indicators:"
    )

    if result["indicators"]:

        for indicator in result[
            "indicators"
        ]:

            print(
                f"- {indicator}"
            )

    else:

        print(
            "No obvious suspicious "
            "URL indicators detected."
        )

    # -----------------------------------------------------
    # Structured result
    # -----------------------------------------------------

    print(
        "\nStructured Result:"
    )

    print(
        result
    )


# =========================================================
# TEST THE COMPLETE ENGINE
# =========================================================

if __name__ == "__main__":

    test_cases = [

        # -------------------------------------------------
        # Test 1 — Email only
        # -------------------------------------------------

        {
            "name":
                "Email Only — Suspicious Account",

            "text":
                """
                URGENT SECURITY ALERT

                Your account has been temporarily suspended
                due to unusual login activity.

                You must verify your account immediately
                to prevent permanent suspension.

                Click the verification link and confirm
                your username, password and account details.

                Failure to complete verification within 24 hours
                will result in account closure.
                """,

            "url":
                None,
        },

        # -------------------------------------------------
        # Test 2 — URL only
        # -------------------------------------------------

        {
            "name":
                "URL Only — Suspicious URL",

            "text":
                None,

            "url":
                (
                    "http://192.168.1.10/"
                    "login?verify=12345"
                    "&token=98765"
                ),
        },

        # -------------------------------------------------
        # Test 3 — Email + URL
        # -------------------------------------------------

        {
            "name":
                "Email + URL — Combined",

            "text":
                """
                Your account has been flagged for suspicious activity.
                Please verify your identity immediately to avoid
                account suspension.
                """,

            "url":
                (
                    "http://secure-login.example.com/"
                    "account/verify?id=12345"
                ),
        },

        # -------------------------------------------------
        # Test 4 — Normal email + normal URL
        # -------------------------------------------------

        {
            "name":
                "Normal Email + Normal URL",

            "text":
                """
                Hi John,

                The project meeting has been moved to 3 PM tomorrow.
                Please review the attached agenda before the meeting.

                Thanks.
                """,

            "url":
                "https://www.google.com",
        },

        # -------------------------------------------------
        # Test 5 — Empty input
        # -------------------------------------------------

        {
            "name":
                "Empty Input",

            "text":
                None,

            "url":
                None,
        },
    ]

    # =====================================================
    # RUN TEST CASES
    # =====================================================

    for test_case in test_cases:

        print(
            "\n" + "=" * 70
        )

        print(
            test_case["name"]
        )

        print(
            "=" * 70
        )

        try:

            result = analyze_threat(

                text=test_case[
                    "text"
                ],

                url=test_case[
                    "url"
                ],
            )

            print_result(
                result
            )

        except ValueError as error:

            print(
                "\nValidation error:",
                error
            )
