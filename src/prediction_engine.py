import joblib
import pandas as pd

from src.realtime_url_features import (
    extract_realtime_url_features,
)

from src.risk_fusion import (
    calculate_risk_score,
    classify_risk,
    generate_explanation,
)


# =========================================================
# PRODUCTION MODEL PATHS
# =========================================================

TEXT_MODEL_FILE = (
    "models/final_text_model.joblib"
)

TEXT_VECTORIZER_FILE = (
    "models/final_text_vectorizer.joblib"
)

URL_MODEL_FILE = (
    "models/realtime_url_top25_model.joblib"
)


# =========================================================
# DEFAULT URL THRESHOLD
#
# The actual saved threshold is loaded from the model
# package. This value is only a fallback.
# =========================================================

DEFAULT_URL_THRESHOLD = 0.45


# =========================================================
# LOAD TRAINED MODELS
# =========================================================

def load_models():

    # -----------------------------------------------------
    # Load final text model
    # -----------------------------------------------------

    text_model = joblib.load(
        TEXT_MODEL_FILE
    )

    # -----------------------------------------------------
    # Load final text vectorizer
    # -----------------------------------------------------

    text_vectorizer = joblib.load(
        TEXT_VECTORIZER_FILE
    )

    # -----------------------------------------------------
    # Load URL model package
    # -----------------------------------------------------

    url_package = joblib.load(
        URL_MODEL_FILE
    )

    # -----------------------------------------------------
    # Validate URL package
    # -----------------------------------------------------

    if not isinstance(
        url_package,
        dict,
    ):

        raise TypeError(
            "Expected the real-time URL model "
            "to be a dictionary package."
        )

    required_keys = [
        "model",
        "features",
        "threshold",
    ]

    missing_keys = [
        key
        for key in required_keys
        if key not in url_package
    ]

    if missing_keys:

        raise ValueError(
            "URL model package is missing: "
            + ", ".join(missing_keys)
        )

    # -----------------------------------------------------
    # Extract actual Random Forest model
    # -----------------------------------------------------

    url_model = url_package["model"]

    # -----------------------------------------------------
    # Extract the exact feature order saved during training
    # -----------------------------------------------------

    url_features = list(
        url_package["features"]
    )

    # -----------------------------------------------------
    # Extract trained decision threshold
    # -----------------------------------------------------

    url_threshold = float(
        url_package["threshold"]
    )

    # -----------------------------------------------------
    # Safety validation
    # -----------------------------------------------------

    if not hasattr(
        url_model,
        "predict_proba",
    ):

        raise TypeError(
            "The stored URL model does not "
            "support predict_proba()."
        )

    if not url_features:

        raise ValueError(
            "The stored URL feature list is empty."
        )

    return {
        "text_model":
            text_model,

        "text_vectorizer":
            text_vectorizer,

        "url_model":
            url_model,

        "url_features":
            url_features,

        "url_threshold":
            url_threshold,

        "url_package":
            url_package,
    }


# =========================================================
# TEXT PREDICTION
# =========================================================

def predict_text(
    text,
    text_model,
    text_vectorizer,
):
    """
    Predict phishing probability for message/email text.

    Returns:
        float:
            Phishing probability between 0 and 1.
    """

    text = str(text)

    text_vector = (
        text_vectorizer.transform(
            [text]
        )
    )

    probability = (
        text_model.predict_proba(
            text_vector
        )[0][1]
    )

    return float(
        probability
    )


# =========================================================
# URL PREDICTION
# =========================================================

def predict_url(
    url,
    url_model,
    url_features,
):
    """
    Predict phishing probability using the final
    Top-25 real-time URL model.

    Features are extracted directly from the raw URL.

    No:
        WHOIS
        DNS reputation
        Google index
        PageRank
        Web traffic
        Webpage fetching
        Brand database
    """

    # -----------------------------------------------------
    # Extract real-time URL features
    # -----------------------------------------------------

    features = (
        extract_realtime_url_features(
            url
        )
    )

    # -----------------------------------------------------
    # Check for missing features
    # -----------------------------------------------------

    missing_features = [
        feature
        for feature in url_features
        if feature not in features
    ]

    if missing_features:

        raise ValueError(
            "Missing URL features: "
            + ", ".join(
                missing_features
            )
        )

    # -----------------------------------------------------
    # Build feature DataFrame
    #
    # IMPORTANT:
    # Use the feature order saved inside the model package.
    # -----------------------------------------------------

    feature_values = pd.DataFrame(
        [[
            features[feature]
            for feature in url_features
        ]],
        columns=url_features,
    )

    # -----------------------------------------------------
    # Predict phishing probability
    # -----------------------------------------------------

    probability = (
        url_model.predict_proba(
            feature_values
        )[0][1]
    )

    return float(
        probability
    )


# =========================================================
# URL FEATURE EXPLANATION
# =========================================================

def explain_realtime_url(
    url,
):
    """
    Generate human-readable indicators from
    locally calculated URL features.
    """

    features = (
        extract_realtime_url_features(
            url
        )
    )

    indicators = []

    # -----------------------------------------------------
    # IP address
    # -----------------------------------------------------

    if features.get("ip") == 1:

        indicators.append(
            "URL uses an IP address instead of a domain name."
        )

    # -----------------------------------------------------
    # Hyphens
    # -----------------------------------------------------

    if features.get(
        "nb_hyphens",
        0,
    ) >= 2:

        indicators.append(
            "URL contains multiple hyphens."
        )

    # -----------------------------------------------------
    # Subdomains
    # -----------------------------------------------------

    if features.get(
        "nb_subdomains",
        0,
    ) >= 2:

        indicators.append(
            "URL contains multiple subdomains."
        )

    # -----------------------------------------------------
    # Digit ratio
    # -----------------------------------------------------

    if features.get(
        "ratio_digits_url",
        0,
    ) >= 0.20:

        indicators.append(
            "URL contains a relatively high proportion of digits."
        )

    # -----------------------------------------------------
    # Host digit ratio
    # -----------------------------------------------------

    if features.get(
        "ratio_digits_host",
        0,
    ) >= 0.20:

        indicators.append(
            "Hostname contains a relatively high proportion of digits."
        )

    # -----------------------------------------------------
    # Phishing hints
    # -----------------------------------------------------

    if features.get(
        "phish_hints",
        0,
    ) > 0:

        indicators.append(
            "URL contains phishing-related keywords or hints."
        )

    # -----------------------------------------------------
    # @ symbol
    # -----------------------------------------------------

    if features.get(
        "nb_at",
        0,
    ) > 0:

        indicators.append(
            "URL contains an @ symbol."
        )

    # -----------------------------------------------------
    # Query marker
    # -----------------------------------------------------

    if features.get(
        "nb_qm",
        0,
    ) > 0:

        indicators.append(
            "URL contains query parameters."
        )

    # -----------------------------------------------------
    # Equal signs
    # -----------------------------------------------------

    if features.get(
        "nb_eq",
        0,
    ) > 0:

        indicators.append(
            "URL contains parameter assignment characters."
        )

    # -----------------------------------------------------
    # Underscores
    # -----------------------------------------------------

    if features.get(
        "nb_underscore",
        0,
    ) > 0:

        indicators.append(
            "URL contains underscore characters."
        )

    # -----------------------------------------------------
    # Character repetition
    # -----------------------------------------------------

    if features.get(
        "char_repeat",
        0,
    ) >= 5:

        indicators.append(
            "URL contains repeated character patterns."
        )

    # -----------------------------------------------------
    # Long hostname
    # -----------------------------------------------------

    if features.get(
        "length_hostname",
        0,
    ) >= 30:

        indicators.append(
            "Hostname is relatively long."
        )

    # -----------------------------------------------------
    # Many lexical components
    # -----------------------------------------------------

    if features.get(
        "length_words_raw",
        0,
    ) >= 10:

        indicators.append(
            "URL contains many lexical components."
        )

    # -----------------------------------------------------
    # Prefix/suffix
    # -----------------------------------------------------

    if features.get(
        "prefix_suffix",
        0,
    ) == 1:

        indicators.append(
            "Hostname contains a prefix/suffix pattern."
        )

    # -----------------------------------------------------
    # Path extension
    # -----------------------------------------------------

    if features.get(
        "path_extension",
        0,
    ) == 1:

        indicators.append(
            "URL contains a path extension."
        )

    return {
        "features":
            features,

        "indicators":
            indicators,
    }


# =========================================================
# COMPLETE THREAT ANALYSIS
# =========================================================

def analyze_threat(
    text=None,
    url=None,
):
    """
    Analyze:

        Text only
        URL only
        Text + URL

    Returns a unified threat-analysis dictionary.
    """

    # =====================================================
    # INPUT VALIDATION
    # =====================================================

    text_is_empty = (
        text is None
        or not str(text).strip()
    )

    url_is_empty = (
        url is None
        or not str(url).strip()
    )

    if (
        text_is_empty
        and url_is_empty
    ):

        raise ValueError(
            "Provide at least an email message or a URL."
        )

    # =====================================================
    # LOAD MODELS
    # =====================================================

    models = load_models()

    text_model = (
        models["text_model"]
    )

    text_vectorizer = (
        models["text_vectorizer"]
    )

    url_model = (
        models["url_model"]
    )

    url_features = (
        models["url_features"]
    )

    url_threshold = (
        models["url_threshold"]
    )

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

    url_features_result = None

    url_indicators = []

    if not url_is_empty:

        url_probability = predict_url(
            url,
            url_model,
            url_features,
        )

        url_analysis = (
            explain_realtime_url(
                url
            )
        )

        url_features_result = (
            url_analysis["features"]
        )

        url_indicators = (
            url_analysis["indicators"]
        )

    # =====================================================
    # RISK FUSION
    # =====================================================

    risk_score = (
        calculate_risk_score(
            text_probability,
            url_probability,
        )
    )

    # =====================================================
    # THREAT CLASSIFICATION
    # =====================================================

    threat_level = (
        classify_risk(
            risk_score
        )
    )

    # =====================================================
    # EXPLANATION
    # =====================================================

    explanation = (
        generate_explanation(
            text_probability,
            url_probability,
            risk_score,
            threat_level,
        )
    )

    # =====================================================
    # COMBINED INDICATORS
    # =====================================================

    indicators = []

    if text_probability is not None:

        if text_probability >= 0.70:

            indicators.append(
                "Text model detected a strong phishing signal."
            )

        elif text_probability >= 0.40:

            indicators.append(
                "Text model detected a suspicious signal."
            )

        else:

            indicators.append(
                "Text model detected a low phishing signal."
            )

    indicators.extend(
        url_indicators
    )

    # =====================================================
    # FINAL RESULT
    # =====================================================

    return {

        "text_probability":
            text_probability,

        "url_probability":
            url_probability,

        "risk_score":
            float(
                risk_score
            ),

        "threat_level":
            threat_level,

        "explanation":
            explanation,

        "indicators":
            indicators,

        "url_indicators":
            url_indicators,

        "url_features":
            url_features_result,

        "url_threshold":
            url_threshold,

        "models": {

            "text_model":
                TEXT_MODEL_FILE,

            "text_vectorizer":
                TEXT_VECTORIZER_FILE,

            "url_model":
                URL_MODEL_FILE,
        },
    }


# =========================================================
# COMMAND-LINE TEST
# =========================================================

if __name__ == "__main__":

    print(
        "=" * 70
    )

    print(
        "UNIFIED PHISHING THREAT PREDICTION ENGINE"
    )

    print(
        "=" * 70
    )

    result = analyze_threat(

        text=(
            "URGENT: Your account has been suspended. "
            "Verify your password immediately."
        ),

        url=(
            "http://secure-login.example.com/"
            "account/verify?id=12345"
        ),
    )

    print(
        "\nText probability:",
        f"{result['text_probability']:.4f}"
    )

    print(
        "URL probability:",
        f"{result['url_probability']:.4f}"
    )

    print(
        "URL threshold:",
        f"{result['url_threshold']:.2f}"
    )

    print(
        "Risk score:",
        f"{result['risk_score']:.4f}"
    )

    print(
        "Threat level:",
        result["threat_level"]
    )

    print(
        "\nExplanation:"
    )

    print(
        result["explanation"]
    )

    print(
        "\nIndicators:"
    )

    for indicator in result[
        "indicators"
    ]:

        print(
            "-",
            indicator
        )

    print(
        "\n" + "=" * 70
    )
