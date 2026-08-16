# =========================================================
# THREAT INTELLIGENCE REPORT GENERATOR
# =========================================================
#
# Converts the prediction engine output into a structured
# security-oriented threat intelligence report.
#
# This module does NOT perform machine learning.
# It works on the result produced by prediction_engine.py.
#
# =========================================================


# =========================================================
# THREAT SUMMARY
# =========================================================

def generate_threat_summary(
    threat_level,
    risk_score,
    text_probability=None,
    url_probability=None,
):
    """
    Generate a human-readable summary of the threat.

    Parameters:
        threat_level:
            HIGH / MEDIUM / LOW

        risk_score:
            Final fused risk score between 0 and 1.

        text_probability:
            Text model phishing probability.

        url_probability:
            URL model phishing probability.

    Returns:
        str:
            Human-readable threat summary.
    """

    risk_percentage = (
        float(risk_score) * 100
    )

    # -----------------------------------------------------
    # HIGH RISK
    # -----------------------------------------------------

    if threat_level == "HIGH":

        summary = (
            f"The analysis indicates a HIGH phishing risk "
            f"with an overall risk score of "
            f"{risk_percentage:.2f}%."
        )

        if (
            text_probability is not None
            and text_probability >= 0.70
        ):

            summary += (
                " The message content contains strong "
                "phishing indicators."
            )

        if (
            url_probability is not None
            and url_probability >= 0.70
        ):

            summary += (
                " The URL contains strong characteristics "
                "associated with phishing."
            )

        return summary


    # -----------------------------------------------------
    # MEDIUM RISK
    # -----------------------------------------------------

    if threat_level == "MEDIUM":

        summary = (
            f"The analysis indicates a MEDIUM phishing risk "
            f"with an overall risk score of "
            f"{risk_percentage:.2f}%."
        )

        if (
            text_probability is not None
            and text_probability >= 0.40
        ):

            summary += (
                " The message content shows suspicious "
                "characteristics."
            )

        if (
            url_probability is not None
            and url_probability >= 0.40
        ):

            summary += (
                " The URL shows suspicious characteristics."
            )

        return summary


    # -----------------------------------------------------
    # LOW RISK
    # -----------------------------------------------------

    return (
        f"The analysis indicates a LOW phishing risk "
        f"with an overall risk score of "
        f"{risk_percentage:.2f}%. "
        "No strong phishing indicators were detected."
    )


# =========================================================
# RECOMMENDED ACTIONS
# =========================================================

def generate_recommended_actions(
    threat_level,
    indicators=None,
):
    """
    Generate security recommendations based on
    the detected threat level and indicators.

    Returns:
        list:
            Recommended security actions.
    """

    if indicators is None:
        indicators = []


    # -----------------------------------------------------
    # HIGH RISK
    # -----------------------------------------------------

    if threat_level == "HIGH":

        actions = [
            "Do not click suspicious links.",
            "Do not enter passwords, OTPs, or other credentials.",
            "Do not download files from the suspicious message.",
            "Verify the sender through an official communication channel.",
            "If credentials were already entered, change the password immediately.",
        ]

        return actions


    # -----------------------------------------------------
    # MEDIUM RISK
    # -----------------------------------------------------

    if threat_level == "MEDIUM":

        actions = [
            "Exercise caution before interacting with the message or URL.",
            "Verify the sender and destination website independently.",
            "Do not provide sensitive information until the source is verified.",
        ]

        return actions


    # -----------------------------------------------------
    # LOW RISK
    # -----------------------------------------------------

    return [
        "No immediate phishing action is indicated.",
        "Continue following normal security practices.",
        "Verify unexpected requests before sharing sensitive information.",
    ]


# =========================================================
# DETECTION SIGNALS
# =========================================================

def generate_detection_signals(
    text_probability=None,
    url_probability=None,
):
    """
    Convert model probabilities into human-readable
    detection signals.

    Returns:
        list of dictionaries.
    """

    signals = []


    # -----------------------------------------------------
    # TEXT MODEL
    # -----------------------------------------------------

    if text_probability is not None:

        text_percentage = (
            float(text_probability) * 100
        )

        if text_probability >= 0.70:

            signals.append(
                {
                    "source": "Text Model",
                    "signal": "Strong phishing signal",
                    "probability": (
                        f"{text_percentage:.2f}%"
                    ),
                }
            )

        elif text_probability >= 0.40:

            signals.append(
                {
                    "source": "Text Model",
                    "signal": "Suspicious signal",
                    "probability": (
                        f"{text_percentage:.2f}%"
                    ),
                }
            )

        else:

            signals.append(
                {
                    "source": "Text Model",
                    "signal": "Low phishing signal",
                    "probability": (
                        f"{text_percentage:.2f}%"
                    ),
                }
            )


    # -----------------------------------------------------
    # URL MODEL
    # -----------------------------------------------------

    if url_probability is not None:

        url_percentage = (
            float(url_probability) * 100
        )

        if url_probability >= 0.70:

            signals.append(
                {
                    "source": "URL Model",
                    "signal": "Strong phishing signal",
                    "probability": (
                        f"{url_percentage:.2f}%"
                    ),
                }
            )

        elif url_probability >= 0.40:

            signals.append(
                {
                    "source": "URL Model",
                    "signal": "Suspicious signal",
                    "probability": (
                        f"{url_percentage:.2f}%"
                    ),
                }
            )

        else:

            signals.append(
                {
                    "source": "URL Model",
                    "signal": "Low phishing signal",
                    "probability": (
                        f"{url_percentage:.2f}%"
                    ),
                }
            )


    return signals


# =========================================================
# COMPLETE THREAT REPORT
# =========================================================

def generate_threat_report(
    prediction_result,
):
    """
    Generate a complete structured threat intelligence
    report from prediction_engine.analyze_threat() output.

    Parameters:
        prediction_result:
            Dictionary returned by analyze_threat().

    Returns:
        Dictionary containing the complete threat report.
    """

    if not isinstance(
        prediction_result,
        dict,
    ):

        raise TypeError(
            "prediction_result must be a dictionary."
        )


    # -----------------------------------------------------
    # Extract prediction values
    # -----------------------------------------------------

    text_probability = prediction_result.get(
        "text_probability"
    )

    url_probability = prediction_result.get(
        "url_probability"
    )

    risk_score = prediction_result.get(
        "risk_score"
    )

    threat_level = prediction_result.get(
        "threat_level"
    )

    indicators = prediction_result.get(
        "indicators",
        [],
    )

    url_features = prediction_result.get(
        "url_features"
    )


    # -----------------------------------------------------
    # Validate required values
    # -----------------------------------------------------

    if risk_score is None:

        raise ValueError(
            "Prediction result does not contain risk_score."
        )

    if threat_level is None:

        raise ValueError(
            "Prediction result does not contain threat_level."
        )


    # -----------------------------------------------------
    # Generate report components
    # -----------------------------------------------------

    summary = generate_threat_summary(
        threat_level=threat_level,
        risk_score=risk_score,
        text_probability=text_probability,
        url_probability=url_probability,
    )


    detection_signals = generate_detection_signals(
        text_probability=text_probability,
        url_probability=url_probability,
    )


    recommended_actions = (
        generate_recommended_actions(
            threat_level=threat_level,
            indicators=indicators,
        )
    )


    # -----------------------------------------------------
    # Final report
    # -----------------------------------------------------

    report = {

        "threat_level": threat_level,

        "risk_score": float(
            risk_score
        ),

        "risk_percentage": (
            float(risk_score) * 100
        ),

        "text_probability":
            text_probability,

        "url_probability":
            url_probability,

        "detection_signals":
            detection_signals,

        "url_indicators":
            indicators,

        "url_features":
            url_features,

        "summary":
            summary,

        "recommended_actions":
            recommended_actions,
    }


    return report


# =========================================================
# DISPLAY REPORT IN TERMINAL
# =========================================================

def print_threat_report(
    report,
):
    """
    Print a formatted threat intelligence report.
    """

    print(
        "\n" + "=" * 70
    )

    print(
        "PHISHING THREAT INTELLIGENCE REPORT"
    )

    print(
        "=" * 70
    )


    # -----------------------------------------------------
    # Threat level
    # -----------------------------------------------------

    print(
        f"\nThreat Level: "
        f"{report['threat_level']}"
    )

    print(
        f"Risk Score: "
        f"{report['risk_percentage']:.2f}%"
    )


    # -----------------------------------------------------
    # Detection signals
    # -----------------------------------------------------

    print(
        "\nDetection Signals"
    )

    print(
        "-" * 30
    )


    for signal in report[
        "detection_signals"
    ]:

        print(
            f"{signal['source']}: "
            f"{signal['signal']} "
            f"({signal['probability']})"
        )


    # -----------------------------------------------------
    # URL indicators
    # -----------------------------------------------------

    if report[
        "url_indicators"
    ]:

        print(
            "\nURL Indicators"
        )

        print(
            "-" * 30
        )


        for indicator in report[
            "url_indicators"
        ]:

            print(
                f"- {indicator}"
            )


    # -----------------------------------------------------
    # Summary
    # -----------------------------------------------------

    print(
        "\nAssessment"
    )

    print(
        "-" * 30
    )

    print(
        report["summary"]
    )


    # -----------------------------------------------------
    # Recommended actions
    # -----------------------------------------------------

    print(
        "\nRecommended Actions"
    )

    print(
        "-" * 30
    )


    for action in report[
        "recommended_actions"
    ]:

        print(
            f"- {action}"
        )


# =========================================================
# TEST
# =========================================================

if __name__ == "__main__":

    # -----------------------------------------------------
    # Example prediction result
    #
    # This simulates the output of prediction_engine.py.
    # -----------------------------------------------------

    example_prediction = {

        "text_probability":
            0.9271,

        "url_probability":
            0.9400,

        "risk_score":
            0.9343,

        "threat_level":
            "HIGH",

        "explanation":
            "The message content shows strong phishing indicators.",

        "indicators": [
            "URL is relatively long",
            "URL uses an IP address instead of a domain",
            "URL does not use HTTPS",
            "URL contains many special characters",
            "URL contains a high proportion of digits",
        ],

        "url_features": {
            "url_length": 50,
            "dot_count": 3,
            "has_https": 0,
            "has_ip_address": 1,
            "special_chars": 4,
            "subdomain_count": 0,
            "path_depth": 1,
            "digit_ratio": 0.38,
            "has_at_symbol": 0,
            "domain_length": 12,
        },
    }


    # -----------------------------------------------------
    # Generate report
    # -----------------------------------------------------

    report = generate_threat_report(
        example_prediction
    )


    # -----------------------------------------------------
    # Print report
    # -----------------------------------------------------

    print_threat_report(
        report
    )
