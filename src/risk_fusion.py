
def calculate_risk_score(
    text_probability=None,
    url_probability=None,
):
    """
    Calculate a combined threat score.

    Both signals:
        Text = 60%
        URL  = 40%

    Text only:
        Text = 100%

    URL only:
        URL = 100%
    """

    # -----------------------------------------------------
    # Validate that at least one signal exists
    # -----------------------------------------------------

    if (
        text_probability is None
        and url_probability is None
    ):
        raise ValueError(
            "At least one probability is required."
        )

    # -----------------------------------------------------
    # Validate text probability
    # -----------------------------------------------------

    if text_probability is not None:

        if not 0.0 <= text_probability <= 1.0:
            raise ValueError(
                "text_probability must be between 0 and 1."
            )

    # -----------------------------------------------------
    # Validate URL probability
    # -----------------------------------------------------

    if url_probability is not None:

        if not 0.0 <= url_probability <= 1.0:
            raise ValueError(
                "url_probability must be between 0 and 1."
            )

    # -----------------------------------------------------
    # CASE 1: Both signals available
    # -----------------------------------------------------

    if (
        text_probability is not None
        and url_probability is not None
    ):

        risk_score = (
            0.60 * text_probability
            + 0.40 * url_probability
        )

        return risk_score

    # -----------------------------------------------------
    # CASE 2: Text only
    # -----------------------------------------------------

    if text_probability is not None:

        return text_probability

    # -----------------------------------------------------
    # CASE 3: URL only
    # -----------------------------------------------------

    return url_probability

def classify_risk(risk_score):
    """
    Convert numerical risk score into
    a human-readable threat level.
    """

    if not 0.0 <= risk_score <= 1.0:
        raise ValueError(
            "risk_score must be between 0 and 1"
        )

    if risk_score >= 0.70:
        return "HIGH"

    elif risk_score >= 0.40:
        return "MEDIUM"

    else:
        return "LOW"


def generate_explanation(
    text_probability,
    url_probability,
    risk_score,
    threat_level,
):

    reasons = []

    # -----------------------------------------------------
    # Text explanation
    # -----------------------------------------------------

    if text_probability is not None:

        if text_probability >= 0.70:

            reasons.append(
                "The message content shows strong phishing indicators."
            )

        elif text_probability >= 0.40:

            reasons.append(
                "The message content shows some suspicious indicators."
            )

        else:

            reasons.append(
                "The message content does not show strong phishing indicators."
            )

    # -----------------------------------------------------
    # URL explanation
    # -----------------------------------------------------

    if url_probability is not None:

        if url_probability >= 0.70:

            reasons.append(
                "The URL shows strong phishing characteristics."
            )

        elif url_probability >= 0.40:

            reasons.append(
                "The URL shows some suspicious characteristics."
            )

        else:

            reasons.append(
                "The URL does not show strong phishing characteristics."
            )

    # -----------------------------------------------------
    # Final explanation
    # -----------------------------------------------------

    if not reasons:

        return "No analysis signals were available."

    return " ".join(reasons)


if __name__ == "__main__":

    # Test cases

    test_cases = [
        {
            "name": "Low Risk",
            "text": 0.10,
            "url": 0.15,
        },
        {
            "name": "Medium Risk",
            "text": 0.50,
            "url": 0.45,
        },
        {
            "name": "High Risk",
            "text": 0.90,
            "url": 0.85,
        },
    ]

    for case in test_cases:

        score = calculate_risk_score(
            case["text"],
            case["url"],
        )

        level = classify_risk(score)

        explanation = generate_explanation(
            case["text"],
            case["url"],
            score,
            level,
        )

        print("\n" + "=" * 60)
        print(case["name"])
        print("=" * 60)

        print(
            f"Text probability: {case['text']:.2f}"
        )

        print(
            f"URL probability:  {case['url']:.2f}"
        )

        print(
            f"Risk score:       {score:.2f}"
        )

        print(
            f"Threat level:     {level}"
        )

        print(
            f"Explanation:      {explanation}"
        )
