from src.prediction_engine import (
    analyze_threat,
)


# =========================================================
# TEST CASES
# =========================================================

test_cases = [

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

    {
        "name":
            "Empty Input",

        "text":
            None,

        "url":
            None,
    },
]


# =========================================================
# DISPLAY RESULT
# =========================================================

def print_result(result):

    text_probability = (
        result["text_probability"]
    )

    url_probability = (
        result["url_probability"]
    )

    if text_probability is None:
        text_display = "None"
    else:
        text_display = (
            f"{text_probability:.4f}"
        )

    if url_probability is None:
        url_display = "None"
    else:
        url_display = (
            f"{url_probability:.4f}"
        )

    print(
        "\nText phishing probability:",
        text_display,
    )

    print(
        "URL phishing probability: ",
        url_display,
    )

    print(
        "Combined risk score:       ",
        f"{result['risk_score']:.4f}",
    )

    print(
        "Threat level:              ",
        result["threat_level"],
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

    if result["indicators"]:

        for indicator in result[
            "indicators"
        ]:

            print(
                f"- {indicator}"
            )

    else:

        print(
            "No suspicious URL indicators."
        )


# =========================================================
# RUN TESTS
# =========================================================

if __name__ == "__main__":

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
