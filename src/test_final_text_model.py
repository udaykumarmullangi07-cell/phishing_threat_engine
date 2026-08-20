import joblib


# =========================================================
# MODEL FILES
# =========================================================

MODEL_FILE = (
    "models/final_text_model.joblib"
)

VECTORIZER_FILE = (
    "models/final_text_vectorizer.joblib"
)


# =========================================================
# LOAD FINAL MODEL
# =========================================================

def load_final_model():

    model = joblib.load(
        MODEL_FILE
    )

    vectorizer = joblib.load(
        VECTORIZER_FILE
    )

    return model, vectorizer


# =========================================================
# PREDICT MESSAGE
# =========================================================

def predict_message(
    message,
    model,
    vectorizer,
):
    """
    Predict whether a message is legitimate
    or phishing.

    Returns:
        dictionary containing prediction details.
    """

    # Convert message into TF-IDF representation

    text_vector = vectorizer.transform(
        [message]
    )

    # Prediction

    prediction = model.predict(
        text_vector
    )[0]

    # Probability

    probabilities = model.predict_proba(
        text_vector
    )[0]

    legitimate_probability = probabilities[0]

    phishing_probability = probabilities[1]

    # Convert model output to human-readable class

    if prediction == 1:

        predicted_class = "PHISHING"

        confidence = phishing_probability

    else:

        predicted_class = "LEGITIMATE"

        confidence = legitimate_probability


    return {

        "prediction":
            int(prediction),

        "predicted_class":
            predicted_class,

        "legitimate_probability":
            legitimate_probability,

        "phishing_probability":
            phishing_probability,

        "confidence":
            confidence,

    }


# =========================================================
# TEST CASES
# =========================================================

TEST_CASES = [

    {
        "name":
            "Obvious Account Phishing",

        "expected":
            1,

        "message":
            """
            URGENT SECURITY ALERT

            Your account has been suspended because
            of unusual login activity.

            You must verify your account immediately.

            Click the verification link and confirm
            your username and password within 24 hours.

            Failure to verify your account will result
            in permanent account closure.
            """,
    },


    {
        "name":
            "Credential Theft Attempt",

        "expected":
            1,

        "message":
            """
            IMPORTANT: Your banking account requires
            immediate verification.

            We detected suspicious activity on your
            account.

            Please confirm your login credentials,
            password and OTP immediately to prevent
            your account from being blocked.

            Failure to complete verification today
            will result in account suspension.
            """,
    },


    {
        "name":
            "Urgent Payment Phishing",

        "expected":
            1,

        "message":
            """
            Your payment account has been restricted.

            Immediate action is required to restore
            access.

            Please verify your account information
            and payment details using the secure
            verification page.

            If you do not complete verification today,
            your account will be permanently disabled.
            """,
    },


    {
        "name":
            "Normal Academic Message",

        "expected":
            0,

        "message":
            """
            Dear students,

            The Department of Computer Science will
            conduct the project review on Monday at
            10:00 AM.

            Please bring your project documentation
            and presentation slides.

            Regards,
            Department Coordinator
            """,
    },


    {
        "name":
            "Normal Business Message",

        "expected":
            0,

        "message":
            """
            Hello,

            The monthly team meeting is scheduled for
            Friday at 3 PM.

            Please review the project progress report
            before the meeting and prepare any updates
            that need to be discussed.

            Thank you.
            """,
    },

]


# =========================================================
# DISPLAY RESULT
# =========================================================

def display_result(
    test_number,
    test_case,
    result,
):

    expected_class = (
        "PHISHING"
        if test_case["expected"] == 1
        else "LEGITIMATE"
    )

    actual_class = (
        result["predicted_class"]
    )

    passed = (
        test_case["expected"]
        == result["prediction"]
    )

    status = (
        "PASS"
        if passed
        else "FAIL"
    )


    print(
        "\n" + "-" * 70
    )

    print(
        f"TEST {test_number}: "
        f"{test_case['name']}"
    )

    print(
        "-" * 70
    )


    print(
        "\nMessage:"
    )

    print(
        test_case["message"].strip()
    )


    print(
        "\nExpected class:"
    )

    print(
        expected_class
    )


    print(
        "\nPredicted class:"
    )

    print(
        actual_class
    )


    print(
        "\nLegitimate probability:"
    )

    print(
        f"{result['legitimate_probability']:.4f}"
    )


    print(
        "Phishing probability:"
    )

    print(
        f"{result['phishing_probability']:.4f}"
    )


    print(
        "Confidence:"
    )

    print(
        f"{result['confidence']:.4f}"
    )


    print(
        "\nResult:"
    )

    print(
        f"[ {status} ]"
    )


    return passed


# =========================================================
# MAIN TEST FUNCTION
# =========================================================

def main():

    print("=" * 70)

    print(
        "FINAL TEXT MODEL — PREDICTION TEST SUITE"
    )

    print("=" * 70)


    # =====================================================
    # LOAD MODEL
    # =====================================================

    print(
        "\nLoading final model..."
    )

    model, vectorizer = (
        load_final_model()
    )


    print(
        "Model loaded successfully."
    )

    print(
        f"Model type: "
        f"{type(model).__name__}"
    )

    print(
        f"TF-IDF features: "
        f"{len(vectorizer.get_feature_names_out())}"
    )


    # =====================================================
    # RUN TESTS
    # =====================================================

    passed_tests = 0

    failed_tests = 0


    for number, test_case in enumerate(
        TEST_CASES,
        start=1,
    ):

        result = predict_message(
            test_case["message"],
            model,
            vectorizer,
        )


        passed = display_result(
            number,
            test_case,
            result,
        )


        if passed:

            passed_tests += 1

        else:

            failed_tests += 1


    # =====================================================
    # FINAL SUMMARY
    # =====================================================

    total_tests = (
        passed_tests
        + failed_tests
    )


    accuracy = (
        passed_tests
        / total_tests
    )


    print(
        "\n" + "=" * 70
    )

    print(
        "FINAL TEST SUMMARY"
    )

    print(
        "=" * 70
    )


    print(
        f"\nTotal tests : {total_tests}"
    )

    print(
        f"Passed      : {passed_tests}"
    )

    print(
        f"Failed      : {failed_tests}"
    )

    print(
        f"Test result : "
        f"{accuracy * 100:.2f}%"
    )


    if failed_tests == 0:

        print(
            "\nSTATUS: ALL TESTS PASSED"
        )

    else:

        print(
            "\nSTATUS: SOME TESTS FAILED"
        )


    print(
        "\n" + "=" * 70
    )


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":

    main()
