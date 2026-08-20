import json
import joblib


# =========================================================
# MODEL FILES
# =========================================================

MODEL_FILE = "models/final_text_model.joblib"

VECTORIZER_FILE = (
    "models/final_text_vectorizer.joblib"
)

REPORT_FILE = (
    "reports/legitimate_robustness_results.json"
)


# =========================================================
# LEGITIMATE TEST CASES
# =========================================================

TEST_CASES = [

    {
        "category": "Academic Notice",
        "message": """
        Dear students,

        The Department of Computer Science will conduct
        the project review on Monday at 10:00 AM.

        Please bring your project documentation and
        presentation slides.

        Regards,
        Department Coordinator
        """,
    },

    {
        "category": "Business Meeting",
        "message": """
        Hello team,

        The monthly project meeting is scheduled for
        Friday at 3 PM in the conference room.

        Please review the progress report and prepare
        any updates that need to be discussed.

        Thank you.
        """,
    },

    {
        "category": "College Event",
        "message": """
        Dear students,

        The annual technical symposium will be held
        next week in the main auditorium.

        Students interested in participating can register
        with the event coordinator before Friday.

        Regards,
        Student Activities Committee
        """,
    },

    {
        "category": "Project Announcement",
        "message": """
        The software engineering project review will
        take place next Wednesday.

        Each team should prepare a short presentation
        explaining the implementation, testing process,
        and results.

        Please contact the faculty coordinator if you
        have any questions.
        """,
    },

    {
        "category": "HR Administrative Message",
        "message": """
        Dear employees,

        The annual leave submission window will remain
        open until the end of this month.

        Please submit your planned leave dates through
        the internal employee portal.

        Contact the HR department for assistance.
        """,
    },

    {
        "category": "Technical Notification",
        "message": """
        System maintenance is scheduled for Saturday
        from 10 PM to midnight.

        During this period, some internal services may
        be temporarily unavailable.

        No action is required from users.

        Thank you for your understanding.
        """,
    },

    {
        "category": "Newsletter",
        "message": """
        Hello everyone,

        This week's newsletter includes updates about
        upcoming workshops, department activities,
        student achievements, and new learning resources.

        We hope you find the information useful.

        Regards,
        Communications Team
        """,
    },

    {
        "category": "General Communication",
        "message": """
        Hi everyone,

        Please remember that the office will remain
        closed on Monday due to the scheduled holiday.

        Normal working hours will resume on Tuesday.

        Thank you.
        """,
    },

]


# =========================================================
# LOAD MODEL
# =========================================================

def load_model():

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

    vector = vectorizer.transform(
        [message]
    )

    probabilities = (
        model.predict_proba(vector)[0]
    )

    prediction = model.predict(
        vector
    )[0]

    return {
        "prediction": int(prediction),
        "legitimate_probability":
            float(probabilities[0]),
        "phishing_probability":
            float(probabilities[1]),
    }


# =========================================================
# MAIN
# =========================================================

def main():

    print("=" * 70)
    print("LEGITIMATE MESSAGE ROBUSTNESS TEST")
    print("=" * 70)

    print(
        "\nPurpose:"
    )

    print(
        "Evaluate whether the final text model "
        "incorrectly flags realistic legitimate messages."
    )


    # =====================================================
    # LOAD MODEL
    # =====================================================

    print(
        "\nLoading final model..."
    )

    model, vectorizer = load_model()

    print(
        "Model loaded successfully."
    )

    print(
        f"Model: {type(model).__name__}"
    )

    print(
        f"TF-IDF features: "
        f"{len(vectorizer.get_feature_names_out())}"
    )


    # =====================================================
    # TEST RESULTS
    # =====================================================

    results = []

    false_positives = 0

    phishing_probabilities = []


    # =====================================================
    # RUN TEST CASES
    # =====================================================

    for index, test_case in enumerate(
        TEST_CASES,
        start=1,
    ):

        result = predict_message(
            test_case["message"],
            model,
            vectorizer,
        )

        predicted_class = (
            "PHISHING"
            if result["prediction"] == 1
            else "LEGITIMATE"
        )

        passed = (
            result["prediction"] == 0
        )

        if not passed:

            false_positives += 1

        phishing_probability = (
            result["phishing_probability"]
        )

        phishing_probabilities.append(
            phishing_probability
        )


        print(
            "\n" + "-" * 70
        )

        print(
            f"TEST {index}: "
            f"{test_case['category']}"
        )

        print(
            "-" * 70
        )

        print(
            "Predicted class:",
            predicted_class,
        )

        print(
            "Legitimate probability:",
            f"{result['legitimate_probability']:.4f}",
        )

        print(
            "Phishing probability:",
            f"{result['phishing_probability']:.4f}",
        )

        print(
            "Result:",
            "[ PASS ]"
            if passed
            else "[ FALSE POSITIVE ]",
        )


        results.append({

            "category":
                test_case["category"],

            "predicted_class":
                predicted_class,

            "legitimate_probability":
                result[
                    "legitimate_probability"
                ],

            "phishing_probability":
                result[
                    "phishing_probability"
                ],

            "passed":
                passed,

        })


    # =====================================================
    # STATISTICS
    # =====================================================

    total_tests = len(
        TEST_CASES
    )

    passed_tests = (
        total_tests
        - false_positives
    )

    accuracy = (
        passed_tests
        / total_tests
    )

    average_phishing_probability = (
        sum(phishing_probabilities)
        / len(phishing_probabilities)
    )

    highest_risk_index = max(
        range(len(results)),
        key=lambda i:
            results[i][
                "phishing_probability"
            ],
    )

    highest_risk_case = (
        results[highest_risk_index]
    )


    # =====================================================
    # SUMMARY
    # =====================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "ROBUSTNESS TEST SUMMARY"
    )

    print(
        "=" * 70
    )

    print(
        f"\nTotal legitimate tests : "
        f"{total_tests}"
    )

    print(
        f"Correctly classified  : "
        f"{passed_tests}"
    )

    print(
        f"False positives       : "
        f"{false_positives}"
    )

    print(
        f"Legitimate accuracy   : "
        f"{accuracy * 100:.2f}%"
    )

    print(
        f"Average phishing risk : "
        f"{average_phishing_probability * 100:.2f}%"
    )

    print(
        "\nHighest-risk legitimate message:"
    )

    print(
        highest_risk_case["category"]
    )

    print(
        "Phishing probability:",
        f"{highest_risk_case['phishing_probability'] * 100:.2f}%",
    )


    # =====================================================
    # SAVE REPORT
    # =====================================================

    report = {

        "total_tests":
            total_tests,

        "correct_predictions":
            passed_tests,

        "false_positives":
            false_positives,

        "legitimate_accuracy":
            accuracy,

        "average_phishing_probability":
            average_phishing_probability,

        "highest_risk_case":
            highest_risk_case,

        "results":
            results,

    }


    # Create reports directory if necessary

    import os

    os.makedirs(
        "reports",
        exist_ok=True,
    )


    with open(
        REPORT_FILE,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            report,
            file,
            indent=4,
        )


    print(
        "\nDetailed report saved to:"
    )

    print(
        REPORT_FILE
    )


    # =====================================================
    # FINAL STATUS
    # =====================================================

    print(
        "\n" + "=" * 70
    )

    if false_positives == 0:

        print(
            "STATUS: ALL LEGITIMATE TESTS PASSED"
        )

    else:

        print(
            "STATUS: FALSE POSITIVES DETECTED"
        )

        print(
            "Further investigation is recommended."
        )

    print(
        "=" * 70
    )


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":

    main()
