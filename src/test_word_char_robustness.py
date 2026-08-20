import joblib
from scipy.sparse import hstack


# =========================================================
# EXPERIMENTAL MODEL FILES
# =========================================================

MODEL_FILE = (
    "models/experimental_word_char_model.joblib"
)

WORD_VECTORIZER_FILE = (
    "models/experimental_word_vectorizer.joblib"
)

CHAR_VECTORIZER_FILE = (
    "models/experimental_char_vectorizer.joblib"
)


# =========================================================
# LEGITIMATE TEST CASES
# EXACT SAME ROBUSTNESS CATEGORIES
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
# MAIN
# =========================================================

def main():

    print("=" * 70)
    print(
        "WORD + CHARACTER MODEL — LEGITIMATE ROBUSTNESS TEST"
    )
    print("=" * 70)

    print(
        "\nPurpose:"
    )

    print(
        "Test whether the experimental model reduces"
        " false positives on realistic legitimate messages."
    )

    print(
        "\nProduction model will NOT be modified."
    )

    # -----------------------------------------------------
    # Load model
    # -----------------------------------------------------

    print(
        "\nLoading experimental model..."
    )

    model = joblib.load(
        MODEL_FILE
    )

    word_vectorizer = joblib.load(
        WORD_VECTORIZER_FILE
    )

    char_vectorizer = joblib.load(
        CHAR_VECTORIZER_FILE
    )

    print(
        "Model loaded successfully."
    )

    print(
        "Model:",
        type(model).__name__
    )

    print(
        "Word features:",
        len(
            word_vectorizer
            .get_feature_names_out()
        )
    )

    print(
        "Character features:",
        len(
            char_vectorizer
            .get_feature_names_out()
        )
    )

    # -----------------------------------------------------
    # Test
    # -----------------------------------------------------

    false_positives = 0

    total_risk = 0.0

    results = []

    for index, test_case in enumerate(
        TEST_CASES,
        start=1,
    ):

        text = test_case["message"]

        # Word representation
        word_vector = (
            word_vectorizer.transform(
                [text]
            )
        )

        # Character representation
        char_vector = (
            char_vectorizer.transform(
                [text]
            )
        )

        # Combine
        combined_vector = hstack(
            [
                word_vector,
                char_vector,
            ]
        ).tocsr()

        # Prediction
        probabilities = (
            model.predict_proba(
                combined_vector
            )[0]
        )

        prediction = model.predict(
            combined_vector
        )[0]

        legitimate_probability = (
            probabilities[0]
        )

        phishing_probability = (
            probabilities[1]
        )

        predicted_class = (
            "PHISHING"
            if prediction == 1
            else "LEGITIMATE"
        )

        passed = (
            prediction == 0
        )

        if not passed:
            false_positives += 1

        total_risk += phishing_probability

        results.append(
            {
                "category":
                    test_case["category"],

                "legitimate_probability":
                    legitimate_probability,

                "phishing_probability":
                    phishing_probability,

                "prediction":
                    predicted_class,

                "passed":
                    passed,
            }
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
            "Legitimate probability:",
            f"{legitimate_probability:.4f}"
        )

        print(
            "Phishing probability:  ",
            f"{phishing_probability:.4f}"
        )

        print(
            "Prediction:",
            predicted_class
        )

        print(
            "Result:",
            "[ PASS ]"
            if passed
            else "[ FALSE POSITIVE ]"
        )

    # -----------------------------------------------------
    # Summary
    # -----------------------------------------------------

    total_tests = len(
        TEST_CASES
    )

    correct = (
        total_tests - false_positives
    )

    robustness_accuracy = (
        correct / total_tests
    )

    average_risk = (
        total_risk / total_tests
    )

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
        f"{correct}"
    )

    print(
        f"False positives       : "
        f"{false_positives}"
    )

    print(
        f"Legitimate accuracy   : "
        f"{robustness_accuracy * 100:.2f}%"
    )

    print(
        f"Average phishing risk : "
        f"{average_risk * 100:.2f}%"
    )

    # -----------------------------------------------------
    # Highest-risk legitimate message
    # -----------------------------------------------------

    highest_risk = max(
        results,
        key=lambda item:
            item["phishing_probability"]
    )

    print(
        "\nHighest-risk legitimate message:"
    )

    print(
        highest_risk["category"]
    )

    print(
        "Phishing probability:",
        f"{highest_risk['phishing_probability'] * 100:.2f}%"
    )

    # -----------------------------------------------------
    # Final status
    # -----------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    if false_positives == 0:

        print(
            "STATUS: EXCELLENT — "
            "NO FALSE POSITIVES"
        )

    elif false_positives < 2:

        print(
            "STATUS: IMPROVED"
        )

    else:

        print(
            "STATUS: FALSE POSITIVES REMAIN"
        )

    print(
        "=" * 70
    )


if __name__ == "__main__":

    main()
