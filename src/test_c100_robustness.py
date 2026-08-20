import joblib

from sklearn.linear_model import LogisticRegression


# =========================================================
# FILES
# =========================================================

MODEL_FILE = "models/final_text_model.joblib"

VECTORIZER_FILE = (
    "models/final_text_vectorizer.joblib"
)


# =========================================================
# EXPERIMENTAL C VALUE
# =========================================================

EXPERIMENTAL_C = 100


# =========================================================
# LEGITIMATE TEST CASES
# Same messages used in the original robustness test.
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
    print("C=100 TEXT MODEL — ROBUSTNESS EXPERIMENT")
    print("=" * 70)

    print(
        "\nIMPORTANT:"
    )

    print(
        "The production model is NOT being modified."
    )

    # -----------------------------------------------------
    # Load existing TF-IDF vectorizer
    # -----------------------------------------------------

    vectorizer = joblib.load(
        VECTORIZER_FILE
    )

    # -----------------------------------------------------
    # Load the existing final model
    # -----------------------------------------------------
    #
    # We copy its configuration and only change C.
    # This is an experimental model.
    #

    base_model = joblib.load(
        MODEL_FILE
    )

    experimental_model = LogisticRegression(

        C=EXPERIMENTAL_C,

        max_iter=base_model.max_iter,

        class_weight=base_model.class_weight,

        solver=base_model.solver,

        random_state=42,
    )

    # -----------------------------------------------------
    # IMPORTANT
    #
    # We cannot simply change C on the existing fitted
    # model. The model must be trained again.
    #
    # For this experiment we therefore need the original
    # training data.
    # -----------------------------------------------------

    import pandas as pd

    from sklearn.model_selection import train_test_split

    df = pd.read_csv(
        "data/cleaned_text.csv"
    )

    df["processed_text"] = (
        df["processed_text"]
        .fillna("")
    )

    X = df["processed_text"]
    y = df["label"]

    X_train, _, y_train, _ = (
        train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42,
            stratify=y,
        )
    )

    print(
        "\nCreating training TF-IDF representation..."
    )

    X_train_tfidf = (
        vectorizer.transform(
            X_train
        )
    )

    print(
        "Training samples:",
        X_train_tfidf.shape
    )

    print(
        "\nTraining experimental model..."
    )

    experimental_model.fit(
        X_train_tfidf,
        y_train,
    )

    print(
        "Experimental model trained."
    )

    # -----------------------------------------------------
    # Test messages
    # -----------------------------------------------------

    false_positives = 0

    results = []

    for index, test_case in enumerate(
        TEST_CASES,
        start=1,
    ):

        vector = vectorizer.transform(
            [test_case["message"]]
        )

        probabilities = (
            experimental_model
            .predict_proba(vector)[0]
        )

        prediction = (
            experimental_model
            .predict(vector)[0]
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

        results.append(
            (
                test_case["category"],
                probabilities[0],
                probabilities[1],
                predicted_class,
                passed,
            )
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
            f"{probabilities[0]:.4f}",
        )

        print(
            "Phishing probability:",
            f"{probabilities[1]:.4f}",
        )

        print(
            "Prediction:",
            predicted_class,
        )

        print(
            "Result:",
            "[ PASS ]"
            if passed
            else "[ FALSE POSITIVE ]",
        )

    # -----------------------------------------------------
    # Summary
    # -----------------------------------------------------

    total = len(
        TEST_CASES
    )

    passed = total - false_positives

    accuracy = (
        passed / total
    )

    average_risk = (
        sum(
            r[2]
            for r in results
        )
        / total
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "C=100 ROBUSTNESS SUMMARY"
    )

    print(
        "=" * 70
    )

    print(
        f"\nTotal tests       : {total}"
    )

    print(
        f"Correct           : {passed}"
    )

    print(
        f"False positives   : {false_positives}"
    )

    print(
        f"Accuracy          : "
        f"{accuracy * 100:.2f}%"
    )

    print(
        f"Average phishing  : "
        f"{average_risk * 100:.2f}%"
    )

    print(
        "\nProduction model:"
    )

    print(
        "C=30 — unchanged"
    )

    print(
        "\nExperimental model:"
    )

    print(
        "C=100"
    )

    print(
        "\n" + "=" * 70
    )


if __name__ == "__main__":
    main()
