import joblib


MODEL_FILE = "models/final_text_model.joblib"
VECTORIZER_FILE = "models/final_text_vectorizer.joblib"


TEST_MESSAGES = [

    {
        "name": "Academic Notice",
        "text": """
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
        "name": "HR Administrative Message",
        "text": """
        Dear employees,

        The annual leave submission window will remain
        open until the end of this month.

        Please submit your planned leave dates through
        the internal employee portal.

        Contact the HR department for assistance.
        """,
    },

]


def analyze_message(
    name,
    text,
    model,
    vectorizer,
):

    vector = vectorizer.transform(
        [text]
    )

    probability = model.predict_proba(
        vector
    )[0]

    decision_score = model.decision_function(
        vector
    )[0]

    names = (
        vectorizer
        .get_feature_names_out()
    )

    weights = model.coef_[0]

    contributions = []

    for index in vector.nonzero()[1]:

        tfidf_value = float(
            vector[0, index]
        )

        weight = float(
            weights[index]
        )

        contribution = (
            tfidf_value * weight
        )

        contributions.append(
            (
                names[index],
                tfidf_value,
                weight,
                contribution,
            )
        )


    contributions.sort(
        key=lambda x: x[3],
        reverse=True,
    )


    print(
        "\n" + "=" * 70
    )

    print(
        f"FALSE POSITIVE ANALYSIS: {name}"
    )

    print(
        "=" * 70
    )

    print(
        f"\nLegitimate probability: "
        f"{probability[0]:.4f}"
    )

    print(
        f"Phishing probability:   "
        f"{probability[1]:.4f}"
    )

    print(
        f"Decision score:          "
        f"{decision_score:.4f}"
    )


    print(
        "\nTop phishing-driving features:"
    )

    print(
        "-" * 70
    )

    for (
        feature,
        tfidf,
        weight,
        contribution,
    ) in contributions[:10]:

        print(
            f"{feature:30} "
            f"TFIDF={tfidf:.4f} "
            f"Weight={weight:+.4f} "
            f"Contribution={contribution:+.4f}"
        )


    print(
        "\nTop legitimate-driving features:"
    )

    print(
        "-" * 70
    )

    for (
        feature,
        tfidf,
        weight,
        contribution,
    ) in contributions[-10:][::-1]:

        print(
            f"{feature:30} "
            f"TFIDF={tfidf:.4f} "
            f"Weight={weight:+.4f} "
            f"Contribution={contribution:+.4f}"
        )


def main():

    print(
        "Loading final text model..."
    )

    model = joblib.load(
        MODEL_FILE
    )

    vectorizer = joblib.load(
        VECTORIZER_FILE
    )

    print(
        "Model loaded successfully."
    )

    for test_case in TEST_MESSAGES:

        analyze_message(
            test_case["name"],
            test_case["text"],
            model,
            vectorizer,
        )


if __name__ == "__main__":

    main()
