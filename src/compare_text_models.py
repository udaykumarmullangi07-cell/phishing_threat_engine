import pandas as pd


results = [
    {
        "Model": "Logistic Regression",
        "Accuracy": 0.9870,
        "Precision": 0.9867,
        "Recall": 0.9882,
        "F1": 0.9875,
        "ROC-AUC": 0.9988,
    },
    {
        "Model": "Random Forest",
        "Accuracy": 0.9455,
        "Precision": 0.9267,
        "Recall": 0.9721,
        "F1": 0.9489,
        "ROC-AUC": 0.9897,
    },
    {
        "Model": "XGBoost",
        "Accuracy": 0.9706,
        "Precision": 0.9599,
        "Recall": 0.9846,
        "F1": 0.9721,
        "ROC-AUC": 0.9961,
    },
]


def main():

    df = pd.DataFrame(results)

    print("\n" + "=" * 70)
    print("TEXT MODEL COMPARISON")
    print("=" * 70)

    print(
        df.to_string(
            index=False,
            formatters={
                "Accuracy": "{:.4f}".format,
                "Precision": "{:.4f}".format,
                "Recall": "{:.4f}".format,
                "F1": "{:.4f}".format,
                "ROC-AUC": "{:.4f}".format,
            },
        )
    )

    # Rank models using F1 score
    ranked = df.sort_values(
        by="F1",
        ascending=False,
    )

    print("\n" + "=" * 70)
    print("RANKING BY F1 SCORE")
    print("=" * 70)

    print(
        ranked[
            [
                "Model",
                "Accuracy",
                "Precision",
                "Recall",
                "F1",
                "ROC-AUC",
            ]
        ].to_string(index=False)
    )

    best_model = ranked.iloc[0]["Model"]

    print("\nBest text model based on F1:", best_model)


if __name__ == "__main__":
    main()
