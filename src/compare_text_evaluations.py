import json
import os

import pandas as pd


REPORT_DIR = "reports"


GROUP_AWARE_FILE = os.path.join(
    REPORT_DIR,
    "text_group_aware_evaluation.json",
)

TUNING_FILE = os.path.join(
    "models",
    "text_tuning_results.csv",
)

WORD_CHAR_FILE = os.path.join(
    REPORT_DIR,
    "text_word_char_experiment.csv",
)

CHAR_ONLY_FILE = os.path.join(
    REPORT_DIR,
    "text_character_only_experiment.csv",
)

REGULARIZATION_FILE = os.path.join(
    REPORT_DIR,
    "text_regularization_experiment.csv",
)

THRESHOLD_FILE = os.path.join(
    REPORT_DIR,
    "text_threshold_analysis.csv",
)


def load_json(path):

    if not os.path.exists(path):
        return None

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:

        return json.load(f)


def load_csv(path):

    if not os.path.exists(path):
        return None

    return pd.read_csv(path)


def main():

    print("=" * 70)
    print("TEXT MODEL — EXPERIMENT COMPARISON")
    print("=" * 70)

    print(
        """
Purpose:
Create a consolidated record of the text-model experiments.

No production model will be modified.
"""
    )

    # --------------------------------------------------------
    # GROUP-AWARE RESULT
    # --------------------------------------------------------

    group_result = load_json(
        GROUP_AWARE_FILE
    )

    if group_result is None:

        print(
            "\nGroup-aware report not found:"
        )

        print(
            GROUP_AWARE_FILE
        )

        return

    group_metrics = (
        group_result["metrics"]
    )

    # --------------------------------------------------------
    # CURRENT PRODUCTION BASELINE
    # --------------------------------------------------------

    results = []

    results.append({

        "experiment":
            "Production random split",

        "representation":
            "Word TF-IDF",

        "C":
            30,

        "accuracy":
            0.9911,

        "precision":
            0.9911,

        "recall":
            0.9917,

        "f1":
            0.9914,

        "roc_auc":
            0.9994,

        "false_positives":
            76,

        "false_negatives":
            71,

        "evaluation":
            "Random split",
    })

    # --------------------------------------------------------
    # GROUP-AWARE
    # --------------------------------------------------------

    results.append({

        "experiment":
            "Group-aware evaluation",

        "representation":
            "Word TF-IDF",

        "C":
            30,

        "accuracy":
            group_metrics["accuracy"],

        "precision":
            group_metrics["precision"],

        "recall":
            group_metrics["recall"],

        "f1":
            group_metrics["f1"],

        "roc_auc":
            group_metrics["roc_auc"],

        "false_positives":
            group_metrics["false_positives"],

        "false_negatives":
            group_metrics["false_negatives"],

        "evaluation":
            "Group-aware split",
    })

    # --------------------------------------------------------
    # REGULARIZATION EXPERIMENT
    # --------------------------------------------------------

    reg = load_csv(
        REGULARIZATION_FILE
    )

    if reg is not None:

        for _, row in reg.iterrows():

            results.append({

                "experiment":
                    f"Regularization C={row['C']}",

                "representation":
                    "Word TF-IDF",

                "C":
                    row["C"],

                "accuracy":
                    row["accuracy"],

                "precision":
                    row["precision"],

                "recall":
                    row["recall"],

                "f1":
                    row["f1"],

                "roc_auc":
                    row["roc_auc"],

                "false_positives":
                    row["false_positives"],

                "false_negatives":
                    row["false_negatives"],

                "evaluation":
                    "Random split",
            })

    # --------------------------------------------------------
    # WORD + CHARACTER
    # --------------------------------------------------------

    word_char = load_csv(
        WORD_CHAR_FILE
    )

    if word_char is not None:

        row = word_char.iloc[0]

        results.append({

            "experiment":
                "Word + Character TF-IDF",

            "representation":
                "Word + Character TF-IDF",

            "C":
                30,

            "accuracy":
                row["accuracy"],

            "precision":
                row["precision"],

            "recall":
                row["recall"],

            "f1":
                row["f1"],

            "roc_auc":
                row["roc_auc"],

            "false_positives":
                row["false_positives"],

            "false_negatives":
                row["false_negatives"],

            "evaluation":
                "Random split",
        })

    # --------------------------------------------------------
    # CHARACTER ONLY
    # --------------------------------------------------------

    char_only = load_csv(
        CHAR_ONLY_FILE
    )

    if char_only is not None:

        row = char_only.iloc[0]

        results.append({

            "experiment":
                "Character-only TF-IDF",

            "representation":
                "Character TF-IDF",

            "C":
                30,

            "accuracy":
                row["accuracy"],

            "precision":
                row["precision"],

            "recall":
                row["recall"],

            "f1":
                row["f1"],

            "roc_auc":
                row["roc_auc"],

            "false_positives":
                row["false_positives"],

            "false_negatives":
                row["false_negatives"],

            "evaluation":
                "Random split",
        })

    # --------------------------------------------------------
    # DATAFRAME
    # --------------------------------------------------------

    comparison = pd.DataFrame(
        results
    )

    comparison = comparison.sort_values(
        "f1",
        ascending=False,
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "CONSOLIDATED RESULTS"
    )

    print(
        "=" * 70
    )

    print(
        comparison.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.4f}"
        )
    )

    # --------------------------------------------------------
    # GROUP-AWARE DELTA
    # --------------------------------------------------------

    production = comparison[
        comparison["experiment"]
        == "Production random split"
    ].iloc[0]

    group = comparison[
        comparison["experiment"]
        == "Group-aware evaluation"
    ].iloc[0]

    print(
        "\n" + "=" * 70
    )

    print(
        "PRODUCTION vs GROUP-AWARE"
    )

    print(
        "=" * 70
    )

    metrics = [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
    ]

    for metric in metrics:

        difference = (
            group[metric]
            - production[metric]
        )

        print(
            f"{metric:12} "
            f"{difference:+.6f}"
        )

    # --------------------------------------------------------
    # MODEL DECISION
    # --------------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "MODEL SELECTION GUIDANCE"
    )

    print(
        "=" * 70
    )

    print(
        """
Production model:
    Word TF-IDF + Logistic Regression
    C=30

Group-aware evaluation:
    Word TF-IDF + Logistic Regression
    C=30

The group-aware model should be treated as
the stronger generalization benchmark because
its train/test groups do not overlap.

The production model is NOT automatically replaced.

Additional external-data testing should be performed
before declaring the final model production-ready.
"""
    )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    output_file = os.path.join(
        REPORT_DIR,
        "text_model_experiment_comparison.csv",
    )

    comparison.to_csv(
        output_file,
        index=False,
    )

    print(
        "\nComparison saved to:"
    )

    print(
        output_file
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "COMPARISON COMPLETE"
    )

    print(
        "=" * 70
    )


if __name__ == "__main__":
    main()
