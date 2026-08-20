import json
import os
from datetime import datetime

import joblib


# =========================================================
# SOURCE MODEL FILES
# =========================================================

SOURCE_MODEL_FILE = (
    "models/text_logistic_regression_tuned.joblib"
)

SOURCE_VECTORIZER_FILE = (
    "models/text_tfidf_vectorizer_tuned.joblib"
)


# =========================================================
# FINAL MODEL FILES
# =========================================================

FINAL_MODEL_FILE = (
    "models/final_text_model.joblib"
)

FINAL_VECTORIZER_FILE = (
    "models/final_text_vectorizer.joblib"
)

METADATA_FILE = (
    "models/final_text_model_metadata.json"
)


# =========================================================
# FINAL MODEL INFORMATION
# =========================================================

MODEL_NAME = "Logistic Regression"

MODEL_VERSION = "1.0"

MODEL_PURPOSE = (
    "Phishing email/message classification"
)


# =========================================================
# MAIN
# =========================================================

def main():

    print("=" * 70)
    print("FINAL TEXT MODEL PACKAGING")
    print("=" * 70)


    # =====================================================
    # 1. CHECK SOURCE FILES
    # =====================================================

    print("\nChecking source model files...")

    if not os.path.exists(
        SOURCE_MODEL_FILE
    ):
        raise FileNotFoundError(
            f"Source model not found: "
            f"{SOURCE_MODEL_FILE}"
        )

    if not os.path.exists(
        SOURCE_VECTORIZER_FILE
    ):
        raise FileNotFoundError(
            f"Source vectorizer not found: "
            f"{SOURCE_VECTORIZER_FILE}"
        )


    print(
        "Source model: OK"
    )

    print(
        "Source vectorizer: OK"
    )


    # =====================================================
    # 2. LOAD TUNED MODEL
    # =====================================================

    print(
        "\nLoading tuned Logistic Regression..."
    )

    model = joblib.load(
        SOURCE_MODEL_FILE
    )


    # =====================================================
    # 3. LOAD TUNED VECTORIZER
    # =====================================================

    print(
        "Loading tuned TF-IDF vectorizer..."
    )

    vectorizer = joblib.load(
        SOURCE_VECTORIZER_FILE
    )


    # =====================================================
    # 4. VERIFY MODEL
    # =====================================================

    print(
        "\nVerifying model configuration..."
    )

    print(
        f"Model type       : "
        f"{type(model).__name__}"
    )

    print(
        f"C                : "
        f"{model.C}"
    )

    print(
        f"Solver           : "
        f"{model.solver}"
    )

    print(
        f"Class weight     : "
        f"{model.class_weight}"
    )

    print(
        f"Max iterations   : "
        f"{model.max_iter}"
    )


    # =====================================================
    # 5. VERIFY VECTORIZER
    # =====================================================

    feature_count = len(
        vectorizer.get_feature_names_out()
    )

    print(
        "\nVerifying TF-IDF configuration..."
    )

    print(
        f"Feature count    : "
        f"{feature_count}"
    )

    print(
        f"N-gram range     : "
        f"{vectorizer.ngram_range}"
    )

    print(
        f"Minimum document : "
        f"{vectorizer.min_df}"
    )

    print(
        f"Maximum document : "
        f"{vectorizer.max_df}"
    )

    print(
        f"Sublinear TF     : "
        f"{vectorizer.sublinear_tf}"
    )


    # =====================================================
    # 6. SAVE FINAL MODEL
    # =====================================================

    print(
        "\nSaving final model..."
    )

    joblib.dump(
        model,
        FINAL_MODEL_FILE
    )


    # =====================================================
    # 7. SAVE FINAL VECTORIZER
    # =====================================================

    print(
        "Saving final vectorizer..."
    )

    joblib.dump(
        vectorizer,
        FINAL_VECTORIZER_FILE
    )


    # =====================================================
    # 8. CREATE METADATA
    # =====================================================

    metadata = {

        "model_name":
            MODEL_NAME,

        "model_version":
            MODEL_VERSION,

        "purpose":
            MODEL_PURPOSE,

        "model_file":
            FINAL_MODEL_FILE,

        "vectorizer_file":
            FINAL_VECTORIZER_FILE,

        "source_model":
            SOURCE_MODEL_FILE,

        "source_vectorizer":
            SOURCE_VECTORIZER_FILE,

        "model_configuration": {

            "algorithm":
                "Logistic Regression",

            "C":
                model.C,

            "solver":
                model.solver,

            "class_weight":
                model.class_weight,

            "max_iter":
                model.max_iter,

            "random_state":
                model.random_state,

        },

        "tfidf_configuration": {

            "max_features":
                vectorizer.max_features,

            "ngram_range":
                list(
                    vectorizer.ngram_range
                ),

            "min_df":
                vectorizer.min_df,

            "max_df":
                vectorizer.max_df,

            "sublinear_tf":
                vectorizer.sublinear_tf,

        },

        "feature_count":
            feature_count,

        "training_configuration": {

            "test_size":
                0.20,

            "random_state":
                42,

            "stratified_split":
                True,

        },

        "evaluation_metrics": {

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

        },

        "best_hyperparameter":
            {
                "C": 30
            },

        "created_at":
            datetime.now().isoformat(),

    }


    # =====================================================
    # 9. SAVE METADATA
    # =====================================================

    print(
        "Saving model metadata..."
    )

    with open(
        METADATA_FILE,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metadata,
            file,
            indent=4,
        )


    # =====================================================
    # 10. FINAL SUMMARY
    # =====================================================

    print("\n" + "=" * 70)
    print("FINAL TEXT MODEL READY")
    print("=" * 70)

    print(
        f"\nFinal model:"
    )

    print(
        FINAL_MODEL_FILE
    )

    print(
        "\nFinal vectorizer:"
    )

    print(
        FINAL_VECTORIZER_FILE
    )

    print(
        "\nMetadata:"
    )

    print(
        METADATA_FILE
    )

    print(
        "\nModel:"
    )

    print(
        MODEL_NAME
    )

    print(
        "\nBest C:"
    )

    print(
        model.C
    )

    print(
        "\nTF-IDF features:"
    )

    print(
        feature_count
    )

    print(
        "\nFinal F1:"
    )

    print(
        "0.9914"
    )

    print(
        "\nFinal ROC-AUC:"
    )

    print(
        "0.9994"
    )

    print("\n" + "=" * 70)


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":

    main()
