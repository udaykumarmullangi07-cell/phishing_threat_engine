import os
import joblib
import pandas as pd

from scipy.sparse import hstack

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


# =========================================================
# CONFIGURATION
# =========================================================

INPUT_FILE = "data/cleaned_text.csv"

MODEL_DIR = "models"

MODEL_FILE = os.path.join(
    MODEL_DIR,
    "experimental_word_char_model.joblib"
)

WORD_VECTORIZER_FILE = os.path.join(
    MODEL_DIR,
    "experimental_word_vectorizer.joblib"
)

CHAR_VECTORIZER_FILE = os.path.join(
    MODEL_DIR,
    "experimental_char_vectorizer.joblib"
)

RANDOM_STATE = 42


# =========================================================
# MAIN
# =========================================================

def main():

    print("=" * 70)
    print("WORD + CHARACTER MODEL — EXPERIMENTAL PACKAGING")
    print("=" * 70)

    print(
        "\nIMPORTANT:"
    )

    print(
        "Production model will NOT be modified."
    )

    # -----------------------------------------------------
    # Load dataset
    # -----------------------------------------------------

    print(
        "\nLoading dataset..."
    )

    df = pd.read_csv(
        INPUT_FILE
    )

    df["processed_text"] = (
        df["processed_text"]
        .fillna("")
    )

    X = df["processed_text"]
    y = df["label"]

    # -----------------------------------------------------
    # Same split
    # -----------------------------------------------------

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=RANDOM_STATE,
            stratify=y,
        )
    )

    print(
        "Training samples:",
        len(X_train)
    )

    print(
        "Testing samples:",
        len(X_test)
    )

    # -----------------------------------------------------
    # Word vectorizer
    # -----------------------------------------------------

    print(
        "\nTraining word TF-IDF..."
    )

    word_vectorizer = TfidfVectorizer(
        max_features=50000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
    )

    X_train_word = (
        word_vectorizer.fit_transform(
            X_train
        )
    )

    # -----------------------------------------------------
    # Character vectorizer
    # -----------------------------------------------------

    print(
        "Training character TF-IDF..."
    )

    char_vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(3, 5),
        min_df=3,
        max_features=30000,
        sublinear_tf=True,
    )

    X_train_char = (
        char_vectorizer.fit_transform(
            X_train
        )
    )

    # -----------------------------------------------------
    # Combine
    # -----------------------------------------------------

    X_train_combined = hstack(
        [
            X_train_word,
            X_train_char,
        ]
    ).tocsr()

    print(
        "\nCombined training shape:",
        X_train_combined.shape
    )

    # -----------------------------------------------------
    # Model
    # -----------------------------------------------------

    print(
        "\nTraining Logistic Regression..."
    )

    model = LogisticRegression(
        C=30,
        max_iter=1000,
        class_weight="balanced",
        solver="liblinear",
        random_state=RANDOM_STATE,
    )

    model.fit(
        X_train_combined,
        y_train,
    )

    # -----------------------------------------------------
    # Save
    # -----------------------------------------------------

    os.makedirs(
        MODEL_DIR,
        exist_ok=True,
    )

    joblib.dump(
        model,
        MODEL_FILE,
    )

    joblib.dump(
        word_vectorizer,
        WORD_VECTORIZER_FILE,
    )

    joblib.dump(
        char_vectorizer,
        CHAR_VECTORIZER_FILE,
    )

    # -----------------------------------------------------
    # Verify
    # -----------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "EXPERIMENTAL MODEL READY"
    )

    print(
        "=" * 70
    )

    print(
        "\nModel:"
    )

    print(
        MODEL_FILE
    )

    print(
        "\nWord vectorizer:"
    )

    print(
        WORD_VECTORIZER_FILE
    )

    print(
        "\nCharacter vectorizer:"
    )

    print(
        CHAR_VECTORIZER_FILE
    )

    print(
        "\nWord features:",
        X_train_word.shape[1]
    )

    print(
        "Character features:",
        X_train_char.shape[1]
    )

    print(
        "Total features:",
        X_train_combined.shape[1]
    )

    print(
        "\nProduction model remains unchanged."
    )

    print(
        "=" * 70
    )


if __name__ == "__main__":

    main()
