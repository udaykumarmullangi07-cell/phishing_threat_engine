import pandas as pd
import nltk

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()


def preprocess_text(text):
    """
    Tokenization + stopword removal + stemming.
    """

    if pd.isna(text):
        return ""

    tokens = nltk.word_tokenize(str(text))

    processed_tokens = [
        stemmer.stem(word)
        for word in tokens
        if word.isalpha() and word not in stop_words
    ]

    return " ".join(processed_tokens)


if __name__ == "__main__":

    input_file = "data/cleaned_text.csv"

    df = pd.read_csv(input_file)

    print("Input shape:", df.shape)

    df["processed_text"] = df["clean_text"].apply(preprocess_text)

    df.to_csv(input_file, index=False)

    print("NLP preprocessing completed.")
    print("Output shape:", df.shape)
    print("Saved to:", input_file)
