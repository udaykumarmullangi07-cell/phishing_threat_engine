import re
import pandas as pd


def clean_text(text):
    """
    Day 4:
    Basic cleaning of raw email text.
    """

    # Handle null values
    if pd.isna(text):
        return ""

    # Convert to string and lowercase
    text = str(text).lower()

    # Remove HTML tags
    text = re.sub(r"<.*?>", " ", text)

    # Replace URLs with a common token
    text = re.sub(r"http\S+|www\S+", " URL ", text)

    # Remove punctuation, numbers and unwanted symbols
    text = re.sub(r"[^a-zA-Z\s]", " ", text)

    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


if __name__ == "__main__":

    input_file = "data/phishing_emails.csv"
    output_file = "data/cleaned_text.csv"

    df = pd.read_csv(input_file)

    print("Original shape:", df.shape)

    df["clean_text"] = df["text_combined"].apply(clean_text)

    df.to_csv(output_file, index=False)

    print("Cleaning completed.")
    print("Output shape:", df.shape)
    print("Saved to:", output_file)
