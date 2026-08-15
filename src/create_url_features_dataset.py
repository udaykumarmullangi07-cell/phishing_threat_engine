import pandas as pd

from url_features import extract_url_features


INPUT_FILE = "data/url_dataset.csv"
OUTPUT_FILE = "data/url_features.csv"


def main():
    # Load original URL dataset
    df = pd.read_csv(INPUT_FILE)

    print("Original shape:", df.shape)

    # Extract our 10 URL features
    feature_rows = df["url"].apply(extract_url_features)

    # Convert feature dictionaries into DataFrame
    features_df = pd.DataFrame(feature_rows.tolist())

    # Keep original URL and label
    result_df = pd.concat(
        [
            df[["url", "status"]].reset_index(drop=True),
            features_df.reset_index(drop=True),
        ],
        axis=1,
    )

    # Reorder columns
    columns = [
        "url",
        "url_length",
        "dot_count",
        "has_https",
        "has_ip_address",
        "special_chars",
        "subdomain_count",
        "path_depth",
        "digit_ratio",
        "has_at_symbol",
        "domain_length",
        "status",
    ]

    result_df = result_df[columns]

    # Save new dataset
    result_df.to_csv(OUTPUT_FILE, index=False)

    print("URL feature extraction completed.")
    print("Output shape:", result_df.shape)
    print("Output columns:", result_df.columns.tolist())
    print("Saved to:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
