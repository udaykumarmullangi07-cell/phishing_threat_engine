import json
import os

import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = "data/url_dataset.csv"
REPORT_FILE = "reports/url_lexical_reputation_robustness.json"

RANDOM_STATE = 42


# ============================================================
# FEATURES
# ============================================================

LEXICAL_FEATURES = [
    "length_url",
    "length_hostname",
    "ip",
    "nb_dots",
    "nb_hyphens",
    "nb_at",
    "nb_qm",
    "nb_and",
    "nb_eq",
    "nb_underscore",
    "nb_tilde",
    "nb_percent",
    "nb_slash",
    "nb_star",
    "nb_colon",
    "nb_comma",
    "nb_semicolumn",
    "nb_dollar",
    "nb_space",
    "nb_www",
    "nb_com",
    "nb_dslash",
    "http_in_path",
    "https_token",
    "ratio_digits_url",
    "ratio_digits_host",
    "punycode",
    "port",
    "tld_in_path",
    "tld_in_subdomain",
    "abnormal_subdomain",
    "nb_subdomains",
    "prefix_suffix",
    "random_domain",
    "shortening_service",
    "path_extension",
    "length_words_raw",
    "char_repeat",
    "shortest_words_raw",
    "shortest_word_host",
    "shortest_word_path",
    "longest_words_raw",
    "longest_word_host",
    "longest_word_path",
    "avg_words_raw",
    "avg_word_host",
    "avg_word_path",
    "phish_hints",
    "domain_in_brand",
    "brand_in_subdomain",
    "brand_in_path",
    "suspecious_tld",
]

REPUTATION_FEATURES = [
    "whois_registered_domain",
    "domain_registration_length",
    "domain_age",
    "web_traffic",
    "dns_record",
    "google_index",
    "page_rank",
]

FEATURES = list(
    dict.fromkeys(
        LEXICAL_FEATURES + REPUTATION_FEATURES
    )
)


# ============================================================
# EXTERNAL ROBUSTNESS TEST CASES
# ============================================================

TEST_URLS = [

    # --------------------------------------------------------
    # LEGITIMATE
    # --------------------------------------------------------

    {
        "name": "Google",
        "url": "https://www.google.com",
        "expected": 0,
    },

    {
        "name": "Microsoft",
        "url": "https://www.microsoft.com",
        "expected": 0,
    },

    {
        "name": "GitHub",
        "url": "https://github.com",
        "expected": 0,
    },

    {
        "name": "Amazon India",
        "url": "https://www.amazon.in",
        "expected": 0,
    },

    {
        "name": "IIT Delhi",
        "url": "https://home.iitd.ac.in",
        "expected": 0,
    },

    {
        "name": "NIT Warangal",
        "url": "https://www.nitw.ac.in",
        "expected": 0,
    },

    {
        "name": "Python",
        "url": "https://www.python.org",
        "expected": 0,
    },

    {
        "name": "Ubuntu",
        "url": "https://ubuntu.com",
        "expected": 0,
    },

    {
        "name": "GitHub Repository",
        "url": "https://github.com/python/cpython/issues",
        "expected": 0,
    },

    {
        "name": "Microsoft Documentation",
        "url": (
            "https://learn.microsoft.com/en-us/"
            "documentation/"
        ),
        "expected": 0,
    },


    # --------------------------------------------------------
    # PHISHING-STYLE
    # --------------------------------------------------------

    {
        "name": "IP Login",
        "url": "http://192.168.1.25/login",
        "expected": 1,
    },

    {
        "name": "Fake Bank Login",
        "url": (
            "http://secure-bank-login.example.com/"
            "account/verify"
        ),
        "expected": 1,
    },

    {
        "name": "Account Verification",
        "url": (
            "http://account-security.example.net/"
            "login/verify/update"
        ),
        "expected": 1,
    },

    {
        "name": "Password Reset",
        "url": (
            "http://secure-update.example.org/"
            "password/reset/account"
        ),
        "expected": 1,
    },

    {
        "name": "Long Suspicious URL",
        "url": (
            "http://secure-login.example.com/"
            "account/verify/update?"
            "session=983472983472"
            "&token=928374928374"
            "&confirm=1"
        ),
        "expected": 1,
    },

    {
        "name": "IP Payment URL",
        "url": (
            "http://185.23.44.91/"
            "secure/login/payment/verify"
        ),
        "expected": 1,
    },

    {
        "name": "Suspicious Subdomains",
        "url": (
            "http://login.secure.account.verify.example.com/"
            "update"
        ),
        "expected": 1,
    },

    {
        "name": "Credential Verification",
        "url": (
            "http://verify-account.example.com/"
            "credential/confirmation/login"
        ),
        "expected": 1,
    },

    {
        "name": "Fake Cloud Login",
        "url": (
            "http://cloud-security.example.net/"
            "signin/account/verify/password"
        ),
        "expected": 1,
    },

    {
        "name": "Suspicious Numeric URL",
        "url": (
            "http://secure-login.example.com/"
            "verify/839274982374982374"
        ),
        "expected": 1,
    },
]


# ============================================================
# TRAIN EXPERIMENTAL MODEL
# ============================================================

def train_model():

    print("\nLoading dataset...")

    df = pd.read_csv(INPUT_FILE)

    print(
        f"Dataset samples: {len(df)}"
    )

    missing = [
        feature
        for feature in FEATURES
        if feature not in df.columns
    ]

    if missing:

        raise ValueError(
            f"Missing features: {missing}"
        )

    # Remove constant features
    usable_features = [
        feature
        for feature in FEATURES
        if df[feature].nunique() > 1
    ]

    y = (
        df["status"]
        .map({
            "legitimate": 0,
            "phishing": 1,
        })
        .astype(int)
    )

    X = df[usable_features]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print(
        f"Training samples: {len(X_train)}"
    )

    print(
        f"Testing samples : {len(X_test)}"
    )

    print("\nTraining RF-500-depth20...")

    model = RandomForestClassifier(
        n_estimators=500,
        max_depth=20,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="sqrt",
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    model.fit(
        X_train,
        y_train
    )

    predictions = model.predict(X_test)

    tn, fp, fn, tp = confusion_matrix(
        y_test,
        predictions
    ).ravel()

    print("\nValidation result:")
    print(f"True Negatives : {tn}")
    print(f"False Positives: {fp}")
    print(f"False Negatives: {fn}")
    print(f"True Positives : {tp}")

    return model, usable_features


# ============================================================
# EXTERNAL URL FEATURE EXTRACTION
# ============================================================

def extract_external_features(
    model_features,
    url,
):

    # Import our existing feature extractor.
    from url_features import extract_url_features

    # IMPORTANT:
    # The current url_features.py produces only the
    # 10 engineered features. Therefore we cannot fabricate
    # the remaining 49 dataset features.
    #
    # This robustness test therefore maps only the available
    # features and leaves unavailable dataset-derived features
    # at zero.
    #
    # This is intentionally conservative and is NOT being used
    # as a production prediction pipeline.

    extracted = extract_url_features(url)

    row = {}

    for feature in model_features:

        if feature in extracted:

            row[feature] = extracted[feature]

        else:

            row[feature] = 0

    return pd.DataFrame(
        [row],
        columns=model_features,
    )


# ============================================================
# MAIN ROBUSTNESS TEST
# ============================================================

def main():

    print("=" * 70)
    print("URL MODEL — LEXICAL + REPUTATION ROBUSTNESS TEST")
    print("=" * 70)

    print("""
Purpose:
Evaluate the experimental RF-500-depth20 model against
realistic legitimate and phishing-style URLs.

Production URL model will NOT be modified.
""")

    model, model_features = train_model()

    results = []

    correct = 0
    false_positives = 0
    false_negatives = 0

    print("\n")
    print("=" * 70)
    print("EXTERNAL ROBUSTNESS TESTS")
    print("=" * 70)

    for index, test in enumerate(
        TEST_URLS,
        start=1,
    ):

        url = test["url"]
        expected = test["expected"]

        X_url = extract_external_features(
            model_features,
            url,
        )

        probability = model.predict_proba(
            X_url
        )[0][1]

        prediction = int(
            probability >= 0.5
        )

        passed = (
            prediction == expected
        )

        if passed:

            correct += 1

            result_text = "PASS"

        else:

            result_text = "FAIL"

            if expected == 0 and prediction == 1:

                false_positives += 1

            elif expected == 1 and prediction == 0:

                false_negatives += 1

        print("\n" + "-" * 70)

        print(
            f"TEST {index}: {test['name']}"
        )

        print(
            f"URL: {url}"
        )

        print(
            f"Expected: "
            f"{'PHISHING' if expected else 'LEGITIMATE'}"
        )

        print(
            f"Legitimate probability: "
            f"{1 - probability:.4f}"
        )

        print(
            f"Phishing probability:   "
            f"{probability:.4f}"
        )

        print(
            f"Prediction: "
            f"{'PHISHING' if prediction else 'LEGITIMATE'}"
        )

        print(
            f"Result: [ {result_text} ]"
        )

        results.append(
            {
                "name": test["name"],
                "url": url,
                "expected": expected,
                "prediction": prediction,
                "legitimate_probability":
                    float(1 - probability),
                "phishing_probability":
                    float(probability),
                "passed": bool(passed),
            }
        )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    total = len(TEST_URLS)

    accuracy = (
        correct / total
        if total
        else 0
    )

    print("\n")
    print("=" * 70)
    print("ROBUSTNESS SUMMARY")
    print("=" * 70)

    print(
        f"Total tests       : {total}"
    )

    print(
        f"Correct           : {correct}"
    )

    print(
        f"False positives   : {false_positives}"
    )

    print(
        f"False negatives   : {false_negatives}"
    )

    print(
        f"Accuracy          : {accuracy:.2%}"
    )

    # --------------------------------------------------------
    # HIGH-RISK LEGITIMATE URLS
    # --------------------------------------------------------

    legitimate_results = [
        r
        for r in results
        if r["expected"] == 0
    ]

    if legitimate_results:

        highest_fp = max(
            legitimate_results,
            key=lambda r:
                r["phishing_probability"],
        )

        print("\nHighest-risk legitimate URL:")

        print(
            highest_fp["name"]
        )

        print(
            f"Phishing probability: "
            f"{highest_fp['phishing_probability']:.2%}"
        )

    # --------------------------------------------------------
    # LOWEST-RISK PHISHING URL
    # --------------------------------------------------------

    phishing_results = [
        r
        for r in results
        if r["expected"] == 1
    ]

    if phishing_results:

        lowest_phishing = min(
            phishing_results,
            key=lambda r:
                r["phishing_probability"],
        )

        print("\nLowest-risk phishing-style URL:")

        print(
            lowest_phishing["name"]
        )

        print(
            f"Phishing probability: "
            f"{lowest_phishing['phishing_probability']:.2%}"
        )

    # --------------------------------------------------------
    # SAVE REPORT
    # --------------------------------------------------------

    os.makedirs(
        os.path.dirname(REPORT_FILE),
        exist_ok=True,
    )

    report = {
        "model": "RandomForestClassifier",
        "configuration": {
            "n_estimators": 500,
            "max_depth": 20,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "max_features": "sqrt",
            "class_weight": "balanced",
        },
        "feature_count": len(model_features),
        "feature_names": model_features,
        "total_tests": total,
        "correct": correct,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "accuracy": accuracy,
        "tests": results,
    }

    with open(
        REPORT_FILE,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            report,
            f,
            indent=4,
        )

    print("\n")
    print(
        "Robustness report saved to:"
    )

    print(
        REPORT_FILE
    )

    print("\n")
    print("=" * 70)
    print("URL ROBUSTNESS TEST COMPLETE")
    print("=" * 70)

    print(
        "\nIMPORTANT:"
    )

    print(
        "The production URL model was NOT modified."
    )

    print(
        "\nNOTE:"
    )

    print(
        "This experiment intentionally does not treat "
        "the external URLs as production predictions."
    )

    print(
        "The current URL feature extractor does not "
        "provide all 59 Lexical + Reputation features."
    )


if __name__ == "__main__":
    main()
