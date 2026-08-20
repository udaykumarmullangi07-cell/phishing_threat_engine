import os
import sys
import argparse


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

SRC_DIR = os.path.join(
    PROJECT_ROOT,
    "src"
)

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)


# ============================================================
# IMPORT URL PREDICTOR
# ============================================================

from realtime_url_predictor import (
    load_model,
    predict_url,
)


# ============================================================
# DISPLAY
# ============================================================

def print_result(result):
    """
    Display a URL prediction in a clean format.
    """

    print()
    print("=" * 70)
    print("URL THREAT DETECTION RESULT")
    print("=" * 70)

    print()
    print("URL:")
    print(result["url"])

    print()
    print(
        "Phishing probability:",
        result["phishing_probability"]
    )

    print(
        "Decision threshold  :",
        result["threshold"]
    )

    print(
        "Prediction          :",
        result["prediction"].upper()
    )

    print(
        "Risk level          :",
        result["risk"]
    )

    print()
    print("=" * 70)


# ============================================================
# ARGUMENT PARSER
# ============================================================

def parse_arguments():

    parser = argparse.ArgumentParser(
        description=(
            "Real-time phishing URL detector "
            "using the validated Top-25 Random Forest model."
        )
    )

    parser.add_argument(
        "url",
        help="URL to analyze"
    )

    return parser.parse_args()


# ============================================================
# MAIN
# ============================================================

def main():

    args = parse_arguments()

    url = args.url.strip()

    if not url:

        print(
            "ERROR: URL cannot be empty."
        )

        sys.exit(1)

    try:

        # ----------------------------------------------------
        # Load the frozen model
        # ----------------------------------------------------

        model_package = load_model()

        # ----------------------------------------------------
        # Perform prediction
        # ----------------------------------------------------

        result = predict_url(
            url,
            model_package
        )

        # ----------------------------------------------------
        # Display result
        # ----------------------------------------------------

        print_result(
            result
        )

    except Exception as error:

        print()
        print(
            "ERROR:",
            error
        )

        sys.exit(1)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()
