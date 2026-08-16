import re
import tldextract
from urllib.parse import urlparse


def extract_url_features(url):
    """
    Extract 10 URL-based phishing features.

    IMPORTANT:
    The feature names and calculations are kept compatible
    with the trained URL Random Forest model.
    """

    url = str(url)

    # Extract domain information
    ext = tldextract.extract(url)

    # Parse URL components
    parsed_url = urlparse(url)

    features = {
        # 1. Total URL length
        "url_length": len(url),

        # 2. Number of dots
        "dot_count": url.count("."),

        # 3. HTTPS present
        "has_https": int(
            url.lower().startswith("https://")
        ),

        # 4. IP address used instead of domain
        "has_ip_address": int(
            bool(
                re.search(
                    r"\d+\.\d+\.\d+\.\d+",
                    url
                )
            )
        ),

        # 5. Number of suspicious special characters
        "special_chars": len(
            re.findall(
                r"[@!?=&%]",
                url
            )
        ),

        # 6. Number of subdomains
        "subdomain_count": (
            len(
                ext.subdomain.split(".")
            )
            if ext.subdomain
            else 0
        ),

        # 7. URL path depth
        # Count "/" only inside the actual path
        "path_depth": parsed_url.path.count("/"),

        # 8. Ratio of digits to total URL length
        "digit_ratio": (
            sum(
                c.isdigit()
                for c in url
            )
            / max(len(url), 1)
        ),

        # 9. @ symbol present
        "has_at_symbol": int(
            "@" in url
        ),

        # 10. Domain length
        "domain_length": len(
            ext.domain
        ),
    }

    return features


# =========================================================
# URL EXPLANATION
# =========================================================

def explain_url_features(url):
    """
    Generate human-readable explanations
    from the extracted URL features.

    This function does NOT change the ML features.
    It only explains potentially suspicious characteristics.
    """

    features = extract_url_features(url)

    indicators = []

    # -----------------------------------------------------
    # URL length
    # -----------------------------------------------------

    if features["url_length"] >= 75:

        indicators.append(
            "URL is unusually long"
        )

    elif features["url_length"] >= 50:

        indicators.append(
            "URL is relatively long"
        )

    # -----------------------------------------------------
    # IP address
    # -----------------------------------------------------

    if features["has_ip_address"] == 1:

        indicators.append(
            "URL uses an IP address instead of a domain"
        )

    # -----------------------------------------------------
    # HTTPS
    # -----------------------------------------------------

    if features["has_https"] == 0:

        indicators.append(
            "URL does not use HTTPS"
        )

    # -----------------------------------------------------
    # Special characters
    # -----------------------------------------------------

    if features["special_chars"] >= 4:

        indicators.append(
            "URL contains many special characters"
        )

    elif features["special_chars"] >= 2:

        indicators.append(
            "URL contains multiple special characters"
        )

    # -----------------------------------------------------
    # Subdomains
    # -----------------------------------------------------

    if features["subdomain_count"] >= 3:

        indicators.append(
            "URL contains multiple subdomains"
        )

    elif features["subdomain_count"] >= 2:

        indicators.append(
            "URL contains several subdomains"
        )

    # -----------------------------------------------------
    # Path depth
    # -----------------------------------------------------

    if features["path_depth"] >= 4:

        indicators.append(
            "URL has a deep path structure"
        )

    elif features["path_depth"] >= 3:

        indicators.append(
            "URL has several path levels"
        )

    # -----------------------------------------------------
    # Digit ratio
    # -----------------------------------------------------

    if features["digit_ratio"] >= 0.25:

        indicators.append(
            "URL contains a high proportion of digits"
        )

    elif features["digit_ratio"] >= 0.15:

        indicators.append(
            "URL contains a noticeable number of digits"
        )

    # -----------------------------------------------------
    # @ symbol
    # -----------------------------------------------------

    if features["has_at_symbol"] == 1:

        indicators.append(
            "URL contains an @ symbol"
        )

    # -----------------------------------------------------
    # Domain length
    # -----------------------------------------------------

    if features["domain_length"] >= 20:

        indicators.append(
            "Domain name is unusually long"
        )

    # -----------------------------------------------------
    # Final result
    # -----------------------------------------------------

    return {
        "features": features,
        "indicators": indicators,
    }


# =========================================================
# TEST
# =========================================================

if __name__ == "__main__":

    test_urls = [

        "https://www.google.com",

        "http://192.168.1.10/login",

        "https://secure-login.example.com/account/verify",

        (
            "http://secure-login.example.com/"
            "account/verify/update?id=12345"
            "&token=98765"
        ),
    ]

    for test_url in test_urls:

        print("\n" + "=" * 70)
        print("URL:", test_url)
        print("=" * 70)

        result = explain_url_features(
            test_url
        )

        print("\nFeatures:")

        for name, value in result["features"].items():

            print(
                f"{name}: {value}"
            )

        print("\nIndicators:")

        if result["indicators"]:

            for indicator in result["indicators"]:

                print(
                    f"- {indicator}"
                )

        else:

            print(
                "No obvious suspicious indicators detected."
            )
