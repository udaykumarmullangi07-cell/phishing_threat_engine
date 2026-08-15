import re
import tldextract
from urllib.parse import urlparse


def extract_url_features(url):
    """
    Extract 10 URL-based phishing features.
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
        "has_https": int(url.lower().startswith("https://")),

        # 4. IP address used instead of domain
        "has_ip_address": int(
            bool(re.search(r"\d+\.\d+\.\d+\.\d+", url))
        ),

        # 5. Number of suspicious special characters
        "special_chars": len(
            re.findall(r"[@!?=&%]", url)
        ),

        # 6. Number of subdomains
        "subdomain_count": (
            len(ext.subdomain.split("."))
            if ext.subdomain
            else 0
        ),

        # 7. URL path depth
        # Count "/" only inside the actual path
        "path_depth": parsed_url.path.count("/"),

        # 8. Ratio of digits to total URL length
        "digit_ratio": (
            sum(c.isdigit() for c in url) / max(len(url), 1)
        ),

        # 9. @ symbol present
        "has_at_symbol": int("@" in url),

        # 10. Domain length
        "domain_length": len(ext.domain),
    }

    return features


if __name__ == "__main__":

    test_urls = [
        "https://www.google.com",
        "http://192.168.1.10/login",
        "https://secure-login.example.com/account/verify",
    ]

    for test_url in test_urls:

        print("\nURL:", test_url)

        features = extract_url_features(test_url)

        for name, value in features.items():
            print(f"{name}: {value}")
