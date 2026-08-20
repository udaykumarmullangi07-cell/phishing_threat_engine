import re
from urllib.parse import urlparse


# =========================================================
# URL FEATURE EXTRACTION
# =========================================================
#
# This module extracts URL-local features.
#
# No external services are used:
#
# - WHOIS              : NO
# - DNS reputation     : NO
# - Google index       : NO
# - PageRank           : NO
# - Web traffic        : NO
# - Webpage fetching   : NO
# - Brand database     : NO
#
# IMPORTANT:
# The feature names used here must remain compatible with
# the real-time URL model.
# =========================================================


# =========================================================
# PHISHING HINTS
# =========================================================

PHISHING_HINT_WORDS = [
    "login",
    "signin",
    "sign-in",
    "verify",
    "verification",
    "account",
    "update",
    "secure",
    "security",
    "confirm",
    "confirmation",
    "password",
    "credential",
    "payment",
    "billing",
    "wallet",
    "recover",
    "recovery",
    "unlock",
    "suspended",
    "suspension",
    "authenticate",
    "authentication",
    "authorize",
    "authorization",
    "bank",
    "paypal",
    "apple",
    "microsoft",
    "google",
    "facebook",
    "amazon",
    "instagram",
    "linkedin",
]


# =========================================================
# SUSPICIOUS TLDs
# =========================================================

SUSPICIOUS_TLDS = {
    "zip",
    "review",
    "country",
    "kim",
    "cricket",
    "science",
    "work",
    "party",
    "top",
    "click",
    "download",
    "stream",
    "gq",
    "tk",
    "ml",
    "ga",
    "cf",
}


# =========================================================
# URL SHORTENING SERVICES
# =========================================================

SHORTENING_SERVICES = {
    "bit.ly",
    "goo.gl",
    "tinyurl.com",
    "ow.ly",
    "t.co",
    "is.gd",
    "buff.ly",
    "cutt.ly",
    "rb.gy",
    "shorturl.at",
    "rebrand.ly",
    "tiny.cc",
    "lnkd.in",
}


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def _safe_urlparse(url):
    """
    Parse URL safely.

    If a scheme is missing, temporarily add http://
    so that urlparse can correctly identify the hostname.
    """

    url = str(url).strip()

    if not re.match(
        r"^[a-zA-Z][a-zA-Z0-9+.-]*://",
        url,
    ):

        url_for_parse = (
            "http://" + url
        )

    else:

        url_for_parse = url

    return urlparse(
        url_for_parse
    )


def _is_ip_address(hostname):
    """
    Determine whether hostname is an IPv4 address.
    """

    if not hostname:
        return 0

    ipv4_pattern = (
        r"^(?:\d{1,3}\.){3}\d{1,3}$"
    )

    return int(
        bool(
            re.match(
                ipv4_pattern,
                hostname,
            )
        )
    )


def _ratio_digits(text):
    """
    Calculate ratio of digits to characters.

    Returns 0.0 for empty strings.
    """

    if not text:
        return 0.0

    digit_count = sum(
        char.isdigit()
        for char in text
    )

    return (
        digit_count / len(text)
    )


def _split_words(text):
    """
    Split URL components into lexical words.
    """

    if not text:
        return []

    words = re.split(
        r"[^A-Za-z0-9]+",
        text,
    )

    return [
        word
        for word in words
        if word
    ]


def _character_repeat_count(text):
    """
    Calculate a simple character repetition score.

    Consecutive repeated characters are counted.

    Example:
        aabbcc -> 3 repeated transitions

    This feature is used by the model.
    """

    if not text:
        return 0

    repeat_count = 0

    for i in range(
        1,
        len(text),
    ):

        if text[i] == text[i - 1]:

            repeat_count += 1

    return repeat_count


def _calculate_word_statistics(words):
    """
    Calculate lexical statistics.
    """

    if not words:

        return {
            "shortest": 0,
            "longest": 0,
            "average": 0.0,
        }

    lengths = [
        len(word)
        for word in words
    ]

    return {
        "shortest": min(lengths),
        "longest": max(lengths),
        "average": sum(lengths) / len(lengths),
    }


# =========================================================
# REAL-TIME URL FEATURE EXTRACTION
# =========================================================

def extract_url_features(url):
    """
    Extract URL-local features.

    Returns a dictionary containing the real-time
    URL features used by the production Top-25 model
    and the broader experimental feature set.
    """

    url = str(url).strip()

    parsed = _safe_urlparse(
        url
    )

    hostname = (
        parsed.hostname
        or ""
    )

    path = (
        parsed.path
        or ""
    )

    query = (
        parsed.query
        or ""
    )

    fragment = (
        parsed.fragment
        or ""
    )

    # -----------------------------------------------------
    # Basic URL components
    # -----------------------------------------------------

    hostname_lower = hostname.lower()

    url_lower = url.lower()

    raw_path = (
        path
        + "?"
        + query
        + "#"
        + fragment
    )

    # -----------------------------------------------------
    # Word extraction
    # -----------------------------------------------------

    raw_words = _split_words(
        url
    )

    host_words = _split_words(
        hostname
    )

    path_words = _split_words(
        path
    )

    raw_stats = _calculate_word_statistics(
        raw_words
    )

    host_stats = _calculate_word_statistics(
        host_words
    )

    path_stats = _calculate_word_statistics(
        path_words
    )

    # -----------------------------------------------------
    # TLD
    # -----------------------------------------------------

    tld = ""

    if "." in hostname:

        tld = (
            hostname_lower
            .split(".")[-1]
        )

    # -----------------------------------------------------
    # Subdomain count
    # -----------------------------------------------------

    if hostname:

        hostname_parts = [
            part
            for part in hostname_lower.split(".")
            if part
        ]

        if len(hostname_parts) >= 3:

            nb_subdomains = (
                len(hostname_parts) - 2
            )

        else:

            nb_subdomains = 0

    else:

        nb_subdomains = 0

    # -----------------------------------------------------
    # Query information
    # -----------------------------------------------------

    query_parameter_count = 0

    if query:

        query_parameter_count = (
            len(
                [
                    part
                    for part in query.split("&")
                    if part
                ]
            )
        )

    # -----------------------------------------------------
    # Phishing hints
    # -----------------------------------------------------

    phish_hints = 0

    for hint in PHISHING_HINT_WORDS:

        if hint in url_lower:

            phish_hints += 1

    # -----------------------------------------------------
    # Domain / brand-related placeholders
    #
    # These require a brand vocabulary for meaningful
    # calculation and therefore remain None.
    # -----------------------------------------------------

    domain_in_brand = None
    brand_in_subdomain = None
    brand_in_path = None

    # -----------------------------------------------------
    # Port
    # -----------------------------------------------------

    port = 0

    try:

        if parsed.port is not None:

            # Standard ports are not treated as suspicious.
            if parsed.port not in {
                80,
                443,
            }:

                port = 1

    except ValueError:

        port = 1

    # -----------------------------------------------------
    # Path extension
    # -----------------------------------------------------

    path_extension = 0

    path_last_part = (
        path.rstrip("/")
        .split("/")[-1]
        if path
        else ""
    )

    if "." in path_last_part:

        extension = (
            path_last_part
            .split(".")[-1]
            .lower()
        )

        if (
            extension
            and len(extension) <= 10
            and extension.isalnum()
        ):

            path_extension = 1

    # -----------------------------------------------------
    # URL feature dictionary
    # -----------------------------------------------------

    features = {

        # =================================================
        # Length
        # =================================================

        "length_url":
            len(url),

        "length_hostname":
            len(hostname),

        # =================================================
        # Structural
        # =================================================

        "ip":
            _is_ip_address(hostname),

        "nb_dots":
            url.count("."),

        "nb_hyphens":
            url.count("-"),

        "nb_at":
            url.count("@"),

        "nb_qm":
            url.count("?"),

        "nb_and":
            url.count("&"),

        "nb_eq":
            url.count("="),

        "nb_underscore":
            url.count("_"),

        "nb_tilde":
            url.count("~"),

        "nb_percent":
            url.count("%"),

        "nb_slash":
            url.count("/"),

        "nb_star":
            url.count("*"),

        "nb_colon":
            url.count(":"),

        "nb_comma":
            url.count(","),

        "nb_semicolumn":
            url.count(";"),

        "nb_dollar":
            url.count("$"),

        "nb_space":
            url.count(" "),

        # =================================================
        # Tokens
        # =================================================

        "nb_www":
            url_lower.count("www"),

        "nb_com":
            url_lower.count(".com"),

        "nb_dslash":
            url.count("//") - 1
            if url.count("//") > 0
            else 0,

        "http_in_path":
            int(
                "http" in path.lower()
                or "https" in path.lower()
            ),

        # =================================================
        # HTTPS
        # =================================================

        "https_token":
            int(
                "https" in hostname_lower
                or "https" in path.lower()
            ),

        # =================================================
        # Ratios
        # =================================================

        "ratio_digits_url":
            _ratio_digits(url),

        "ratio_digits_host":
            _ratio_digits(hostname),

        # =================================================
        # Domain characteristics
        # =================================================

        "punycode":
            int(
                "xn--" in hostname_lower
            ),

        "port":
            port,

        "tld_in_path":
            int(
                bool(tld)
                and tld in path.lower().split(".")
            ),

        "tld_in_subdomain":
            int(
                bool(tld)
                and any(
                    part == tld
                    for part in hostname_lower.split(".")[:-1]
                )
            ),

        "abnormal_subdomain":
            int(
                nb_subdomains >= 3
            ),

        "nb_subdomains":
            nb_subdomains,

        "prefix_suffix":
            int(
                "-" in hostname
            ),

        "random_domain":
            0,

        "shortening_service":
            int(
                hostname_lower
                in SHORTENING_SERVICES
            ),

        "path_extension":
            path_extension,

        # =================================================
        # Redirection indicators
        # =================================================

        "nb_redirection":
            query_parameter_count
            if (
                "redirect" in url_lower
                or "redir" in url_lower
                or "return" in url_lower
                or "url=" in url_lower
            )
            else 0,

        "nb_external_redirection":
            int(
                (
                    "http://" in query.lower()
                    or "https://" in query.lower()
                )
                and (
                    "http://" in url_lower
                    or "https://" in url_lower
                )
            ),

        # =================================================
        # Lexical features
        # =================================================

        "length_words_raw":
            len(raw_words),

        "char_repeat":
            _character_repeat_count(url),

        "shortest_words_raw":
            raw_stats["shortest"],

        "shortest_word_host":
            host_stats["shortest"],

        "shortest_word_path":
            path_stats["shortest"],

        "longest_words_raw":
            raw_stats["longest"],

        "longest_word_host":
            host_stats["longest"],

        "longest_word_path":
            path_stats["longest"],

        "avg_words_raw":
            raw_stats["average"],

        "avg_word_host":
            host_stats["average"],

        "avg_word_path":
            path_stats["average"],

        # =================================================
        # Phishing hints
        # =================================================

        "phish_hints":
            phish_hints,

        # =================================================
        # Brand-related features
        # =================================================

        "domain_in_brand":
            domain_in_brand,

        "brand_in_subdomain":
            brand_in_subdomain,

        "brand_in_path":
            brand_in_path,

        # =================================================
        # TLD
        # =================================================

        "suspecious_tld":
            int(
                tld in SUSPICIOUS_TLDS
            ),
    }

    return features


# =========================================================
# HUMAN-READABLE URL EXPLANATION
# =========================================================

def explain_url_features(url):
    """
    Generate human-readable explanations from the
    real-time URL features.

    IMPORTANT:
    This function does NOT modify the ML model.

    It only converts URL characteristics into
    security-oriented indicators.
    """

    features = extract_url_features(
        url
    )

    indicators = []

    # =====================================================
    # IP ADDRESS
    # =====================================================

    if features.get("ip", 0) == 1:

        indicators.append(
            "URL uses an IP address instead of a domain name."
        )

    # =====================================================
    # URL LENGTH
    # =====================================================

    if features.get("length_url", 0) >= 100:

        indicators.append(
            "URL is unusually long."
        )

    elif features.get("length_url", 0) >= 75:

        indicators.append(
            "URL is relatively long."
        )

    # =====================================================
    # HOSTNAME LENGTH
    # =====================================================

    if features.get("length_hostname", 0) >= 40:

        indicators.append(
            "Hostname is unusually long."
        )

    # =====================================================
    # SUBDOMAINS
    # =====================================================

    if features.get("nb_subdomains", 0) >= 3:

        indicators.append(
            "URL contains multiple subdomains."
        )

    elif features.get("nb_subdomains", 0) >= 2:

        indicators.append(
            "URL contains several subdomains."
        )

    # =====================================================
    # HYPHENS
    # =====================================================

    if features.get("nb_hyphens", 0) >= 3:

        indicators.append(
            "Hostname contains multiple hyphens."
        )

    # =====================================================
    # @ SYMBOL
    # =====================================================

    if features.get("nb_at", 0) > 0:

        indicators.append(
            "URL contains an @ symbol."
        )

    # =====================================================
    # QUERY PARAMETERS
    # =====================================================

    if features.get("nb_qm", 0) > 0:

        indicators.append(
            "URL contains query parameters."
        )

    # =====================================================
    # PARAMETER ASSIGNMENT
    # =====================================================

    if features.get("nb_eq", 0) >= 2:

        indicators.append(
            "URL contains multiple parameter assignments."
        )

    elif features.get("nb_eq", 0) == 1:

        indicators.append(
            "URL contains parameter assignment characters."
        )

    # =====================================================
    # FULL URL DIGIT RATIO
    # =====================================================

    if features.get(
        "ratio_digits_url",
        0,
    ) >= 0.25:

        indicators.append(
            "URL contains a relatively high proportion of digits."
        )

    # =====================================================
    # HOST DIGIT RATIO
    # =====================================================

    if features.get(
        "ratio_digits_host",
        0,
    ) >= 0.50:

        indicators.append(
            "Hostname contains a relatively high proportion of digits."
        )

    # =====================================================
    # PHISHING HINTS
    # =====================================================

    if features.get(
        "phish_hints",
        0,
    ) >= 3:

        indicators.append(
            "URL contains multiple phishing-related keywords or hints."
        )

    elif features.get(
        "phish_hints",
        0,
    ) > 0:

        indicators.append(
            "URL contains phishing-related keywords or hints."
        )

    # =====================================================
    # PUNYCODE
    # =====================================================

    if features.get(
        "punycode",
        0,
    ) == 1:

        indicators.append(
            "URL uses punycode in the domain."
        )

    # =====================================================
    # NON-STANDARD PORT
    # =====================================================

    if features.get(
        "port",
        0,
    ) == 1:

        indicators.append(
            "URL specifies a non-standard port."
        )

    # =====================================================
    # PREFIX / SUFFIX
    # =====================================================

    if features.get(
        "prefix_suffix",
        0,
    ) == 1:

        indicators.append(
            "Domain contains a hyphenated prefix or suffix pattern."
        )

    # =====================================================
    # URL SHORTENER
    # =====================================================

    if features.get(
        "shortening_service",
        0,
    ) == 1:

        indicators.append(
            "URL appears to use a URL shortening service."
        )

    # =====================================================
    # RANDOM DOMAIN
    # =====================================================

    if features.get(
        "random_domain",
        0,
    ) == 1:

        indicators.append(
            "Domain appears to contain a random-looking structure."
        )

    # =====================================================
    # DEEP PATH
    # =====================================================

    if features.get(
        "nb_slash",
        0,
    ) >= 6:

        indicators.append(
            "URL contains a deep path structure."
        )

    # =====================================================
    # PATH EXTENSION
    # =====================================================

    if features.get(
        "path_extension",
        0,
    ) == 1:

        indicators.append(
            "URL contains a file-like path extension."
        )

    # =====================================================
    # HTTP TOKEN INSIDE PATH
    # =====================================================

    if features.get(
        "http_in_path",
        0,
    ) == 1:

        indicators.append(
            "URL contains an HTTP or HTTPS token inside its path."
        )

    # =====================================================
    # REDIRECTION
    # =====================================================

    if features.get(
        "nb_redirection",
        0,
    ) >= 2:

        indicators.append(
            "URL contains multiple redirection indicators."
        )

    # =====================================================
    # EXTERNAL REDIRECTION
    # =====================================================

    if features.get(
        "nb_external_redirection",
        0,
    ) > 0:

        indicators.append(
            "URL contains an external redirection indicator."
        )

    # =====================================================
    # SPECIAL CHARACTERS
    # =====================================================

    special_character_count = (
        features.get("nb_at", 0)
        + features.get("nb_qm", 0)
        + features.get("nb_and", 0)
        + features.get("nb_eq", 0)
        + features.get("nb_percent", 0)
        + features.get("nb_underscore", 0)
        + features.get("nb_tilde", 0)
    )

    if special_character_count >= 6:

        indicators.append(
            "URL contains a high number of special characters."
        )

    # =====================================================
    # CHARACTER REPETITION
    # =====================================================
    #
    # IMPORTANT:
    # Small char_repeat values are normal.
    #
    # We only explain unusually high repetition.
    # =====================================================

    if features.get(
        "char_repeat",
        0,
    ) >= 8:

        indicators.append(
            "URL contains unusually high character repetition."
        )

    # =====================================================
    # MANY LEXICAL COMPONENTS
    # =====================================================

    if features.get(
        "length_words_raw",
        0,
    ) >= 14:

        indicators.append(
            "URL contains many lexical components."
        )

    # =====================================================
    # SUSPICIOUS TLD
    # =====================================================

    if features.get(
        "suspecious_tld",
        0,
    ) == 1:

        indicators.append(
            "URL uses a potentially suspicious top-level domain."
        )

    # =====================================================
    # RETURN
    # =====================================================

    return {
        "features": features,
        "indicators": indicators,
    }


# =========================================================
# LOCAL TEST
# =========================================================

if __name__ == "__main__":

    test_urls = [

        # -------------------------------------------------
        # Normal
        # -------------------------------------------------

        "https://www.google.com",

        "https://github.com/python/cpython/issues",

        # -------------------------------------------------
        # IP based
        # -------------------------------------------------

        "http://192.168.1.25/login",

        # -------------------------------------------------
        # Suspicious domain
        # -------------------------------------------------

        "http://secure-login.example.com/account/verify",

        # -------------------------------------------------
        # Suspicious URL with parameters
        # -------------------------------------------------

        (
            "http://secure-login.example.com/"
            "account/verify/update?"
            "session=983472983472&"
            "token=928374928374"
        ),
    ]

    for test_url in test_urls:

        print("\n" + "=" * 70)

        print(
            "URL:"
        )

        print(
            test_url
        )

        print(
            "=" * 70
        )

        result = explain_url_features(
            test_url
        )

        print(
            "\nFeatures:"
        )

        for name, value in result[
            "features"
        ].items():

            print(
                f"{name}: {value}"
            )

        print(
            "\nDetection indicators:"
        )

        if result[
            "indicators"
        ]:

            for indicator in result[
                "indicators"
            ]:

                print(
                    f"- {indicator}"
                )

        else:

            print(
                "No suspicious URL indicators."
            )
