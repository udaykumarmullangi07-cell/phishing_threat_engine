import re
import tldextract
from urllib.parse import urlparse


# =========================================================
# EXPERIMENTAL URL FEATURE PIPELINE
# =========================================================
#
# This module is ONLY for the experimental
# Lexical + Reputation URL model.
#
# Production 10-feature URL pipeline is NOT modified.
#
# Experimental model:
#   52 lexical / structural features
#   + 7 reputation / domain features
#   = 59 features
#
# IMPORTANT:
# Reputation features are NEVER fabricated.
# They must come from a real enrichment source.
# =========================================================


# =========================================================
# 52 LEXICAL / STRUCTURAL FEATURES
# =========================================================

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


# =========================================================
# 7 REPUTATION / DOMAIN FEATURES
# =========================================================

REPUTATION_FEATURES = [
    "whois_registered_domain",
    "domain_registration_length",
    "domain_age",
    "web_traffic",
    "dns_record",
    "google_index",
    "page_rank",
]


# =========================================================
# COMPLETE 59-FEATURE ORDER
# =========================================================

EXPERIMENTAL_FEATURES = (
    LEXICAL_FEATURES
    + REPUTATION_FEATURES
)


# Safety check
if len(LEXICAL_FEATURES) != 52:
    raise RuntimeError(
        "Lexical feature definition must contain exactly 52 features."
    )

if len(REPUTATION_FEATURES) != 7:
    raise RuntimeError(
        "Reputation feature definition must contain exactly 7 features."
    )

if len(EXPERIMENTAL_FEATURES) != 59:
    raise RuntimeError(
        "Experimental model must contain exactly 59 features."
    )


# =========================================================
# HELPERS
# =========================================================

def _safe_words(value):
    """Split a URL component into lexical words."""

    if not value:
        return []

    return [
        word
        for word in re.split(
            r"[^A-Za-z0-9]+",
            value,
        )
        if word
    ]


def _longest_word(words):
    if not words:
        return 0

    return max(
        len(word)
        for word in words
    )


def _shortest_word(words):
    if not words:
        return 0

    return min(
        len(word)
        for word in words
    )


def _average_word_length(words):
    if not words:
        return 0.0

    return (
        sum(len(word) for word in words)
        / len(words)
    )


def _count_repeated_characters(url):
    """Count adjacent repeated characters."""

    if not url:
        return 0

    count = 0

    for index in range(1, len(url)):

        if url[index] == url[index - 1]:
            count += 1

    return count


def _has_ip(url):
    """Detect IPv4-style addresses."""

    return int(
        bool(
            re.search(
                r"(?<!\d)"
                r"(?:\d{1,3}\.){3}"
                r"\d{1,3}"
                r"(?!\d)",
                url,
            )
        )
    )


def _contains_shortening_service(hostname):
    """Detect common URL-shortening domains."""

    shortening_domains = {
        "bit.ly",
        "tinyurl.com",
        "t.co",
        "goo.gl",
        "ow.ly",
        "is.gd",
        "buff.ly",
        "rebrand.ly",
        "cutt.ly",
        "shorturl.at",
    }

    return int(
        hostname.lower()
        in shortening_domains
    )


def _suspicious_tld(hostname):
    """
    Lightweight lexical suspicious-TLD heuristic.

    This is NOT an external reputation lookup.
    """

    suspicious_tlds = {
        "zip",
        "top",
        "xyz",
        "click",
        "link",
        "work",
        "live",
        "loan",
        "gq",
        "tk",
        "ml",
        "cf",
    }

    parts = hostname.lower().split(".")

    if len(parts) < 2:
        return 0

    return int(
        parts[-1] in suspicious_tlds
    )


def _phish_hint_count(url):
    """Count common phishing-related lexical hints."""

    hints = [
        "login",
        "signin",
        "verify",
        "verification",
        "secure",
        "security",
        "account",
        "password",
        "credential",
        "update",
        "confirm",
        "confirmation",
        "bank",
        "payment",
        "wallet",
        "recover",
        "reset",
    ]

    lower_url = url.lower()

    return sum(
        lower_url.count(hint)
        for hint in hints
    )


def _random_domain_indicator(domain):
    """
    Lightweight heuristic for domains containing
    a long alphabetic component followed by digits.
    """

    return int(
        bool(
            re.search(
                r"[a-zA-Z]{5,}[0-9]{2,}",
                domain,
            )
        )
    )


# =========================================================
# LEXICAL FEATURE EXTRACTION
# =========================================================

def extract_lexical_features(url):
    """
    Extract exactly the 52 locally computable
    lexical / structural features.

    No external data is used.
    """

    url = str(url).strip()

    parsed = urlparse(url)

    hostname = (
        parsed.hostname or ""
    ).lower()

    path = (
        parsed.path or ""
    )

    query = (
        parsed.query or ""
    )

    ext = tldextract.extract(url)

    subdomain = (
        ext.subdomain or ""
    )

    domain = (
        ext.domain or ""
    )

    suffix = (
        ext.suffix or ""
    )

    host_words = _safe_words(
        hostname
    )

    path_words = _safe_words(
        path
    )

    raw_words = _safe_words(
        url
    )

    digits_url = sum(
        character.isdigit()
        for character in url
    )

    digits_host = sum(
        character.isdigit()
        for character in hostname
    )

    path_extension = int(
        bool(
            re.search(
                r"\.[A-Za-z0-9]{1,8}$",
                path,
            )
        )
    )

    subdomain_parts = [
        part
        for part in subdomain.split(".")
        if part
    ]

    # -----------------------------------------------------
    # Brand-related features
    # -----------------------------------------------------
    #
    # These features were present in the training dataset.
    #
    # Without a defined brand database, we cannot reproduce
    # them reliably from the URL alone.
    #
    # Therefore they are explicitly marked as unavailable.
    # -----------------------------------------------------

    features = {

        "length_url":
            len(url),

        "length_hostname":
            len(hostname),

        "ip":
            _has_ip(url),

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

        "nb_www":
            url.lower().count("www"),

        "nb_com":
            url.lower().count(".com"),

        "nb_dslash":
            (
                url.count("//") - 1
                if url.count("//") > 0
                else 0
            ),

        "http_in_path":
            int(
                "http" in path.lower()
            ),

        "https_token":
            int(
                "https" in url.lower()
            ),

        "ratio_digits_url":
            digits_url / max(
                len(url),
                1,
            ),

        "ratio_digits_host":
            digits_host / max(
                len(hostname),
                1,
            ),

        "punycode":
            int(
                "xn--" in hostname.lower()
            ),

        "port":
            int(
                parsed.port is not None
            ),

        "tld_in_path":
            int(
                bool(
                    suffix
                    and suffix.lower()
                    in path.lower()
                )
            ),

        "tld_in_subdomain":
            int(
                bool(
                    suffix
                    and suffix.lower()
                    in subdomain.lower()
                )
            ),

        "abnormal_subdomain":
            int(
                len(subdomain_parts) >= 3
            ),

        "nb_subdomains":
            len(subdomain_parts),

        "prefix_suffix":
            int("-" in domain),

        "random_domain":
            _random_domain_indicator(
                domain
            ),

        "shortening_service":
            _contains_shortening_service(
                hostname
            ),

        "path_extension":
            path_extension,

        "length_words_raw":
            len(raw_words),

        "char_repeat":
            _count_repeated_characters(
                url
            ),

        "shortest_words_raw":
            _shortest_word(
                raw_words
            ),

        "shortest_word_host":
            _shortest_word(
                host_words
            ),

        "shortest_word_path":
            _shortest_word(
                path_words
            ),

        "longest_words_raw":
            _longest_word(
                raw_words
            ),

        "longest_word_host":
            _longest_word(
                host_words
            ),

        "longest_word_path":
            _longest_word(
                path_words
            ),

        "avg_words_raw":
            _average_word_length(
                raw_words
            ),

        "avg_word_host":
            _average_word_length(
                host_words
            ),

        "avg_word_path":
            _average_word_length(
                path_words
            ),

        "phish_hints":
            _phish_hint_count(
                url
            ),

        # Cannot be reliably reproduced without
        # the original brand/reference database.
        "domain_in_brand":
            None,

        "brand_in_subdomain":
            None,

        "brand_in_path":
            None,

        "suspecious_tld":
            _suspicious_tld(
                hostname
            ),
    }

    return features


# =========================================================
# REPUTATION ENRICHMENT
# =========================================================

def validate_reputation_features(
    reputation_features
):
    """
    Validate the seven external reputation/domain features.

    All seven must be supplied.
    """

    if reputation_features is None:

        raise ValueError(
            "Reputation enrichment is required "
            "for the experimental 59-feature model."
        )

    missing = [
        feature
        for feature in REPUTATION_FEATURES
        if feature not in reputation_features
    ]

    if missing:

        raise ValueError(
            "Missing reputation features: "
            + ", ".join(missing)
        )

    unavailable = [
        feature
        for feature in REPUTATION_FEATURES
        if reputation_features[feature] is None
    ]

    if unavailable:

        raise ValueError(
            "Reputation features cannot be None: "
            + ", ".join(unavailable)
        )


# =========================================================
# BUILD COMPLETE 59-FEATURE VECTOR
# =========================================================

def build_experimental_features(
    url,
    reputation_features,
):
    """
    Build the exact 59-feature representation expected
    by the experimental Lexical + Reputation model.

    Reputation data must come from a real enrichment source.
    """

    lexical = extract_lexical_features(
        url
    )

    validate_reputation_features(
        reputation_features
    )

    features = {}

    for feature in LEXICAL_FEATURES:

        value = lexical[feature]

        if value is None:

            raise ValueError(
                f"Lexical feature '{feature}' "
                "cannot be reproduced with the "
                "current extractor."
            )

        features[feature] = value

    for feature in REPUTATION_FEATURES:

        features[feature] = (
            reputation_features[feature]
        )

    # Final schema validation
    missing = [
        feature
        for feature in EXPERIMENTAL_FEATURES
        if feature not in features
    ]

    if missing:

        raise ValueError(
            "Incomplete experimental feature vector: "
            + ", ".join(missing)
        )

    return {
        feature: features[feature]
        for feature in EXPERIMENTAL_FEATURES
    }


# =========================================================
# CHECK WHETHER URL-ONLY EXTRACTION IS POSSIBLE
# =========================================================

def inspect_experimental_url(url):
    """
    Show which experimental features can and cannot
    currently be obtained from the URL itself.
    """

    lexical = extract_lexical_features(
        url
    )

    available = []
    unavailable = []

    for feature in LEXICAL_FEATURES:

        if lexical[feature] is None:
            unavailable.append(feature)
        else:
            available.append(feature)

    return {
        "url": url,
        "lexical_available": available,
        "lexical_unavailable": unavailable,
        "reputation_required":
            REPUTATION_FEATURES,
    }


# =========================================================
# TEST
# =========================================================

if __name__ == "__main__":

    test_url = (
        "http://secure-login.example.com/"
        "account/verify/update"
        "?session=983472983472"
        "&token=928374928374"
    )

    print("=" * 70)
    print("EXPERIMENTAL URL FEATURE PIPELINE TEST")
    print("=" * 70)

    print("\nURL:")
    print(test_url)

    # -----------------------------------------------------
    # Inspect local extraction
    # -----------------------------------------------------

    inspection = inspect_experimental_url(
        test_url
    )

    print("\nLocally available features:")
    print(
        len(
            inspection["lexical_available"]
        )
    )

    print("\nLocally unavailable lexical features:")

    for feature in inspection[
        "lexical_unavailable"
    ]:

        print(
            f"- {feature}"
        )

    print("\nExternal reputation features required:")

    for feature in inspection[
        "reputation_required"
    ]:

        print(
            f"- {feature}"
        )

    # -----------------------------------------------------
    # Demonstrate that missing reputation data
    # is rejected rather than fabricated.
    # -----------------------------------------------------

    print(
        "\nAttempting 59-feature construction "
        "without reputation data..."
    )

    try:

        build_experimental_features(
            test_url,
            None,
        )

    except ValueError as error:

        print(
            "Rejected correctly:"
        )

        print(
            error
        )

    print("\n" + "=" * 70)
    print("EXPERIMENTAL PIPELINE TEST COMPLETE")
    print("=" * 70)
