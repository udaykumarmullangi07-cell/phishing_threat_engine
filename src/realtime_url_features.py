import re
from urllib.parse import urlparse

import tldextract


# =========================================================
# REAL-TIME URL FEATURE EXTRACTOR
# =========================================================
#
# Purpose:
# Extract URL-local features directly from a raw URL.
#
# No:
# - WHOIS
# - DNS reputation
# - Google index
# - PageRank
# - Web traffic
# - Webpage fetching
# - Brand database
#
# is required.
#
# IMPORTANT:
# The feature extraction logic below must remain compatible
# with the trained realtime_url_top25_model.joblib model.
#
# Human-readable indicators are generated separately and
# must NOT modify the ML feature values.
# =========================================================


# =========================================================
# COMPLETE REAL-TIME FEATURE SCHEMA
# =========================================================

REALTIME_FEATURES = [

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
    "nb_redirection",
    "nb_external_redirection",
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
# VALIDATED TOP-25 FEATURES
# =========================================================
#
# These are the exact features selected by the permutation
# validation experiment.
#
# DO NOT CHANGE THIS ORDER.
# =========================================================

TOP25_FEATURES = [

    "nb_www",
    "phish_hints",
    "nb_slash",
    "nb_hyphens",
    "nb_subdomains",
    "ratio_digits_host",
    "char_repeat",
    "ratio_digits_url",
    "longest_words_raw",
    "length_hostname",
    "nb_underscore",
    "https_token",
    "nb_dots",
    "longest_word_host",
    "avg_word_host",
    "nb_com",
    "shortest_word_host",
    "length_words_raw",
    "path_extension",
    "avg_word_path",
    "prefix_suffix",
    "longest_word_path",
    "shortest_words_raw",
    "nb_qm",
    "nb_eq",
]


# =========================================================
# HELPERS
# =========================================================


def _safe_words(value):

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


def _shortest_word(words):

    if not words:
        return 0

    return min(
        len(word)
        for word in words
    )


def _longest_word(words):

    if not words:
        return 0

    return max(
        len(word)
        for word in words
    )


def _average_word_length(words):

    if not words:
        return 0.0

    return (
        sum(
            len(word)
            for word in words
        )
        / len(words)
    )


def _has_ip(url):

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


def _random_domain(domain):

    return int(
        bool(
            re.search(
                r"[A-Za-z]{5,}[0-9]{2,}",
                domain,
            )
        )
    )


def _shortening_service(hostname):

    services = {

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
        in services
    )


def _suspicious_tld(hostname):

    suspicious = {

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

    parts = (
        hostname.lower()
        .split(".")
    )

    if len(parts) < 2:

        return 0

    return int(
        parts[-1]
        in suspicious
    )


def _phish_hints(url):

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

    lower = url.lower()

    return sum(
        lower.count(hint)
        for hint in hints
    )


def _char_repeat(url):

    count = 0

    for index in range(
        1,
        len(url),
    ):

        if (
            url[index]
            == url[index - 1]
        ):

            count += 1

    return count


# =========================================================
# MAIN FEATURE EXTRACTION
# =========================================================


def extract_realtime_url_features(url):

    url = str(url).strip()

    if not url:

        raise ValueError(
            "URL cannot be empty."
        )

    parsed = urlparse(url)

    hostname = (
        parsed.hostname
        or ""
    ).lower()

    path = (
        parsed.path
        or ""
    )

    query = (
        parsed.query
        or ""
    )

    extracted = tldextract.extract(
        url
    )

    subdomain = (
        extracted.subdomain
        or ""
    )

    domain = (
        extracted.domain
        or ""
    )

    suffix = (
        extracted.suffix
        or ""
    )

    # -----------------------------------------------------
    # Word groups
    # -----------------------------------------------------

    host_words = _safe_words(
        hostname
    )

    path_words = _safe_words(
        path
    )

    raw_words = _safe_words(
        url
    )

    # -----------------------------------------------------
    # Digit statistics
    # -----------------------------------------------------

    digits_url = sum(
        character.isdigit()
        for character in url
    )

    digits_host = sum(
        character.isdigit()
        for character in hostname
    )

    # -----------------------------------------------------
    # Subdomain statistics
    # -----------------------------------------------------

    subdomain_parts = [

        part

        for part in
        subdomain.split(".")

        if part

    ]

    # =====================================================
    # FEATURE EXTRACTION
    # =====================================================

    features = {

        # -------------------------------------------------
        # Basic URL statistics
        # -------------------------------------------------

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
            max(
                url.count("//") - 1,
                0,
            ),

        # -------------------------------------------------
        # Protocol / path
        # -------------------------------------------------

        "http_in_path":
            int(
                "http"
                in path.lower()
            ),

        # IMPORTANT:
        #
        # Project definition:
        #
        # HTTP  = 1
        # HTTPS = 0
        #
        # This MUST remain unchanged because the trained
        # model expects this encoding.
        #

        "https_token":
            int(
                parsed.scheme.lower()
                == "http"
            ),

        # -------------------------------------------------
        # Digit ratios
        # -------------------------------------------------

        "ratio_digits_url":
            digits_url
            / max(
                len(url),
                1,
            ),

        "ratio_digits_host":
            digits_host
            / max(
                len(hostname),
                1,
            ),

        # -------------------------------------------------
        # Domain characteristics
        # -------------------------------------------------

        "punycode":
            int(
                "xn--"
                in hostname.lower()
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
                len(subdomain_parts)
                >= 3
            ),

        "nb_subdomains":
            len(subdomain_parts),

        "prefix_suffix":
            int(
                "-"
                in domain
            ),

        "random_domain":
            _random_domain(
                domain
            ),

        "shortening_service":
            _shortening_service(
                hostname
            ),

        # -------------------------------------------------
        # Path extension
        # -------------------------------------------------

        "path_extension":
            int(
                bool(
                    re.search(
                        r"\.[A-Za-z0-9]{1,8}$",
                        path,
                    )
                )
            ),

        # -------------------------------------------------
        # Local redirect indicators
        # -------------------------------------------------

        "nb_redirection":
            (
                url.lower().count(
                    "redirect"
                )
                +
                url.lower().count(
                    "redir"
                )
            ),

        "nb_external_redirection":
            int(
                bool(
                    re.search(
                        r"https?://",
                        query.lower(),
                    )
                )
            ),

        # -------------------------------------------------
        # Raw lexical statistics
        # -------------------------------------------------

        "length_words_raw":
            len(raw_words),

        # IMPORTANT:
        #
        # char_repeat remains part of the ML feature
        # schema and Top-25 model.
        #
        # It is NOT used as a standalone human warning.
        #

        "char_repeat":
            _char_repeat(url),

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

        # -------------------------------------------------
        # Phishing hints
        # -------------------------------------------------

        "phish_hints":
            _phish_hints(url),

        # -------------------------------------------------
        # Brand features
        # -------------------------------------------------
        #
        # No brand database is currently used.
        #
        # Therefore these remain None.
        #

        "domain_in_brand":
            None,

        "brand_in_subdomain":
            None,

        "brand_in_path":
            None,

        # -------------------------------------------------
        # Suspicious TLD
        # -------------------------------------------------

        "suspecious_tld":
            _suspicious_tld(
                hostname
            ),
    }

    # =====================================================
    # SCHEMA VALIDATION
    # =====================================================

    missing = [

        feature

        for feature
        in REALTIME_FEATURES

        if feature
        not in features

    ]

    if missing:

        raise ValueError(
            "Extractor missing features: "
            + ", ".join(missing)
        )

    # -----------------------------------------------------
    # Return exact feature schema
    # -----------------------------------------------------

    return {

        feature:
        features[feature]

        for feature
        in REALTIME_FEATURES

    }


# =========================================================
# TOP-25 FEATURE EXTRACTION
# =========================================================


def extract_top25_features(url):

    features = (
        extract_realtime_url_features(
            url
        )
    )

    missing = [

        feature

        for feature
        in TOP25_FEATURES

        if feature
        not in features

    ]

    if missing:

        raise ValueError(
            "Top-25 feature extraction failed. "
            "Missing: "
            + ", ".join(missing)
        )

    return {

        feature:
        features[feature]

        for feature
        in TOP25_FEATURES

    }


# =========================================================
# HUMAN-READABLE URL INDICATORS
# =========================================================
#
# IMPORTANT:
#
# This function is completely separate from ML feature
# extraction.
#
# It explains suspicious characteristics to the user.
#
# We intentionally DO NOT report char_repeat here because
# normal URLs can naturally contain repeated characters.
#
# Example:
#
# https://www.google.com
#
# has char_repeat > 0 because of "www".
#
# That is not by itself a phishing indicator.
# =========================================================


def explain_realtime_url_features(url):

    features = (
        extract_realtime_url_features(
            url
        )
    )

    indicators = []

    # =====================================================
    # URL LENGTH
    # =====================================================

    if features["length_url"] >= 75:

        indicators.append(
            "URL is unusually long."
        )

    elif features["length_url"] >= 50:

        indicators.append(
            "URL is relatively long."
        )

    # =====================================================
    # IP ADDRESS
    # =====================================================

    if features["ip"] == 1:

        indicators.append(
            "URL uses an IP address instead of a domain name."
        )

    # =====================================================
    # HTTP
    # =====================================================

    if features["https_token"] == 1:

        indicators.append(
            "URL does not use HTTPS."
        )

    # =====================================================
    # SPECIAL CHARACTERS
    # =====================================================

    special_character_count = (

        features["nb_at"]
        + features["nb_qm"]
        + features["nb_and"]
        + features["nb_eq"]
        + features["nb_underscore"]
        + features["nb_percent"]
        + features["nb_star"]
        + features["nb_comma"]
        + features["nb_semicolumn"]
        + features["nb_dollar"]

    )

    if special_character_count >= 6:

        indicators.append(
            "URL contains many special characters."
        )

    elif special_character_count >= 3:

        indicators.append(
            "URL contains multiple special characters."
        )

    # =====================================================
    # AT SYMBOL
    # =====================================================

    if features["nb_at"] > 0:

        indicators.append(
            "URL contains an @ symbol."
        )

    # =====================================================
    # SUBDOMAINS
    # =====================================================

    if features["nb_subdomains"] >= 3:

        indicators.append(
            "URL contains multiple subdomains."
        )

    elif features["nb_subdomains"] >= 2:

        indicators.append(
            "URL contains several subdomains."
        )

    # =====================================================
    # DIGIT RATIO
    # =====================================================

    if features["ratio_digits_url"] >= 0.25:

        indicators.append(
            "URL contains a relatively high proportion of digits."
        )

    elif features["ratio_digits_url"] >= 0.15:

        indicators.append(
            "URL contains a noticeable number of digits."
        )

    # =====================================================
    # HOST DIGIT RATIO
    # =====================================================

    if features["ratio_digits_host"] >= 0.50:

        indicators.append(
            "Hostname contains a relatively high proportion of digits."
        )

    # =====================================================
    # PHISHING HINTS
    # =====================================================

    if features["phish_hints"] >= 4:

        indicators.append(
            "URL contains multiple phishing-related keywords or hints."
        )

    elif features["phish_hints"] >= 1:

        indicators.append(
            "URL contains phishing-related keywords or hints."
        )

    # =====================================================
    # QUERY PARAMETERS
    # =====================================================

    if features["nb_qm"] > 0:

        indicators.append(
            "URL contains query parameters."
        )

    # =====================================================
    # PARAMETER ASSIGNMENT
    # =====================================================

    if features["nb_eq"] >= 2:

        indicators.append(
            "URL contains multiple parameter assignment characters."
        )

    elif features["nb_eq"] == 1:

        indicators.append(
            "URL contains a parameter assignment character."
        )

    # =====================================================
    # DEEP PATH
    # =====================================================

    # nb_slash includes protocol slashes.
    #
    # For human explanation, subtract the normal protocol
    # component.
    #

    path_slashes = max(
        features["nb_slash"] - 2,
        0,
    )

    if path_slashes >= 4:

        indicators.append(
            "URL contains a relatively deep path structure."
        )

    elif path_slashes >= 2:

        indicators.append(
            "URL contains several path levels."
        )

    # =====================================================
    # PUNYCODE
    # =====================================================

    if features["punycode"] == 1:

        indicators.append(
            "URL uses a punycode-encoded hostname."
        )

    # =====================================================
    # NON-STANDARD PORT
    # =====================================================

    if features["port"] == 1:

        indicators.append(
            "URL uses a non-standard port."
        )

    # =====================================================
    # URL SHORTENER
    # =====================================================

    if features["shortening_service"] == 1:

        indicators.append(
            "URL uses a known URL shortening service."
        )

    # =====================================================
    # PREFIX / SUFFIX
    # =====================================================

    if features["prefix_suffix"] == 1:

        indicators.append(
            "Domain contains a hyphenated structure."
        )

    # =====================================================
    # SUSPICIOUS TLD
    # =====================================================

    if features["suspecious_tld"] == 1:

        indicators.append(
            "URL uses a TLD frequently associated with suspicious domains."
        )

    # =====================================================
    # RANDOM DOMAIN
    # =====================================================

    if features["random_domain"] == 1:

        indicators.append(
            "Hostname contains a potentially random alphanumeric pattern."
        )

    # =====================================================
    # EXTERNAL REDIRECTION
    # =====================================================

    if features["nb_external_redirection"] == 1:

        indicators.append(
            "URL contains an embedded external URL."
        )

    # =====================================================
    # FINAL RESULT
    # =====================================================

    return {

        "features":
            features,

        "indicators":
            indicators,

    }


# =========================================================
# BACKWARD-COMPATIBLE ALIAS
# =========================================================
#
# Some project files may already import:
#
# explain_url_features
#
# Keep this alias so existing code does not break.
# =========================================================


def explain_url_features(url):

    return explain_realtime_url_features(
        url
    )


# =========================================================
# TEST
# =========================================================


if __name__ == "__main__":

    test_urls = [

        "https://www.google.com",

        "https://github.com/python/cpython/issues",

        "http://192.168.1.25/login",

        (
            "http://secure-login.example.com/"
            "account/verify"
        ),

        (
            "http://secure-login.example.com/"
            "account/verify/update"
            "?session=983472983472"
            "&token=928374928374"
        ),

    ]

    for url in test_urls:

        print(
            "\n"
            + "=" * 70
        )

        print(
            "URL:"
        )

        print(
            url
        )

        print(
            "=" * 70
        )

        try:

            result = (
                explain_realtime_url_features(
                    url
                )
            )

            print(
                "\nIndicators:"
            )

            if result["indicators"]:

                for indicator in result[
                    "indicators"
                ]:

                    print(
                        "-",
                        indicator
                    )

            else:

                print(
                    "- No suspicious URL indicators detected."
                )

            print(
                "\nTop-25 Features:"
            )

            for feature in TOP25_FEATURES:

                print(
                    f"{feature}: "
                    f"{result['features'][feature]}"
                )

        except Exception as error:

            print(
                "\nERROR:",
                error
            )
