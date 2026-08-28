import json
from datetime import datetime

import streamlit as st

from src.prediction_engine import analyze_threat
from src.threat_report import (
    generate_threat_summary,
    generate_recommended_actions,
    generate_detection_signals,
)


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Phishing Threat Intelligence Engine",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>

    /* -----------------------------------------------------
       Main page
    ----------------------------------------------------- */

    .main-title {
        font-size: 42px;
        font-weight: 800;
        margin-top: 10px;
        margin-bottom: 4px;
        line-height: 1.2;
    }

    .subtitle {
        font-size: 17px;
        opacity: 0.70;
        margin-bottom: 25px;
    }


    /* -----------------------------------------------------
       Metric cards
    ----------------------------------------------------- */

    .metric-card {
        padding: 18px;
        border-radius: 14px;
        border: 1px solid rgba(128, 128, 128, 0.25);
        text-align: center;
        min-height: 125px;
    }

    .metric-label {
        font-size: 14px;
        opacity: 0.70;
        margin-bottom: 8px;
    }

    .metric-value {
        font-size: 30px;
        font-weight: 800;
    }


    /* -----------------------------------------------------
       Threat banner
    ----------------------------------------------------- */

    .threat-banner {
        padding: 18px;
        border-radius: 14px;
        margin: 15px 0;
        text-align: center;
        font-size: 25px;
        font-weight: 800;
        border: 1px solid rgba(128, 128, 128, 0.25);
    }


    /* -----------------------------------------------------
       Information cards
    ----------------------------------------------------- */

    .info-card {
        padding: 16px;
        border-radius: 12px;
        border: 1px solid rgba(128, 128, 128, 0.25);
        margin-bottom: 10px;
    }


    /* -----------------------------------------------------
       Small text
    ----------------------------------------------------- */

    .small-text {
        font-size: 13px;
        opacity: 0.65;
    }


    /* -----------------------------------------------------
       Footer
    ----------------------------------------------------- */

    .footer {
        text-align: center;
        opacity: 0.55;
        font-size: 13px;
        padding-top: 30px;
        padding-bottom: 20px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# SESSION STATE
# =========================================================

if "analysis_history" not in st.session_state:
    st.session_state.analysis_history = []


# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<div class="main-title">🛡️ Phishing Threat Intelligence Engine</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">'
    "AI-powered detection and risk assessment for suspicious "
    "messages and URLs"
    "</div>",
    unsafe_allow_html=True,
)


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.header("⚙️ System Information")

    st.markdown(
        """
        **Detection Components**

        🧠 Message ML Model

        🔗 Real-Time URL Model

        🧮 Risk Fusion Engine

        📝 Threat Intelligence Report
        """
    )

    st.divider()

    st.subheader("🔐 URL Analysis")

    st.caption(
        "The URL model uses locally extracted lexical "
        "features directly from the supplied URL."
    )

    st.write("WHOIS: ❌")
    st.write("DNS reputation: ❌")
    st.write("Google index: ❌")
    st.write("PageRank: ❌")
    st.write("Web traffic: ❌")
    st.write("Webpage fetching: ❌")
    st.write("Brand database: ❌")

    st.divider()

    st.subheader("🤖 Model Configuration")

    st.write("URL features: **25**")
    st.write("URL threshold: **0.45**")
    st.write("URL model: **Random Forest**")
    st.write("Architecture: **Local ML**")


# =========================================================
# INPUT SECTION
# =========================================================

st.subheader("🔍 Threat Analysis")

col1, col2 = st.columns(2)


# =========================================================
# MESSAGE INPUT
# =========================================================

with col1:

    st.markdown("### 📧 Message Analysis")

    message = st.text_area(
        "Enter an email or message",
        height=250,
        placeholder=(
            "Paste a suspicious email, SMS, or message here..."
        ),
    )


# =========================================================
# URL INPUT
# =========================================================

with col2:

    st.markdown("### 🔗 URL Analysis")

    url = st.text_input(
        "Enter a URL",
        placeholder="https://example.com",
    )

    st.caption(
        "The URL is analyzed using locally extracted "
        "real-time lexical features."
    )


st.info(
    "💡 You can analyze a message, a URL, or both together."
)


# =========================================================
# BUTTONS
# =========================================================

button_col1, button_col2, button_col3 = st.columns(
    [2, 1, 1]
)


with button_col1:

    analyze_button = st.button(
        "🔎 Analyze Threat",
        type="primary",
        use_container_width=True,
    )


with button_col2:

    clear_button = st.button(
        "🧹 Clear",
        use_container_width=True,
    )


with button_col3:

    history_button = st.button(
        "📜 History",
        use_container_width=True,
    )


# =========================================================
# CLEAR
# =========================================================

if clear_button:

    st.session_state.analysis_history = []

    st.rerun()


# =========================================================
# HISTORY
# =========================================================

if history_button:

    st.subheader("📜 Analysis History")

    if not st.session_state.analysis_history:

        st.info(
            "No analysis history available yet."
        )

    else:

        for index, item in enumerate(
            reversed(
                st.session_state.analysis_history
            ),
            start=1,
        ):

            with st.expander(
                f"{index}. "
                f"{item['threat_level']} — "
                f"{item['risk_score']:.2f}%"
            ):

                st.write(
                    "**Date:**",
                    item.get("timestamp", "Unknown"),
                )

                st.write(
                    "**URL:**",
                    item["url"] or "None",
                )

                st.write(
                    "**Message analyzed:**",
                    "Yes"
                    if item["has_text"]
                    else "No",
                )

                st.write(
                    "**Risk score:**",
                    f"{item['risk_score']:.2f}%",
                )

                st.write(
                    "**Threat level:**",
                    item["threat_level"],
                )


# =========================================================
# ANALYSIS
# =========================================================

if analyze_button:

    text_input = (
        message.strip()
        if message
        else None
    )

    url_input = (
        url.strip()
        if url
        else None
    )


    # =====================================================
    # INPUT VALIDATION
    # =====================================================

    if (
        text_input is None
        and url_input is None
    ):

        st.warning(
            "⚠️ Please enter a message, a URL, or both."
        )

    else:

        try:

            # =================================================
            # RUN UNIFIED THREAT ENGINE
            # =================================================

            with st.spinner(
                "🔍 Analyzing threat..."
            ):

                result = analyze_threat(
                    text=text_input,
                    url=url_input,
                )


            # =================================================
            # SAFELY EXTRACT MODEL RESULTS
            # =================================================

            risk_score = float(
                result.get(
                    "risk_score",
                    0.0,
                )
            )

            threat_level = result.get(
                "threat_level",
                "LOW",
            )

            text_probability = result.get(
                "text_probability"
            )

            url_probability = result.get(
                "url_probability"
            )

            url_indicators = result.get(
                "url_indicators",
                [],
            )

            url_features = result.get(
                "url_features"
            )

            explanation = result.get(
                "explanation"
            )


            # =================================================
            # TIMESTAMP
            # =================================================

            analysis_time = datetime.now().astimezone()

            timestamp = analysis_time.strftime(
                "%Y-%m-%d %H:%M:%S %Z"
            )


            # =================================================
            # STORE HISTORY
            # =================================================

            st.session_state.analysis_history.append(
                {
                    "timestamp": timestamp,
                    "url": url_input,
                    "has_text": (
                        text_input is not None
                    ),
                    "risk_score": (
                        risk_score * 100
                    ),
                    "threat_level": threat_level,
                }
            )


            # Keep only latest 10 analyses

            st.session_state.analysis_history = (
                st.session_state.analysis_history[-10:]
            )


            # =================================================
            # RESULTS HEADER
            # =================================================

            st.divider()

            st.subheader(
                "📊 Threat Analysis Result"
            )


            # =================================================
            # THREAT BANNER
            # =================================================

            if threat_level == "HIGH":

                banner_text = (
                    "🚨 HIGH RISK — PHISHING THREAT DETECTED"
                )

            elif threat_level == "MEDIUM":

                banner_text = (
                    "⚠️ MEDIUM RISK — SUSPICIOUS ACTIVITY"
                )

            else:

                banner_text = (
                    "✅ LOW RISK — NO STRONG PHISHING SIGNAL"
                )


            st.markdown(
                f"""
                <div class="threat-banner">
                    {banner_text}
                </div>
                """,
                unsafe_allow_html=True,
            )


            # =================================================
            # MAIN METRICS
            # =================================================

            metric1, metric2, metric3, metric4 = (
                st.columns(4)
            )


            # -------------------------------------------------
            # Overall Risk
            # -------------------------------------------------

            with metric1:

                st.markdown(
                    """
                    <div class="metric-card">
                    <div class="metric-label">
                    OVERALL RISK SCORE
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                st.markdown(
                    f"""
                    <div class="metric-value">
                    {risk_score * 100:.2f}%
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                st.caption(
                    "Final fused threat score"
                )

                st.markdown(
                    "</div>",
                    unsafe_allow_html=True,
                )


            # -------------------------------------------------
            # Message Model
            # -------------------------------------------------

            with metric2:

                text_display = (
                    f"{float(text_probability) * 100:.2f}%"
                    if text_probability is not None
                    else "N/A"
                )

                st.markdown(
                    """
                    <div class="metric-card">
                    <div class="metric-label">
                    MESSAGE MODEL
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                st.markdown(
                    f"""
                    <div class="metric-value">
                    {text_display}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                st.caption(
                    "Text ML model"
                    if text_probability is not None
                    else "No message provided"
                )

                st.markdown(
                    "</div>",
                    unsafe_allow_html=True,
                )


            # -------------------------------------------------
            # URL Model
            # -------------------------------------------------

            with metric3:

                url_display = (
                    f"{float(url_probability) * 100:.2f}%"
                    if url_probability is not None
                    else "N/A"
                )

                st.markdown(
                    """
                    <div class="metric-card">
                    <div class="metric-label">
                    URL PHISHING PROBABILITY
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                st.markdown(
                    f"""
                    <div class="metric-value">
                    {url_display}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                st.caption(
                    "Real-time URL model"
                    if url_probability is not None
                    else "No URL provided"
                )

                st.markdown(
                    "</div>",
                    unsafe_allow_html=True,
                )


            # -------------------------------------------------
            # Threat Level
            # -------------------------------------------------

            with metric4:

                st.markdown(
                    """
                    <div class="metric-card">
                    <div class="metric-label">
                    THREAT LEVEL
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                st.markdown(
                    f"""
                    <div class="metric-value">
                    {threat_level}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                st.caption(
                    "Risk classification"
                )

                st.markdown(
                    "</div>",
                    unsafe_allow_html=True,
                )


            # =================================================
            # RISK PROGRESS
            # =================================================

            st.markdown("### 📈 Risk Score")

            st.progress(
                min(
                    max(
                        int(risk_score * 100),
                        0,
                    ),
                    100,
                )
            )


            # =================================================
            # THREAT MESSAGE
            # =================================================

            if threat_level == "HIGH":

                st.error(
                    "🚨 HIGH RISK — Strong phishing "
                    "characteristics detected."
                )

            elif threat_level == "MEDIUM":

                st.warning(
                    "⚠️ MEDIUM RISK — Suspicious "
                    "characteristics detected."
                )

            else:

                st.success(
                    "✅ LOW RISK — No strong phishing "
                    "characteristics detected."
                )


            # =================================================
            # MODEL BREAKDOWN
            # =================================================

            st.subheader(
                "🤖 Model Analysis"
            )

            model_col1, model_col2 = st.columns(2)


            # -------------------------------------------------
            # Message Model
            # -------------------------------------------------

            with model_col1:

                st.markdown(
                    "### 📧 Message Model"
                )

                if text_probability is not None:

                    text_probability = float(
                        text_probability
                    )

                    st.write(
                        "Phishing probability:",
                        f"**{text_probability * 100:.2f}%**",
                    )

                    st.progress(
                        min(
                            max(
                                int(
                                    text_probability * 100
                                ),
                                0,
                            ),
                            100,
                        )
                    )

                else:

                    st.info(
                        "Message analysis was not requested."
                    )


            # -------------------------------------------------
            # URL Model
            # -------------------------------------------------

            with model_col2:

                st.markdown(
                    "### 🔗 URL Model"
                )

                if url_probability is not None:

                    url_probability = float(
                        url_probability
                    )

                    st.write(
                        "Phishing probability:",
                        f"**{url_probability * 100:.2f}%**",
                    )

                    st.progress(
                        min(
                            max(
                                int(
                                    url_probability * 100
                                ),
                                0,
                            ),
                            100,
                        )
                    )

                else:

                    st.info(
                        "URL analysis was not requested."
                    )


            # =================================================
            # THREAT INTELLIGENCE SUMMARY
            # =================================================

            st.subheader(
                "📝 Threat Intelligence Summary"
            )

            summary = generate_threat_summary(
                threat_level,
                risk_score,
                text_probability,
                url_probability,
            )

            st.info(summary)


            # =================================================
            # DETECTION SIGNALS
            # =================================================

            st.subheader(
                "🚩 Detection Signals"
            )

            signals = generate_detection_signals(
                text_probability,
                url_probability,
            )

            if signals:

                for signal in signals:

                    st.write(
                        f"**{signal['source']}** — "
                        f"{signal['signal']} "
                        f"({signal['probability']})"
                    )

            else:

                st.write(
                    "No model-level detection signals."
                )


            # =================================================
            # URL SECURITY ANALYSIS
            # =================================================

            if url_input is not None:

                st.divider()

                st.subheader(
                    "🔗 URL Security Analysis"
                )


                # -------------------------------------------------
                # Examined URL
                # -------------------------------------------------

                st.markdown(
                    "### Examined URL"
                )

                st.code(
                    url_input,
                    language="text",
                )


                # -------------------------------------------------
                # URL Indicators
                # -------------------------------------------------

                st.markdown(
                    "### 🚩 URL Indicators"
                )

                if url_indicators:

                    for indicator in url_indicators:

                        st.warning(
                            f"• {indicator}"
                        )

                else:

                    st.success(
                        "No suspicious URL indicators "
                        "were detected."
                    )


                # -------------------------------------------------
                # URL Features
                # -------------------------------------------------

                if url_features:

                    with st.expander(
                        "🧬 Extracted URL Features"
                    ):

                        st.dataframe(
                            {
                                "Feature": list(
                                    url_features.keys()
                                ),
                                "Value": list(
                                    url_features.values()
                                ),
                            },
                            use_container_width=True,
                            hide_index=True,
                        )


            # =================================================
            # RECOMMENDED SECURITY ACTIONS
            # =================================================

            st.subheader(
                "🛡️ Recommended Security Actions"
            )

            actions = generate_recommended_actions(
                threat_level,
                url_indicators,
            )

            if actions:

                for action in actions:

                    st.write(
                        f"• {action}"
                    )

            else:

                st.write(
                    "No additional security actions."
                )


            # =================================================
            # JSON REPORT
            # =================================================

            st.divider()

            st.subheader(
                "📥 Export Analysis"
            )

            # -------------------------------------------------
            # Determine analysis mode
            # -------------------------------------------------

            if (
                text_input is not None
                and url_input is not None
            ):

                analysis_mode = "Message + URL"

            elif text_input is not None:

                analysis_mode = "Message Only"

            else:

                analysis_mode = "URL Only"


            # -------------------------------------------------
            # Create structured JSON report
            # -------------------------------------------------

            report_data = {
                "report_metadata": {
                    "report_type": (
                        "Phishing Threat Intelligence Report"
                    ),
                    "generated_at": timestamp,
                    "analysis_mode": analysis_mode,
                    "engine": (
                        "Phishing Threat Intelligence Engine"
                    ),
                    "architecture": "Local ML",
                },

                "threat_assessment": {
                    "risk_score": risk_score,
                    "risk_percentage": (
                        round(
                            risk_score * 100,
                            2,
                        )
                    ),
                    "threat_level": threat_level,
                },

                "model_analysis": {
                    "text_probability": (
                        float(text_probability)
                        if text_probability is not None
                        else None
                    ),
                    "url_probability": (
                        float(url_probability)
                        if url_probability is not None
                        else None
                    ),
                },

                "input_analysis": {
                    "message_provided": (
                        text_input is not None
                    ),
                    "url_provided": (
                        url_input is not None
                    ),
                    "url": url_input,
                },

                "url_security_analysis": {
                    "indicators": url_indicators,
                    "features": url_features,
                },

                "threat_intelligence": {
                    "summary": summary,
                    "explanation": explanation,
                    "detection_signals": signals,
                },

                "recommended_actions": actions,

                "model_configuration": {
                    "url_feature_count": 25,
                    "url_threshold": 0.45,
                    "url_model": "Random Forest",
                    "external_reputation_services": False,
                    "url_analysis": (
                        "Local lexical features"
                    ),
                },
            }


            # -------------------------------------------------
            # Convert to JSON
            # -------------------------------------------------

            report_json = json.dumps(
                report_data,
                indent=4,
                ensure_ascii=False,
                default=str,
            )


            # -------------------------------------------------
            # Download button
            # -------------------------------------------------

            st.download_button(
                label="⬇️ Download Threat Report (JSON)",
                data=report_json,
                file_name=(
                    "phishing_threat_report.json"
                ),
                mime="application/json",
                use_container_width=True,
            )


            # -------------------------------------------------
            # Preview JSON
            # -------------------------------------------------

            with st.expander(
                "👁️ Preview JSON Report"
            ):

                st.code(
                    report_json,
                    language="json",
                )


            # =================================================
            # TECHNICAL DETAILS
            # =================================================

            with st.expander(
                "⚙️ Technical Analysis Details"
            ):

                st.write(
                    "### Architecture"
                )

                st.code(
                    """
Raw Message
     │
     ▼
TF-IDF Vectorizer
     │
     ▼
Text ML Model
     │
     ▼
Text Probability
     │
     ├─────────────────┐
     │                 │
Raw URL               │
     │                 │
     ▼                 │
Real-Time URL         │
Feature Extractor     │
     │                 │
     ▼                 │
Top-25 URL            │
Random Forest         │
     │                 │
     ▼                 │
URL Probability       │
     │                 │
     └────────┬────────┘
              ▼
         Risk Fusion
              │
              ▼
       Threat Classification
              │
              ▼
     Threat Intelligence Report
                    """,
                    language="text",
                )

                st.write(
                    "### URL Model"
                )

                st.write(
                    "Feature count: **25**"
                )

                st.write(
                    "Decision threshold: **0.45**"
                )

                st.write(
                    "Model: **Random Forest Classifier**"
                )

                st.write(
                    "External reputation services: **Disabled**"
                )

                st.write(
                    "URL analysis: **Local lexical features**"
                )

                st.write(
                    "Production model modified during "
                    "analysis: **No**"
                )

                st.write(
                    "### Risk Fusion"
                )

                st.write(
                    "Message + URL: Text 60% + URL 40%"
                )

                st.write(
                    "Message only: Text 100%"
                )

                st.write(
                    "URL only: URL 100%"
                )


            # =================================================
            # RAW OUTPUT
            # =================================================

            with st.expander(
                "🔧 Developer / Raw Analysis Output"
            ):

                st.json(result)


        # =====================================================
        # ERROR HANDLING
        # =====================================================

        except Exception as error:

            st.error(
                "❌ An error occurred while analyzing "
                "the input."
            )

            with st.expander(
                "Technical error details"
            ):

                st.exception(error)


# =========================================================
# FOOTER
# =========================================================

st.markdown(
    """
    <div class="footer">
        Phishing Threat Intelligence Engine • Local ML Analysis •
        Real-Time URL Feature Detection
    </div>
    """,
    unsafe_allow_html=True,
)
