import streamlit as st

from src.prediction_engine import analyze_threat


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Phishing Threat Engine",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# =========================================================
# HEADER
# =========================================================

st.title("🛡️ Phishing Threat Engine")

st.markdown(
    "### AI-Powered Social Engineering Threat Detection & Intelligence Engine"
)

st.write(
    "Analyze suspicious email messages, URLs, or both together."
)

st.divider()


# =========================================================
# INPUT SECTION
# =========================================================

st.subheader("🔍 Threat Analysis")


input_col1, input_col2 = st.columns(
    2,
    gap="large",
)


# =========================================================
# EMAIL INPUT
# =========================================================

with input_col1:

    st.markdown("### 📧 Email / Message")

    email_text = st.text_area(
        "Paste suspicious email/message",
        height=260,
        placeholder=(
            "Paste a suspicious email or message here...\n\n"
            "Example:\n"
            "URGENT: Your account has been suspended.\n"
            "Verify your account immediately."
        ),
    )


# =========================================================
# URL INPUT
# =========================================================

with input_col2:

    st.markdown("### 🔗 URL")

    url_text = st.text_input(
        "Enter suspicious URL",
        placeholder="https://example.com/login",
    )

    st.info(
        "💡 You can analyze an email, a URL, or both together."
    )


# =========================================================
# ANALYZE BUTTON
# =========================================================

st.write("")

analyze_button = st.button(
    "🔍 Analyze Threat",
    type="primary",
    use_container_width=True,
)


# =========================================================
# ANALYSIS
# =========================================================

if analyze_button:

    # -----------------------------------------------------
    # Clean inputs
    # -----------------------------------------------------

    email_input = (
        email_text.strip()
        if email_text
        else None
    )

    url_input = (
        url_text.strip()
        if url_text
        else None
    )

    # -----------------------------------------------------
    # Validate input
    # -----------------------------------------------------

    if (
        email_input is None
        and url_input is None
    ):

        st.error(
            "⚠️ Please enter an email/message or a URL."
        )

    else:

        # -------------------------------------------------
        # Run prediction engine
        # -------------------------------------------------

        with st.spinner(
            "🧠 Analyzing threat..."
        ):

            try:

                result = analyze_threat(
                    text=email_input,
                    url=url_input,
                )

            except Exception as error:

                st.error(
                    f"Analysis failed: {error}"
                )

                st.stop()


        # =================================================
        # THREAT ASSESSMENT
        # =================================================

        st.divider()

        st.subheader(
            "📊 Threat Assessment"
        )


        threat_level = result[
            "threat_level"
        ]

        risk_score = float(
            result["risk_score"]
        )

        risk_percentage = (
            risk_score * 100
        )


        # -------------------------------------------------
        # Threat level
        # -------------------------------------------------

        if threat_level == "HIGH":

            st.error(
                f"🚨 THREAT LEVEL: HIGH\n\n"
                f"Risk Score: {risk_percentage:.2f}%"
            )

        elif threat_level == "MEDIUM":

            st.warning(
                f"⚠️ THREAT LEVEL: MEDIUM\n\n"
                f"Risk Score: {risk_percentage:.2f}%"
            )

        else:

            st.success(
                f"✅ THREAT LEVEL: LOW\n\n"
                f"Risk Score: {risk_percentage:.2f}%"
            )


        # =================================================
        # MODEL SIGNALS
        # =================================================

        st.markdown("### 🧠 Model Signals")


        text_probability = result[
            "text_probability"
        ]

        url_probability = result[
            "url_probability"
        ]


        signal_col1, signal_col2, signal_col3 = st.columns(
            3
        )


        # -------------------------------------------------
        # Text signal
        # -------------------------------------------------

        with signal_col1:

            if text_probability is None:

                st.metric(
                    "📧 Text Model",
                    "N/A",
                )

            else:

                st.metric(
                    "📧 Text Model",
                    f"{text_probability * 100:.2f}%",
                )


        # -------------------------------------------------
        # URL signal
        # -------------------------------------------------

        with signal_col2:

            if url_probability is None:

                st.metric(
                    "🔗 URL Model",
                    "N/A",
                )

            else:

                st.metric(
                    "🔗 URL Model",
                    f"{url_probability * 100:.2f}%",
                )


        # -------------------------------------------------
        # Combined risk
        # -------------------------------------------------

        with signal_col3:

            st.metric(
                "🎯 Combined Risk",
                f"{risk_percentage:.2f}%",
            )


        # =================================================
        # OVERALL RISK
        # =================================================

        st.markdown("### 🎯 Overall Risk")

        st.progress(
            min(
                max(
                    risk_score,
                    0.0,
                ),
                1.0,
            )
        )


        # =================================================
        # ANALYSIS EXPLANATION
        # =================================================

        st.markdown(
            "### 🧠 Analysis Explanation"
        )

        st.info(
            result["explanation"]
        )


        # =================================================
        # URL ANALYSIS
        # =================================================

        if url_input is not None:

            st.markdown(
                "### 🔎 URL Analysis"
            )


            indicators = result[
                "indicators"
            ]


            if indicators:

                st.warning(
                    f"⚠️ {len(indicators)} "
                    "suspicious URL indicator(s) detected."
                )

                for indicator in indicators:

                    st.markdown(
                        f"- ⚠️ {indicator}"
                    )

            else:

                st.success(
                    "✅ No obvious suspicious URL "
                    "indicators detected."
                )


            # -------------------------------------------------
            # URL features
            # -------------------------------------------------

            url_features = result[
                "url_features"
            ]


            if url_features is not None:

                with st.expander(
                    "🔬 View Extracted URL Features"
                ):

                    st.json(
                        url_features
                    )


        # =================================================
        # STRUCTURED RESULT
        # =================================================

        with st.expander(
            "🧪 View Structured Prediction Result"
        ):

            st.json(
                result
            )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "🛡️ Phishing Threat Engine • "
    "Local Machine Learning Prototype"
)
