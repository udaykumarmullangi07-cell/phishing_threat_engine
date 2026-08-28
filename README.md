# Phishing Threat Intelligence Engine

## 1. Project Overview

The Phishing Threat Intelligence Engine is an AI/ML-based cybersecurity application designed to analyze suspicious messages and URLs and estimate their phishing risk.

The system supports three analysis modes:

1. Message-only analysis
2. URL-only analysis
3. Combined message and URL analysis

The application provides a final threat level and a structured threat intelligence report.

The system is implemented as a local machine-learning application with a Streamlit-based user interface.

---

## 2. Main Features

- Phishing detection from email/message text
- Real-time URL phishing analysis
- TF-IDF based text representation
- Logistic Regression text classification
- Random Forest URL classification
- Real-time lexical URL feature extraction
- Risk score calculation
- Message and URL risk fusion
- LOW / MEDIUM / HIGH threat classification
- Human-readable detection explanations
- Recommended security actions
- Structured threat intelligence report
- JSON report export from the application

---

## 3. System Architecture

The main processing pipeline is:

Raw Message
    |
    v
TF-IDF Vectorizer
    |
    v
Text ML Model
    |
    v
Text Probability
    |
    +-----------------------------+
                                  |
Raw URL                           |
    |                             |
    v                             |
Real-Time URL Feature Extractor   |
    |                             |
    v                             |
Top-25 URL Features               |
    |                             |
    v                             |
Random Forest Model               |
    |                             |
    v                             |
URL Probability ------------------+
                                  |
                                  v
                             Risk Fusion
                                  |
                                  v
                         Threat Classification
                                  |
                                  v
                    Threat Intelligence Report

---

## 4. Machine Learning Models

### Text Model

The production text model uses:

- Algorithm: Logistic Regression
- Solver: liblinear
- Regularization parameter: C=30
- Class weighting: balanced
- Maximum iterations: 1000
- Random state: 42

The text representation uses:

- TF-IDF Vectorizer
- Maximum features: 50,000
- N-gram range: 1 to 2
- Minimum document frequency: 2
- Maximum document frequency: 0.95
- Sublinear TF: enabled

Production files:

models/final_text_model.joblib
models/final_text_vectorizer.joblib

---

### URL Model

The real-time URL model uses:

- Algorithm: Random Forest Classifier
- Feature count: 25
- Decision threshold: 0.45
- External reputation services: disabled
- Local lexical URL features

Production model package:

models/realtime_url_top25_model.joblib

The model package contains:

- model
- features
- threshold
- random_state
- model_type
- feature_count
- external_reputation

---

## 5. Risk Fusion

When both a message and URL are supplied, the system combines their probabilities.

Message + URL:

Text probability = 60%
URL probability = 40%

Therefore:

Final Risk Score =
(Text Probability × 0.60) +
(URL Probability × 0.40)

When only a message is supplied:

Text probability = 100%

When only a URL is supplied:

URL probability = 100%

The resulting fused score is then used for threat classification.

---

## 6. Threat Levels

The system classifies the final risk into:

- LOW
- MEDIUM
- HIGH

The application also generates a human-readable explanation and recommended security actions based on the detected threat level.

---

## 7. Application Structure

phishing_threat_engine/
|
├── app.py
├── README.md
├── requirements.txt
|
├── data/
|   ├── cleaned_text.csv
|   ├── phishing_emails.csv
|   ├── url_dataset.csv
|   └── url_features.csv
|
├── models/
|   ├── final_text_model.joblib
|   ├── final_text_vectorizer.joblib
|   ├── realtime_url_top25_model.joblib
|   └── other experimental models
|
├── src/
|   ├── prediction_engine.py
|   ├── realtime_url_features.py
|   ├── realtime_url_predictor.py
|   ├── risk_fusion.py
|   ├── threat_report.py
|   ├── nlp_preprocessing.py
|   └── other analysis/training modules
|
├── reports/
|   ├── model evaluation results
|   ├── dataset audit results
|   ├── URL validation results
|   └── experimental results
|
└── presentation/
    └── screenshots/

---

## 8. Software Requirements

Recommended environment:

- Linux / Ubuntu / Linux Mint
- Python 3.12+
- pip
- Python virtual environment
- Modern web browser

The project was developed and tested using:

Python 3.12.3

---

## 9. Installation

Open a terminal and navigate to the project directory:

cd phishing_threat_engine

Create a Python virtual environment:

python3 -m venv .venv

Activate the virtual environment:

source .venv/bin/activate

Upgrade pip:

python -m pip install --upgrade pip

Install the required Python packages:

pip install -r requirements.txt

---

## 10. Running the Application

Activate the virtual environment:

source .venv/bin/activate

Start the Streamlit application:

streamlit run app.py

After starting the application, open the URL displayed by Streamlit in a web browser.

The default local address is normally:

http://localhost:8501

---

## 11. Using the Application

### Message Analysis

1. Open the application.
2. Enter an email, SMS, or suspicious message in the Message Analysis field.
3. Leave the URL field empty if URL analysis is not required.
4. Click Analyze Threat.
5. Review the message probability, risk score, threat level, indicators and recommendations.

---

### URL Analysis

1. Enter a URL in the URL Analysis field.
2. Leave the message field empty.
3. Click Analyze Threat.
4. The application extracts real-time lexical URL features.
5. The URL model calculates the phishing probability.
6. The system displays the final URL risk and threat level.

---

### Combined Analysis

1. Enter the suspicious message.
2. Enter the associated URL.
3. Click Analyze Threat.
4. The system calculates both message and URL probabilities.
5. The probabilities are combined using the risk-fusion mechanism.
6. The final risk score and threat classification are displayed.

---

## 12. Model Loading

The production prediction engine loads the following files:

models/final_text_model.joblib
models/final_text_vectorizer.joblib
models/realtime_url_top25_model.joblib

The URL model package is validated before prediction.

The application verifies:

- Required model keys
- URL feature list
- Prediction probability support
- Saved URL threshold
- Saved feature configuration

---

## 13. URL Analysis

The URL model uses locally extracted lexical features.

The application does not depend on external reputation services during normal URL analysis.

The current configuration has external reputation services disabled.

The system therefore performs local URL analysis without:

- WHOIS lookup
- DNS reputation
- Google index lookup
- PageRank
- Web traffic lookup
- Webpage fetching
- External brand database lookup

---

## 14. Threat Intelligence Report

The threat report module converts prediction results into a structured security-oriented report.

The report contains information such as:

- Threat level
- Overall risk score
- Message probability
- URL probability
- Detection signals
- Recommended actions
- URL feature information

The application also provides a JSON report export.

---

## 15. Important Source Modules

### app.py

Provides the Streamlit user interface and connects the user inputs to the prediction and threat-report modules.

### src/prediction_engine.py

Loads the production models and performs message, URL and combined threat analysis.

### src/realtime_url_features.py

Extracts lexical characteristics from URLs for real-time analysis.

### src/risk_fusion.py

Combines message and URL probabilities into the final risk score and threat classification.

### src/threat_report.py

Generates human-readable threat summaries, detection signals and recommended security actions.

### src/nlp_preprocessing.py

Contains NLP preprocessing functionality used during dataset/model development.

---

## 16. Important Notes

The application is intended as a phishing risk detection and analysis system.

It should not be treated as a replacement for enterprise security systems, email gateways, endpoint security platforms, or human security review.

The URL analysis is based on locally extracted lexical characteristics and does not perform external reputation lookups in the current configuration.

---

## 17. Troubleshooting

### Streamlit command not found

Make sure the virtual environment is activated:

source .venv/bin/activate

Then install the requirements:

pip install -r requirements.txt

---

### Model loading error

Make sure the following files exist:

models/final_text_model.joblib
models/final_text_vectorizer.joblib
models/realtime_url_top25_model.joblib

Run the application from the project root directory.

---

### Import error

Make sure the terminal is inside:

phishing_threat_engine/

Then run:

streamlit run app.py

---

## 18. Project Execution

The complete runtime flow is:

User Input
    |
    v
Message / URL Validation
    |
    +-------------------+
    |                   |
    v                   v
Text Analysis       URL Analysis
    |                   |
    v                   v
TF-IDF + LR       Lexical Features + RF
    |                   |
    v                   v
Text Probability   URL Probability
    |                   |
    +---------+---------+
              |
              v
         Risk Fusion
              |
              v
        Risk Score
              |
              v
      Threat Classification
              |
              v
    Threat Intelligence Report

---

## 19. Project Status

The project contains:

- Production machine-learning models
- Real-time URL feature extraction
- Risk fusion
- Threat classification
- Streamlit application
- Model experimentation and validation reports
- Application screenshots
- Project documentation
