TEXT_TEST_METRICS = {
    "accuracy": 0.9870,
    "precision": 0.9867,
    "recall": 0.9882,
    "f1": 0.9875,
    "roc_auc": 0.9988,
}


URL_TEST_METRICS = {
    "accuracy": 0.8574,
    "precision": 0.8500,
    "recall": 0.8679,
    "f1": 0.8589,
    "roc_auc": 0.9270,
}


print("=" * 70)
print("PHISHING THREAT ENGINE — MODEL SUMMARY")
print("=" * 70)

print("\nTEXT MODEL — LOGISTIC REGRESSION")

for metric, value in TEXT_TEST_METRICS.items():
    print(
        f"{metric.upper():10s}: {value:.4f}"
    )


print("\nURL MODEL — RANDOM FOREST")

for metric, value in URL_TEST_METRICS.items():
    print(
        f"{metric.upper():10s}: {value:.4f}"
    )


print("\n" + "=" * 70)
print("MODEL SELECTION")
print("=" * 70)

print(
    "Text model: Logistic Regression"
)

print(
    "URL model: Random Forest"
)

print(
    "\nThese metrics come from the held-out "
    "test sets used during model training."
)
