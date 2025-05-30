import pandas as pd
import os
from transformers import pipeline

# Paths
CLEANED_CSV = os.path.join('..', 'data', 'monitoring_table_cleaned.csv')
CLASSIFIED_CSV = os.path.join('..', 'data', 'monitoring_table_classified.csv')

# Predefined categories and severity labels
CATEGORIES = [
    "Payment Issue",
    "API Error",
    "Network Issue",
    "Timeout",
    "Authentication Failure",
    "Other"
]
SEVERITY_LABELS = ["High", "Medium", "Low"]

# Load cleaned data
print(f"Loading cleaned data from {CLEANED_CSV}")
df = pd.read_csv(CLEANED_CSV)

# Load zero-shot classification pipeline
print("Loading zero-shot-classification pipeline (facebook/bart-large-mnli)...")
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

def classify(row):
    text = row['combined_text']
    # Classify category
    cat_result = classifier(text, CATEGORIES)
    category = cat_result['labels'][0]
    # Classify severity
    sev_result = classifier(text, SEVERITY_LABELS)
    severity = sev_result['labels'][0]
    return pd.Series({'category': category, 'severity': severity})

print("Classifying errors...")
df[['category', 'severity']] = df.apply(classify, axis=1)

# Save classified data
os.makedirs(os.path.dirname(CLASSIFIED_CSV), exist_ok=True)
df.to_csv(CLASSIFIED_CSV, index=False)
print(f"Classified data saved to {CLASSIFIED_CSV}") 