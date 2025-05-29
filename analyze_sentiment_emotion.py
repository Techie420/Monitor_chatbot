import pandas as pd
import os
from transformers import pipeline

# Paths
CLASSIFIED_CSV = os.path.join('..', 'data', 'monitoring_table_classified.csv')
FULL_ANALYSIS_CSV = os.path.join('..', 'data', 'monitoring_table_full_analysis.csv')

# Load classified data
print(f"Loading classified data from {CLASSIFIED_CSV}")
df = pd.read_csv(CLASSIFIED_CSV)

# Load sentiment and emotion pipelines
print("Loading sentiment pipeline (cardiffnlp/twitter-roberta-base-sentiment)...")
sentiment_pipeline = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")
print("Loading emotion pipeline (j-hartmann/emotion-english-distilroberta-base)...")
emotion_pipeline = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=1)

def analyze(row):
    text = row['combined_text']
    # Sentiment
    sentiment_result = sentiment_pipeline(text)[0]
    sentiment = sentiment_result['label']
    # Emotion
    emotion_result = emotion_pipeline(text)[0]
    emotion = emotion_result['label']
    return pd.Series({'sentiment': sentiment, 'emotion': emotion})

print("Analyzing sentiment and emotion...")
df[['sentiment', 'emotion']] = df.apply(analyze, axis=1)

# Save full analysis data
os.makedirs(os.path.dirname(FULL_ANALYSIS_CSV), exist_ok=True)
df.to_csv(FULL_ANALYSIS_CSV, index=False)
print(f"Full analysis data saved to {FULL_ANALYSIS_CSV}") 