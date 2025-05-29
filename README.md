# Payment Gateway NLP Monitoring Chatbot

This project provides an end-to-end NLP-based monitoring chatbot for analyzing transaction issue data from a payment gateway system. It leverages modern NLP models for error categorization, severity assessment, sentiment, and emotion analysis, and provides a Gradio-based chatbot interface for natural language querying of order issues.

## Project Structure
```
/data
    monitoring_table_raw.csv              # Input: Exported Oracle data
    monitoring_table_cleaned.csv          # Output: Cleaned data
    monitoring_table_classified.csv       # Output: Classified data
    monitoring_table_full_analysis.csv    # Output: Sentiment & emotion analysis
    faiss_index.bin                       # FAISS vector index
    metadata.pkl                          # Order metadata for retrieval
/scripts
    clean_errors.py
    generate_embeddings.py
    classify_errors.py
    analyze_sentiment_emotion.py
    monitoring_chatbot.py
requirements.txt
README.md
```

## Pipeline Overview
1. **Data Cleaning**: Cleans and combines error message fields.
2. **Embedding Generation**: Generates sentence embeddings and builds a FAISS index.
3. **Error Classification**: Categorizes errors and assigns severity using zero-shot classification.
4. **Sentiment & Emotion Analysis**: Analyzes sentiment and emotion of error messages.
5. **Chatbot Interface**: Provides a Gradio chatbot for querying order issues.

## Setup
1. Clone the repository and navigate to the project directory.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Place your exported Oracle data as `/data/monitoring_table_raw.csv`.

## Running the Pipeline

### 1. Data Cleaning
```bash
python scripts/clean_errors.py
```
- Input: `/data/monitoring_table_raw.csv`
- Output: `/data/monitoring_table_cleaned.csv`

### 2. Embedding Generation
```bash
python scripts/generate_embeddings.py
```
- Input: `/data/monitoring_table_cleaned.csv`
- Output: `/data/faiss_index.bin`, `/data/metadata.pkl`

### 3. Error Classification
```bash
python scripts/classify_errors.py
```
- Input: `/data/monitoring_table_cleaned.csv`
- Output: `/data/monitoring_table_classified.csv`

### 4. Sentiment & Emotion Analysis
```bash
python scripts/analyze_sentiment_emotion.py
```
- Input: `/data/monitoring_table_classified.csv`
- Output: `/data/monitoring_table_full_analysis.csv`

### 5. Launch the Chatbot
```bash
python scripts/monitoring_chatbot.py
```
- Opens a Gradio web interface for natural language queries about order issues.

## Example Query
- "Show me recent payment failures"
- "Find orders with authentication errors"
- "Which orders had network issues yesterday?"

## Notes
- All intermediate and output files are stored in the `/data` directory.
- The chatbot returns the top 5 most similar orders to your query, with error details, category, severity, sentiment, and emotion.
- Make sure you have a compatible GPU or sufficient CPU resources for running transformer models efficiently.

## Requirements
See `requirements.txt` for all dependencies.

## License
MIT License 