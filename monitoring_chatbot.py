import gradio as gr
import pandas as pd
import numpy as np
import os
import pickle
from sentence_transformers import SentenceTransformer
import faiss
import re

# Paths
data_dir = os.path.join('..', 'data')
FAISS_INDEX_PATH = os.path.join(data_dir, 'faiss_index.bin')
METADATA_PATH = os.path.join(data_dir, 'metadata.pkl')
FULL_ANALYSIS_CSV = os.path.join(data_dir, 'monitoring_table_full_analysis.csv')

# Load resources
print("Loading FAISS index, metadata, and analysis data...")
index = faiss.read_index(FAISS_INDEX_PATH)
with open(METADATA_PATH, 'rb') as f:
    metadata = pickle.load(f)
df = pd.read_csv(FULL_ANALYSIS_CSV)
model = SentenceTransformer('all-MiniLM-L6-v2')

# Helper to get order info by ORDER_REF_NUM
def get_order_info(order_ref_num):
    row = df[df['ORDER_REF_NUM'] == order_ref_num]
    if row.empty:
        return None
    row = row.iloc[0]
    return {
        'Order ID': row['ORDER_REF_NUM'],
        'Error Code': row['ERROR_CODE'],
        'Category': row.get('category', ''),
        'Severity': row.get('severity', ''),
        'Sentiment': row.get('sentiment', ''),
        'Emotion': row.get('emotion', ''),
        'Error Message': row['combined_text']
    }

def get_error_code_counts():
    counts = df['ERROR_CODE'].value_counts()
    response = "Error Code Counts:\n"
    for code, count in counts.items():
        response += f"{code}: {count}\n"
    return response

def get_category_counts():
    counts = df['category'].value_counts()
    response = "Error Category Counts:\n"
    for cat, count in counts.items():
        response += f"{cat}: {count}\n"
    return response

# --- Then your chatbot_fn (update it to call these functions as needed) ---

def chatbot_fn(query):
    q = query.lower()
    # Handle count queries
    if "count" in q or "how many" in q or "number of" in q:
        if "error code" in q:
            return get_error_code_counts()
        elif "category" in q:
            return get_category_counts()
        # You can add more count logic here if needed

    # Otherwise, do semantic search as before
    query_emb = model.encode([query])
    D, I = index.search(np.array(query_emb).astype('float32'), 5)
    results = []
    for idx in I[0]:
        order = metadata[idx]
        info = get_order_info(order['ORDER_REF_NUM'])
        if info:
            results.append(info)
    if not results:
        return "No matching orders found."

    # Check if user asked for Order ID
    show_order_id = any(kw in query.lower() for kw in ["order id", "order number", "order reference"])

    # Format results
    response = "Top matching orders:\n\n"
    for i, info in enumerate(results, 1):
        if show_order_id:
            response += f"{i}. Order ID: {info['Order ID']}\n"
        else:
            response += f"{i}.\n"
        response += f"   Error Code: {info['Error Code']}\n"
        response += f"   Category: {info['Category']}\n"
        response += f"   Severity: {info['Severity']}\n"
        response += f"   Sentiment: {info['Sentiment']}\n"
        response += f"   Emotion: {info['Emotion']}\n"
        response += f"   Error Message: {info['Error Message']}\n\n"
    return response

iface = gr.Interface(
    fn=chatbot_fn,
    inputs=gr.Textbox(lines=2, placeholder="Ask about order issues, e.g. 'Show me recent payment failures'"),
    outputs="text",
    title="Payment Gateway Monitoring Chatbot",
    description="Ask about transaction issues and get detailed order error analysis."
)

if __name__ == "__main__":
    iface.launch() 