import pandas as pd
import numpy as np
import os
import pickle
from sentence_transformers import SentenceTransformer
import faiss

# Paths
DATA_DIR = os.path.join('..', 'data')
CLEANED_CSV = os.path.join(DATA_DIR, 'monitoring_table_cleaned.csv')
ALL_TEXTS_PATH = os.path.join(DATA_DIR, 'all_combined_texts.pkl')
ALL_EMBEDDINGS_PATH = os.path.join(DATA_DIR, 'all_embeddings.npy')
FAISS_INDEX_PATH = os.path.join(DATA_DIR, 'faiss_index.bin')
METADATA_PATH = os.path.join(DATA_DIR, 'metadata.pkl')

# Load cleaned data
df = pd.read_csv(CLEANED_CSV)
new_texts = df['combined_text'].tolist()

# Load previous unique texts and embeddings if they exist
if os.path.exists(ALL_TEXTS_PATH) and os.path.exists(ALL_EMBEDDINGS_PATH):
    with open(ALL_TEXTS_PATH, 'rb') as f:
        all_texts = pickle.load(f)
    all_embeddings = np.load(ALL_EMBEDDINGS_PATH)
    print(f"Loaded {len(all_texts)} previous unique error messages.")
else:
    all_texts = []
    all_embeddings = np.zeros((0, 384), dtype=np.float32)  # 384 for MiniLM
    print("No previous embeddings found, starting fresh.")

# Find new unique texts
unique_new_texts = list(set(new_texts) - set(all_texts))
print(f"Found {len(unique_new_texts)} new unique error messages.")

# Embed only new unique texts
if unique_new_texts:
    model = SentenceTransformer('all-MiniLM-L6-v2')
    new_embeddings = model.encode(unique_new_texts, show_progress_bar=True, convert_to_numpy=True)
    # Append to all_texts and all_embeddings
    all_texts.extend(unique_new_texts)
    all_embeddings = np.vstack([all_embeddings, new_embeddings])

# Save updated unique texts and embeddings
with open(ALL_TEXTS_PATH, 'wb') as f:
    pickle.dump(all_texts, f)
np.save(ALL_EMBEDDINGS_PATH, all_embeddings)
print(f"Saved {len(all_texts)} total unique error messages and embeddings.")

# Build or update FAISS index
dim = all_embeddings.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(all_embeddings)
faiss.write_index(index, FAISS_INDEX_PATH)
print(f"FAISS index saved to {FAISS_INDEX_PATH}")

# Map each combined_text to its index in all_texts
text2idx = {text: idx for idx, text in enumerate(all_texts)}

# Save metadata for all orders in the current file
metadata = []
for _, row in df.iterrows():
    idx = text2idx[row['combined_text']]
    metadata.append({
        'ORDER_REF_NUM': row['ORDER_REF_NUM'],
        'API_ERROR_CODE': row['API_ERROR_CODE'],
        'API_ERROR_MSG': row['API_ERROR_MSG'],
        'combined_text': row['combined_text'],
        'embedding_idx': idx
    })
with open(METADATA_PATH, 'wb') as f:
    pickle.dump(metadata, f)
print(f"Metadata saved to {METADATA_PATH}") 