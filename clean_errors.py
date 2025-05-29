import pandas as pd
import os

# Paths
RAW_CSV = os.path.join('..', 'data', 'monitoring_table_raw.csv')
CLEANED_CSV = os.path.join('..', 'data', 'monitoring_table_cleaned.csv')

# Load raw data
print(f"Loading raw data from {RAW_CSV}")
df = pd.read_csv(RAW_CSV)

# Fill missing error fields with empty strings
df['API_ERROR_MSG'] = df['API_ERROR_MSG'].fillna('')
df['API_ERROR_CODE'] = df['API_ERROR_CODE'].fillna('')

# Create combined_text by concatenating error fields
def combine_errors(row):
    return ' '.join([
       
        str(row['API_ERROR_MSG']).strip(),
        str(row['API_ERROR_CODE']).strip()
    ]).strip()

df['combined_text'] = df.apply(combine_errors, axis=1)

# Clean whitespace in combined_text
df['combined_text'] = df['combined_text'].str.replace(r'\s+', ' ', regex=True).str.strip()

# Save cleaned data
os.makedirs(os.path.dirname(CLEANED_CSV), exist_ok=True)
df.to_csv(CLEANED_CSV, index=False)
print(f"Cleaned data saved to {CLEANED_CSV}") 