import pandas as pd
import re
import nltk
import spacy
from nltk.corpus import stopwords
import sys
from pathlib import Path

# Add parent directory to path to import config_loader
sys.path.insert(0, str(Path(__file__).parent.parent))
from config_loader import get_config_path

nltk.download('stopwords', quiet=True)

nlp = spacy.load("en_core_web_sm")

raw_data_path = get_config_path('files.raw_data')
df = pd.read_json(raw_data_path, lines=True)

df['text'] = df['headline'].fillna('') + " " + df['short_description'].fillna('')

# Text cleaning function

def clean_text(text):
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)  # remove HTML tags
    text = re.sub(r'[^a-z\s]', '', text)  # keep only letters
    text = re.sub(r'\s+', ' ', text)  # remove extra spaces
    return text.strip()

df['clean_text'] = df['text'].apply(clean_text)

# Remove stopwords

stop_words = set(stopwords.words('english'))
def remove_stopwords(text):
    return ' '.join([word for word in text.split() if word not in stop_words])

df['clean_text'] = df['clean_text'].apply(remove_stopwords)

# Lemmatization using SpaCy

def lemmatize_text(text):
    doc = nlp(text)
    return ' '.join([token.lemma_ for token in doc])

df['clean_text'] = df['clean_text'].apply(lemmatize_text)

# Saving preprocessed dataset

output_path = get_config_path('files.processed_original')
df[['category', 'clean_text']].to_csv(output_path, index=False)

print(f"Preprocessing complete. Dataset saved to {output_path}")
