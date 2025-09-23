import pandas as pd
import re
import nltk
import spacy
from nltk.corpus import stopwords

nltk.download('stopwords')

nlp = spacy.load("en_core_web_sm")

df = pd.read_json("/home/skinny/Documents/Code/MultiClassNewsClassification/Data/raw/NewsData.json", lines=True)

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

df[['category', 'clean_text']].to_csv("/home/skinny/Documents/Code/MultiClassNewsClassification/Data/processed/news_preprocessed.csv", index=False)

print("Preprocessing complete. Dataset saved to data/processed/news_preprocessed.csv")
