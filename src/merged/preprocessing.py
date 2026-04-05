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

# --- 1. Initial Setup ---
print("Step 1: Setting up libraries...")
nltk.download('stopwords', quiet=True) 
nlp = spacy.load("en_core_web_sm")
stop_words = set(stopwords.words('english'))
print("Setup complete.")

# --- 2. Load Raw Data (Same as before) ---
print("Step 2: Loading raw data from JSON...")
raw_data_path = get_config_path('files.raw_data')
df = pd.read_json(raw_data_path, lines=True)
print(f"Loaded {len(df)} articles.")

# --- 3. MERGE CATEGORIES (The New Step) ---
print("Step 3: Merging 42 categories into 13 super-classes...")
category_mapping = {
    # Politics & World News
    'THE WORLDPOST': 'POLITICS & WORLD NEWS',
    'WORLDPOST': 'POLITICS & WORLD NEWS',
    'WORLD NEWS': 'POLITICS & WORLD NEWS',
    'U.S. NEWS': 'POLITICS & WORLD NEWS',
    'MEDIA': 'POLITICS & WORLD NEWS',
    'IMPACT': 'POLITICS & WORLD NEWS',
    'POLITICS': 'POLITICS & WORLD NEWS',
    
    # Arts & Entertainment
    'ARTS': 'ARTS & ENTERTAINMENT',
    'ARTS & CULTURE': 'ARTS & ENTERTAINMENT',
    'CULTURE & ARTS': 'ARTS & ENTERTAINMENT',
    'COMEDY': 'ARTS & ENTERTAINMENT',
    'ENTERTAINMENT': 'ARTS & ENTERTAINMENT',
    
    # Wellness & Health
    'HEALTHY LIVING': 'WELLNESS & HEALTH',
    'FIFTY': 'WELLNESS & HEALTH',
    'WELLNESS': 'WELLNESS & HEALTH',
    
    # Business & Tech
    'MONEY': 'BUSINESS & TECH',
    'TECH': 'BUSINESS & TECH',
    'BUSINESS': 'BUSINESS & TECH',
    
    # Family & Relationships
    'PARENTS': 'FAMILY & RELATIONSHIPS',
    'DIVORCE': 'FAMILY & RELATIONSHIPS',
    'WEDDINGS': 'FAMILY & RELATIONSHIPS',
    'PARENTING': 'FAMILY & RELATIONSHIPS',
    
    # Home & Style
    'STYLE': 'HOME & STYLE',
    'STYLE & BEAUTY': 'HOME & STYLE',
    'HOME & LIVING': 'HOME & STYLE',
    
    # Food & Travel
    'TASTE': 'FOOD & TRAVEL',
    'TRAVEL': 'FOOD & TRAVEL',
    'FOOD & DRINK': 'FOOD & TRAVEL',

    # Science & Environment
    'GREEN': 'SCIENCE & ENVIRONMENT',
    'COLLEGE': 'SCIENCE & ENVIRONMENT',
    'EDUCATION': 'SCIENCE & ENVIRONMENT',
    'ENVIRONMENT': 'SCIENCE & ENVIRONMENT',
    'SCIENCE': 'SCIENCE & ENVIRONMENT',
    
    # Voices & Identity
    'LATINO VOICES': 'VOICES & IDENTITY',
    'QUEER VOICES': 'VOICES & IDENTITY',
    'BLACK VOICES': 'VOICES & IDENTITY',
    'WOMEN': 'VOICES & IDENTITY',
    
    # General News & Oddities
    'GOOD NEWS': 'GENERAL NEWS & ODDITIES',
    'WEIRD NEWS': 'GENERAL NEWS & ODDITIES'
    
    # Crime, Sports, and Religion remain unchanged.
}

# Apply the mapping to the 'category' column
df['category'] = df['category'].replace(category_mapping)
print(f"Merging complete. New number of categories: {df['category'].nunique()}")


# --- 4. Text Cleaning and Processing (Same as before) ---
print("Step 4: Cleaning and processing text...")

df['text'] = df['headline'].fillna('') + " " + df['short_description'].fillna('')

def clean_text(text):
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def remove_stopwords(text):
    return ' '.join([word for word in text.split() if word not in stop_words])

def lemmatize_text(text):
    doc = nlp(text)
    return ' '.join([token.lemma_ for token in doc])

df['clean_text'] = df['text'].apply(clean_text)
df['clean_text'] = df['clean_text'].apply(remove_stopwords)
df['clean_text'] = df['clean_text'].apply(lemmatize_text)
print("Text processing complete.")


# --- 5. Save the New Preprocessed Dataset (Updated Path) ---

print("Step 5: Saving the new merged dataset...")
output_path = get_config_path('files.processed_merged')
df[['category', 'clean_text']].to_csv(output_path, index=False)
print(f"Preprocessing complete. New dataset saved to {output_path}")