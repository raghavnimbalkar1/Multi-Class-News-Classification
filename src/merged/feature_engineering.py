import pandas as pd
import pickle
import sys
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, chi2

# Add parent directory to path to import config_loader
sys.path.insert(0, str(Path(__file__).parent.parent))
from config_loader import get_config_path

# --- 1. Load the Merged Dataset (Updated Path) ---
print("Step 1: Loading the merged and preprocessed dataset...")
input_path = get_config_path('files.processed_merged')
df = pd.read_csv(input_path)
print(f"Loaded dataset from {input_path}")

# Features and labels
X_text = df['clean_text']
y = df['category']

# --- 2. Split Data into Train and Test (Unchanged Logic) ---
print("Step 2: Splitting data into training (80%) and testing (20%) sets...")
X_train_text, X_test_text, y_train, y_test = train_test_split(
    X_text, y, test_size=0.2, random_state=69, stratify=y
)
print("Data splitting complete.")

# Ensure there are no NaN values that might have slipped through
X_train_text = X_train_text.fillna('')
X_test_text = X_test_text.fillna('')

# --- 3. Feature Engineering (Unchanged Logic) ---
print("Step 3: Performing TF-IDF Vectorization...")
tfidf = TfidfVectorizer(max_features=50000, ngram_range=(1,2))
X_train = tfidf.fit_transform(X_train_text)
X_test = tfidf.transform(X_test_text)
print(f"TF-IDF complete. Number of features: {X_train.shape[1]}")

print("Step 4: Performing Feature Selection with Chi-Squared test...")
# By Selecting top 20,000 features
selector = SelectKBest(chi2, k=20000)
X_train = selector.fit_transform(X_train, y_train)
X_test = selector.transform(X_test)
print(f"Feature selection complete. Selected features: {X_train.shape[1]}")

# --- 4. Saving the New Artifacts (Updated Paths) ---
print("Step 5: Saving all processed data and models to '/Models/Merged/'...")

with open(get_config_path('model_artifacts.merged.tfidf_vectorizer'), "wb") as f:
    pickle.dump(tfidf, f)

with open(get_config_path('model_artifacts.merged.chi2_selector'), "wb") as f:
    pickle.dump(selector, f)

with open(get_config_path('model_artifacts.merged.X_train'), "wb") as f:
    pickle.dump(X_train, f)

with open(get_config_path('model_artifacts.merged.X_test'), "wb") as f:
    pickle.dump(X_test, f)

with open(get_config_path('model_artifacts.merged.y_train'), "wb") as f:
    pickle.dump(y_train, f)

with open(get_config_path('model_artifacts.merged.y_test'), "wb") as f:
    pickle.dump(y_test, f)

print("Feature engineering for the merged dataset is complete.")