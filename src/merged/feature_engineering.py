import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, chi2

# --- 1. Load the Merged Dataset (Updated Path) ---
print("Step 1: Loading the merged and preprocessed dataset...")
input_path = "../../Data/processed/news_preprocessed_merged.csv"
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

with open("../../models/merged/tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(tfidf, f)

with open("../../models/merged/chi2_selector.pkl", "wb") as f:
    pickle.dump(selector, f)

with open("../../models/merged/X_train.pkl", "wb") as f:
    pickle.dump(X_train, f)

with open("../../models/merged/X_test.pkl", "wb") as f:
    pickle.dump(X_test, f)

with open("../../models/merged/y_train.pkl", "wb") as f:
    pickle.dump(y_train, f)

with open("../../models/merged/y_test.pkl", "wb") as f:
    pickle.dump(y_test, f)

print("Feature engineering for the merged dataset is complete.")