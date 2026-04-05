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

df = pd.read_csv(get_config_path('files.processed_original'))

# Features and labels
X_text = df['clean_text']
y = df['category']


# Split data into train and test (80/20)
X_train_text, X_test_text, y_train, y_test = train_test_split(
    X_text, y, test_size=0.2, random_state=69, stratify=y
)

# Ensure there are no NaN values
X_train_text = X_train_text.fillna('')
X_test_text = X_test_text.fillna('')

# TF-IDF Vectorization with n-grams (1,2)
tfidf = TfidfVectorizer(max_features=50000, ngram_range=(1,2))
X_train = tfidf.fit_transform(X_train_text)
X_test = tfidf.transform(X_test_text)


print("TF-IDF feature engineering complete.")
print("Number of features:", X_train.shape[1])

#Feature selection using Chi-square
#By Selecting top 20,000 features
selector = SelectKBest(chi2, k=20000)
X_train = selector.fit_transform(X_train, y_train)
X_test = selector.transform(X_test)

print("Feature selection complete. Selected features:", X_train.shape[1])

# Saving the TF-IDF vectorizer, selector, and datasets
with open(get_config_path('model_artifacts.original.tfidf_vectorizer'), "wb") as f:
    pickle.dump(tfidf, f)

with open(get_config_path('model_artifacts.original.chi2_selector'), "wb") as f:
    pickle.dump(selector, f)

with open(get_config_path('model_artifacts.original.X_train'), "wb") as f:
    pickle.dump(X_train, f)

with open(get_config_path('model_artifacts.original.X_test'), "wb") as f:
    pickle.dump(X_test, f)

with open(get_config_path('model_artifacts.original.y_train'), "wb") as f:
    pickle.dump(y_train, f)

with open(get_config_path('model_artifacts.original.y_test'), "wb") as f:
    pickle.dump(y_test, f)

print("Feature matrices and models saved to models/original/")