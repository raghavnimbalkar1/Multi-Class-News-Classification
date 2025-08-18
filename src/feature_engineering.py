import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, chi2

df = pd.read_csv("/home/skinny/Documents/Code/MultiClassNewsClassification/Data/processed/news_preprocessed.csv")

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
with open("/home/skinny/Documents/Code/MultiClassNewsClassification/models/tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(tfidf, f)

with open("/home/skinny/Documents/Code/MultiClassNewsClassification/models/chi2_selector.pkl", "wb") as f:
    pickle.dump(selector, f)

with open("/home/skinny/Documents/Code/MultiClassNewsClassification/models/X_train.pkl", "wb") as f:
    pickle.dump(X_train, f)

with open("/home/skinny/Documents/Code/MultiClassNewsClassification/models/X_test.pkl", "wb") as f:
    pickle.dump(X_test, f)

with open("/home/skinny/Documents/Code/MultiClassNewsClassification/models/y_train.pkl", "wb") as f:
    pickle.dump(y_train, f)

with open("/home/skinny/Documents/Code/MultiClassNewsClassification/models/y_test.pkl", "wb") as f:
    pickle.dump(y_test, f)

print("Feature matrices and models saved to ../models/")