cat << 'EOF' > README.md
# Dynamic Analytics Dashboard System (DADS) & News Classifier

An end-to-end Machine Learning pipeline and web scraping tool that extracts live news articles from websites and automatically classifies them into 13 distinct categories using a custom-trained Support Vector Machine (LinearSVC).

## Project Overview

This project bridges the gap between static machine learning models and live, real-world data. It consists of two completed phases, with a third on the way:
1. **Data Collection (Live Scraper):** Dynamically extracts headlines and article summaries from top news homepages (like The Guardian, BBC, CBC) or single article URLs.
2. **Inference Pipeline (ML Engine):** Processes the scraped text using TF-IDF and Chi-Squared selection, feeding it into an optimized LinearSVC model to predict news categories (e.g., POLITICS, SPORTS, TECHNOLOGY).
3. **Frontend & Analytics (Upcoming):** A Streamlit dashboard to visualize category distributions, trending keywords, and summary metrics.

## Project Structure

`models/`
`├── merged/`
`│   ├── tfidf_vectorizer.pkl   # Trained TF-IDF vectorizer`
`│   ├── chi2_selector.pkl      # Chi-Squared feature selector`
`│   ├── svm_model.pkl          # Optimized 13-class LinearSVC`
`│   └── label_encoder.pkl      # Target label decoder`

`src/`
`├── scraper.py                 # Live web scraping engine (newspaper)`
`└── inference.py               # Connects live data to ML models`

`README.md`

## Setup & Installation

**1. Clone the repository**
git clone https://github.com/yourusername/Multi-Class-News-Classification.git
cd Multi-Class-News-Classification

**2. Install dependencies**
Make sure you have Python 3.8+ installed. 
pip install pandas scikit-learn nltk newspaper3k

*Note for macOS users:* If you encounter OpenSSL/LibreSSL errors while scraping, downgrade `urllib3`:
pip install "urllib3<2"

**3. Download required NLTK datasets**
The scraping and preprocessing engines rely on specific NLTK dictionaries. Run the following commands in your terminal to fetch them:
python3 -m nltk.downloader wordnet omw-1.4 punkt_tab

## Usage

Currently, the pipeline runs entirely via the Command Line Interface (CLI). 

### Testing the Scraper
To extract articles from a news homepage without running predictions, run:
python3 src/scraper.py

*Output: A Pandas DataFrame containing headline and short_description.*

### Running the End-to-End Inference
To scrape a website, preprocess the live text, extract features, and predict the news categories using the trained model, run:
python3 src/inference.py

*Output: A Pandas DataFrame appending a predicted_category column to the scraped articles.*

## Architecture Flow

1. **Input:** User provides a URL (Homepage or Single Article).
2. **Scraper:** `newspaper` fetches HTML, bypasses basic bot-protections, and extracts text. `nltk` summarizes the text.
3. **Preprocessing:** Text is cleaned (regex, lowercased).
4. **Vectorization:** `tfidf_vectorizer` converts text to unigram/bigram numerical weights.
5. **Feature Selection:** `chi2_selector` narrows down the top 20,000 most relevant features.
6. **Prediction:** `svm_model` predicts the classification label.
7. **Output:** Results are decoded via `label_encoder` and returned for analytics.

##  Next Steps
- [ ] Build interactive Streamlit User Interface.
- [ ] Generate keyword distribution analytics.
- [ ] Deploy to cloud hosting.
EOF
