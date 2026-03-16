# Dynamic Analytics Dashboard System (DADS) & News Classifier

An end-to-end Machine Learning pipeline and web scraping tool that extracts live news articles from websites and automatically classifies them into 13 distinct categories using a custom-trained Support Vector Machine (LinearSVC).

## Project Overview

This project bridges the gap between static machine learning models and live, real-world data. It consists of two completed phases, with a third on the way:
1. **Data Collection (Live Scraper):** Dynamically extracts headlines and article summaries from top news homepages (like The Guardian, BBC, CBC) or single article URLs.
2. **Inference Pipeline (ML Engine):** Processes the scraped text using TF-IDF and Chi-Squared selection, feeding it into an optimized LinearSVC model to predict news categories (e.g., POLITICS, SPORTS, TECHNOLOGY).
3. **Frontend & Analytics (Upcoming):** A Streamlit dashboard to visualize category distributions, trending keywords, and summary metrics.

## Project Structure

```text
Multi-Class-News-Classification/
│
├── models/
│   └── merged/
│       ├── tfidf_vectorizer.pkl   # Trained TF-IDF vectorizer
│       ├── chi2_selector.pkl      # Chi-Squared feature selector
│       ├── svm_model.pkl          # Optimized 13-class LinearSVC
│       └── label_encoder.pkl      # Target label decoder
│
├── src/
│   ├── scraper.py                 # Live web scraping engine (newspaper)
│   └── inference.py               # Connects live data to ML models
│
└── README.md

## Setup Installation
