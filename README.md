# Multi-Class News Classification System

A comprehensive machine learning pipeline for classifying news articles into multiple categories using NLP and advanced classification algorithms.

## Project Overview

This project implements a complete end-to-end news classification system with two parallel approaches:

1. **Original Approach**: 42 news categories using LinearSVC
2. **Merged Approach**: 13 consolidated super-categories with both LinearSVC and XGBoost models

### Architecture

The system follows a modular 4-stage pipeline:

```
Stage 1: Data Collection  →  Stage 2: Preprocessing  →  Stage 3: ML Pipeline  →  Stage 4: Analytics
   (Scraper)                (Text Cleaning, NLP)      (Training, Evaluation)    (Reports, Dashboard)
```

## Quick Start

### Prerequisites

- Python 3.8+
- pip or conda
- 4GB+ RAM recommended
- macOS/Linux/Windows

### Installation

1. **Clone the repository**:
```bash
git clone https://github.com/raghavnimbalkar1/Multi-Class-News-Classification.git
cd Multi-Class-News-Classification
```

2. **Create a virtual environment** (recommended):
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

4. **Download NLP models** (first time only):
```bash
python3 -c "
import nltk; import spacy
nltk.download('stopwords')
import subprocess
subprocess.run(['python', '-m', 'spacy', 'download', 'en_core_web_sm'])
"
```

## Usage

### Run Complete Pipeline

```bash
# Run both original and merged approaches
python3 src/pipeline.py --approach all

# Run only original approach (42 categories)
python3 src/pipeline.py --approach original

# Run only merged approach (13 categories)
python3 src/pipeline.py --approach merged
```

### Individual Step Execution

**Original Pipeline (42 categories)**:
```bash
python3 src/original/preprocessing.py          # Clean data
python3 src/original/feature_engineering.py    # TF-IDF + Feature Selection
python3 src/original/train_model.py            # Train SVM
python3 src/original/evaluate.py               # Generate reports
```

**Merged Pipeline (13 categories)**:
```bash
python3 src/merged/preprocessing.py            # Merge + Clean data
python3 src/merged/feature_engineering.py      # TF-IDF + Feature Selection
python3 src/merged/train_model.py              # Train SVM
python3 src/merged/train_xgboost.py            # Train XGBoost
python3 src/merged/evaluate.py                 # Evaluate SVM
python3 src/merged/evaluate_xgboost.py         # Evaluate XGBoost
```

### Analyze & Compare Models

```bash
python3 src/model_analysis.py
```

Generates:
- `model_comparison_report.txt` - Detailed metrics comparison
- `visualizations/accuracy_comparison.png` - Accuracy chart

### Interactive Dashboard (NEW!)

Launch the **Streamlit web application** for an interactive interface:

```bash
# Quick launch with automatic dependency installation
python3 src/launch_dashboard.py

# Or manually start Streamlit:
streamlit run src/app.py
```

The dashboard will open at `http://localhost:8501` and provides:

#### Features:
1. **Single Article Predictor**
   - Paste or type article text
   - Get instant category prediction
   - View confidence scores
   - See top 5 predictions

2. **Batch Processor**
   - Upload CSV with multiple articles
   - Process all at once
   - Download results as CSV
   - Track processing progress

3. **Model Analytics**
   - Compare model performance (Original vs Merged, SVM vs XGBoost)
   - View confusion matrices
   - Analyze per-category metrics
   - Summary statistics

4. **Live News Scraper**
   - Enter news article URLs
   - Automatically scrape and extract content
   - Classify fetched articles
   - View article metadata

5. **Documentation**
   - Quick start guide
   - Category reference
   - Configuration details
   - Troubleshooting

#### Dashboard Models:
- Switch between approaches: Original (42 categories) or Merged (13 categories)
- For merged: Choose between LinearSVC or XGBoost
- Auto-loads correct model based on selection

## Project Structure

```
Multi-Class-News-Classification/
├── config.yaml                    # Configuration (paths, parameters)
├── requirements.txt               # Python dependencies
├── README.md                      # This file
├── Data/
│   ├── raw/                       # Raw news data (NewsData.json)
│   └── processed/                 # Preprocessed datasets
├── models/
│   ├── original/                  # 42-category models + reports
│   └── merged/                    # 13-category models + reports
├── src/
│   ├── config_loader.py           # Configuration management
│   ├── pipeline.py                # Unified pipeline orchestration
│   ├── model_analysis.py          # Model comparison utilities
│   ├── scraper.py                 # Web scraping utilities
│   ├── inference.py               # Prediction utilities (NEW)
│   ├── visualization.py           # Dashboard visualization utilities (NEW)
│   ├── app.py                     # Streamlit dashboard application (NEW)
│   ├── launch_dashboard.py        # Dashboard launcher script (NEW)
│   ├── original/                  # 42-category pipeline
│   │   ├── preprocessing.py
│   │   ├── feature_engineering.py
│   │   ├── train_model.py
│   │   └── evaluate.py
│   └── merged/                    # 13-category pipeline
│       ├── preprocessing.py
│       ├── feature_engineering.py
│       ├── train_model.py
│       ├── train_xgboost.py
│       ├── evaluate.py
│       └── evaluate_xgboost.py
├── visualizations/                # Generated charts and visualizations
└── logs/                          # Pipeline execution logs
```

## Configuration

Edit `config.yaml` to customize paths, preprocessing, and model parameters:

```yaml
preprocessing:
  remove_html: true
  lowercase: true
  remove_stopwords: true
  lemmatization: true

features:
  tfidf:
    max_features: 50000
    ngram_range: [1, 2]
  feature_selection:
    k_best_features: 20000
```

## Pipeline Steps

### 1. Preprocessing
- Clean text (lowercase, remove HTML/special characters)
- Remove stopwords using NLTK
- Lemmatization using SpaCy

### 2. Feature Engineering
- TF-IDF Vectorization (50K features, unigrams + bigrams)
- Chi-Squared Feature Selection (top 20K features)
- Train/Test Split (80/20, stratified)

### 3. Model Training
- **Original**: LinearSVC on 42 categories
- **Merged**: LinearSVC + XGBoost on 13 categories

### 4. Evaluation
- Per-class metrics (precision, recall, F1-score)
- Macro & weighted averages
- Confusion matrices
- Accuracy scores

## Category Mapping (13 Super-Categories)

| Super-Category | Original Categories (Sample) |
|---|---|
| **Politics & World News** | THE WORLDPOST, WORLD NEWS, U.S. NEWS, POLITICS |
| **Arts & Entertainment** | ARTS, COMEDY, ENTERTAINMENT |
| **Wellness & Health** | HEALTHY LIVING, WELLNESS |
| **Business & Tech** | MONEY, TECH, BUSINESS |
| **Science & Environment** | GREEN, EDUCATION, ENVIRONMENT, SCIENCE |
| **Sports** | SPORTS |
| **And 7 more...** | See config.yaml for full mapping |

## Expected Output

### Models
```
models/original/svm_model.pkl                    # 42-class SVM
models/merged/svm_model.pkl                      # 13-class SVM
models/merged/xgboost_model.pkl                  # 13-class XGBoost
```

### Reports
```
models/original/classification_report.csv        # Precision/Recall/F1
models/original/confusion_matrix_numeric.csv     # Confusion Matrix
models/merged/classification_report.csv          # SVM metrics
models/merged/xgboost_classification_report.csv  # XGBoost metrics
```

## Troubleshooting

### Missing NLP models
```bash
python3 -m spacy download en_core_web_sm
python3 -c "import nltk; nltk.download('stopwords')"
```

### Memory issues with large datasets
Reduce these in `config.yaml`:
```yaml
max_features: 25000      # Was 50000
k_best_features: 10000   # Was 20000
```

### Check logs for details
```bash
tail -f logs/pipeline_*.log
```

## Technical Details

**Algorithms**:
- LinearSVC: Support Vector Machine for multi-class classification
- XGBoost: Gradient boosting for enhanced performance

**NLP Preprocessing**:
- Text cleaning with regex
- Stopword removal (English)
- Lemmatization with SpaCy
- TF-IDF feature extraction

**Feature Selection**:
- Chi-squared test for independence
- Selects 20K most informative features

## Python API

### Configuration Access
```python
import sys
sys.path.insert(0, 'src')

from config_loader import get_config_path, load_config

# Get specific path
model_path = get_config_path('model_artifacts.merged.xgboost_model')

# Load full config
config = load_config()
```

### Making Predictions
```python
from inference import get_classifier

# Initialize classifier
classifier = get_classifier(approach='merged', model_type='svm')

# Single prediction
result = classifier.predict("Apple announces new iPhone features")
print(f"Category: {result['prediction']}")
print(f"Confidence: {result['confidence']:.1%}")
print(f"Top 5: {result['probabilities']}")

# Batch predictions
texts = ["Article 1 text", "Article 2 text", "Article 3 text"]
results_df = classifier.batch_predict(texts)
```

### Model Analysis
```python
from model_analysis import ModelAnalyzer

analyzer = ModelAnalyzer()
report = analyzer.save_comparison_report()
analyzer.generate_all_visualizations()
```

## Dependencies

- **ML/Data**: pandas, numpy, scikit-learn, xgboost
- **NLP**: nltk, spacy
- **Web UI**: streamlit (new)
- **Visualization**: matplotlib, seaborn
- **Utils**: PyYAML, newspaper4k

See `requirements.txt` for versions.

## Completed Features

- [Completed] Streamlit web dashboard
- [Completed] Configuration management system
- [Completed] Multi-approach ML pipeline (original 42-class, merged 13-class)
- [Completed] LinearSVC model training and evaluation
- [Completed] XGBoost model training and evaluation
- [Completed] News article web scraper
- [Completed] Model analysis and comparison utilities
- [Completed] Interactive prediction interface
- [Completed] Batch processing capabilities
- [Completed] Model analytics and visualization
- [Completed] Live scraper integration
- [Completed] Comprehensive documentation

## Future Enhancements

- REST API endpoint
- Deep learning models (LSTM, Transformers)
- BERT embeddings
- Docker containerization
- Model versioning
- Hyperparameter optimization

## Support

- Check README for setup issues
- Review logs in `logs/` directory
- Open GitHub issue for bugs

## License

[Add license information]

## Author

Raghav Nimbalkar

---

**Last Updated**: April 5, 2026  
**Version**: 1.0.0  
**Status**: Complete - All Features Implemented and Tested
