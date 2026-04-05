"""
Multi-Class News Classification - Streamlit Dashboard
Interactive web interface for news classification and model analytics
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import warnings

warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent))
from config_loader import get_config_path
from inference import get_classifier, NewsClassifier
from visualization import (
    plot_confusion_matrix, plot_accuracy_comparison,
    plot_top_categories_performance, create_model_summary_table,
    get_category_mapping
)


# Page configuration
st.set_page_config(
    page_title="News Classification Dashboard",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        padding: 2rem;
    }
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 1.2rem;
        font-weight: 600;
    }
    .success-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
    }
    .error-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
    }
</style>
""", unsafe_allow_html=True)


# Initialize session state
if 'classifier' not in st.session_state:
    st.session_state.classifier = None
if 'approach' not in st.session_state:
    st.session_state.approach = 'merged'
if 'model_type' not in st.session_state:
    st.session_state.model_type = 'svm'


@st.cache_resource
def load_classifier_cached(approach: str, model_type: str) -> NewsClassifier:
    """Cache classifier to avoid reloading on every interaction."""
    try:
        classifier = get_classifier(approach=approach, model_type=model_type)
        return classifier
    except Exception as e:
        st.error(f"Failed to load classifier: {e}")
        return None


def header_section():
    """Render header with title and configuration."""
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.title("📰 News Classification Dashboard")
        st.markdown("*Predict news categories using machine learning*")
    
    with col2:
        approach = st.selectbox(
            "Classification Approach",
            ["merged", "original"],
            key="approach_select"
        )
    
    with col3:
        if approach == "merged":
            model_type = st.selectbox(
                "Model Type",
                ["svm", "xgboost"],
                key="model_type_select"
            )
        else:
            model_type = "svm"
            st.selectbox("Model Type", ["svm"], disabled=True)
    
    return approach, model_type


def tab1_predictor(classifier: NewsClassifier):
    """Single article prediction tab."""
    st.header("Single Article Predictor")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Input methods
        input_method = st.radio("Input Method", ["Type Text", "Paste from Clipboard"])
        
        if input_method == "Type Text":
            text = st.text_area(
                "Enter news headline or article text:",
                height=150,
                placeholder="Example: Tech giant Apple announces new iPhone with advanced AI features..."
            )
        else:
            text = st.text_area(
                "Paste your text here:",
                height=150,
                placeholder="Paste news content here..."
            )
    
    with col2:
        st.subheader("Text Info")
        if text:
            st.metric("Characters", len(text))
            st.metric("Words", len(text.split()))
        else:
            st.info("No text entered yet")
    
    # Predict button
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        if st.button("Classify", use_container_width=True):
            if not text or len(text.strip()) < 10:
                st.error("Please enter at least 10 characters")
            else:
                with st.spinner("Classifying..."):
                    result = classifier.predict(text)
                
                if result['success']:
                    col_pred, col_conf = st.columns([1, 1])
                    
                    with col_pred:
                        st.success(f"**Prediction: {result['prediction']}**")
                        st.metric("Confidence", f"{result['confidence']:.1%}")
                    
                    with col_conf:
                        st.subheader("Top 5 Predictions")
                        probs_df = pd.DataFrame(
                            list(result['probabilities'].items()),
                            columns=['Category', 'Probability']
                        )
                        probs_df['Probability'] = probs_df['Probability'].apply(lambda x: f"{x:.1%}")
                        st.dataframe(probs_df, use_container_width=True, hide_index=True)
                else:
                    st.error(f"Classification failed: {result.get('error', 'Unknown error')}")


def tab2_batch_processor(classifier: NewsClassifier):
    """Batch processing tab."""
    st.header("Batch Processor")
    
    upload_method = st.radio("Upload Method", ["CSV File", "Paste Text Lines"])
    
    if upload_method == "CSV File":
        uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
        
        if uploaded_file:
            try:
                df = pd.read_csv(uploaded_file)
                
                # Let user select column
                if len(df.columns) > 1:
                    text_column = st.selectbox("Select text column", df.columns)
                else:
                    text_column = df.columns[0]
                
                texts = df[text_column].astype(str).tolist()
                
                if st.button("🚀 Process Batch"):
                    with st.spinner(f"Processing {len(texts)} articles..."):
                        results_df = classifier.batch_predict(texts)
                    
                    st.success(f"Processed {len(results_df)} articles")
                    st.dataframe(results_df, use_container_width=True)
                    
                    # Download results
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        "📥 Download Results as CSV",
                        csv,
                        "predictions.csv",
                        "text/csv",
                        use_container_width=True
                    )
            except Exception as e:
                st.error(f"Error processing file: {e}")
    
    else:  # Paste Text Lines
        text_input = st.text_area(
            "Paste one headline/text per line:",
            height=200,
            placeholder="Line 1: First article\nLine 2: Second article\n..."
        )
        
        if st.button("🚀 Process Lines"):
            if text_input.strip():
                texts = [line.strip() for line in text_input.split('\n') if line.strip()]
                
                with st.spinner(f"Processing {len(texts)} articles..."):
                    results_df = classifier.batch_predict(texts)
                
                st.success(f"Processed {len(results_df)} articles")
                st.dataframe(results_df, use_container_width=True)
                
                csv = results_df.to_csv(index=False)
                st.download_button(
                    "📥 Download Results as CSV",
                    csv,
                    "predictions.csv",
                    "text/csv",
                    use_container_width=True
                )
            else:
                st.error("Please enter at least one article")


def tab3_model_analytics(approach: str, model_type: str, classifier: NewsClassifier):
    """Model analytics and comparison tab."""
    st.header("Model Analytics")
    
    analytics_tab1, analytics_tab2, analytics_tab3 = st.tabs(
        ["Model Comparison", "Confusion Matrix", "Category Performance"]
    )
    
    with analytics_tab1:
        st.subheader("Overall Model Performance")
        
        # Accuracy comparison
        st.markdown("### Accuracy Comparison")
        fig = plot_accuracy_comparison()
        st.pyplot(fig)
        
        # Model summary
        st.markdown("### Current Model Summary")
        summary_df = create_model_summary_table(approach, model_type)
        st.dataframe(summary_df, use_container_width=True, hide_index=True)
        
        # Model info
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Categories", len(classifier.get_classes()))
        with col2:
            st.metric("Approach", approach.title())
        with col3:
            st.metric("Model Type", model_type.upper())
    
    with analytics_tab2:
        st.markdown("### Confusion Matrix")
        fig = plot_confusion_matrix(approach, model_type)
        st.pyplot(fig)
    
    with analytics_tab3:
        st.markdown("### Per-Category Performance")
        fig = plot_top_categories_performance(approach)
        st.pyplot(fig)


def tab4_live_scraper():
    """Live news scraping tab."""
    st.header("Live News Scraper")
    
    st.info("""
    This tab allows you to scrape news from URLs and classify them automatically.
    **Note:** Requires internet connection and target website accessibility.
    """)
    
    url = st.text_input(
        "Enter news article URL:",
        placeholder="https://example.com/news/article"
    )
    
    if st.button("Scrape & Classify"):
        if not url:
            st.error("Please enter a valid URL")
        else:
            try:
                from scraper import scrape_single_article
                
                with st.spinner("Fetching article..."):
                    article_data = scrape_single_article(url)
                
                if article_data and 'title' in article_data:
                    st.success("Article fetched successfully!")
                    
                    # Display article info
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.subheader("Article Information")
                        st.write(f"**Title:** {article_data.get('title', 'N/A')}")
                        st.write(f"**Authors:** {', '.join(article_data.get('authors', ['Unknown']))}")
                        st.write(f"**Published:** {article_data.get('publish_date', 'N/A')}")
                        st.write(f"**Source:** {article_data.get('source_url', 'N/A')}")
                        
                        # Show excerpt
                        text = article_data.get('text', '')
                        st.write("**Article Preview:**")
                        st.write(text[:500] + "..." if len(text) > 500 else text)
                    
                    with col2:
                        # Classify
                        if st.button("Classify Article"):
                            classifier = load_classifier_cached('merged', 'svm')
                            if classifier:
                                result = classifier.predict(article_data.get('title', '') + ' ' + text)
                                
                                if result['success']:
                                    st.success(f"**Category: {result['prediction']}**")
                                    st.metric("Confidence", f"{result['confidence']:.1%}")
                                else:
                                    st.error("Classification failed")
                else:
                    st.error("Failed to retrieve article. Check URL and try again.")
            
            except ImportError:
                st.warning("Scraper module not available. Please ensure newspaper4k is installed.")
            except Exception as e:
                st.error(f"Error: {e}")


def tab5_documentation():
    """Documentation and help tab."""
    st.header("Documentation & Help")
    
    doc_tab1, doc_tab2, doc_tab3 = st.tabs(["Quick Start", "Categories", "Settings"])
    
    with doc_tab1:
        st.markdown("""
        ### Quick Start Guide
        
        **1. Single Prediction**
        - Enter a news headline or article text
        - Click "Classify"
        - View the predicted category and confidence
        
        **2. Batch Processing**
        - Upload a CSV file or paste multiple texts
        - Select the text column (for CSV)
        - Process to get predictions for all items
        - Download results
        
        **3. Model Analytics**
        - View overall model performance
        - Compare different approaches (Original vs Merged)
        - Analyze per-category metrics
        - View confusion matrices
        
        **4. Live Scraping**
        - Enter a news article URL
        - Scrape and automatically classify
        - View article metadata
        
        ### Model Approaches
        
        - **Original (42 Categories)**: Fine-grained classification using LinearSVC
        - **Merged (13 Categories)**: General categories with LinearSVC or XGBoost
        """)
    
    with doc_tab2:
        st.markdown("### Classification Categories")
        
        # Get categories from config
        try:
            approach = st.session_state.get('approach', 'merged')
            suffix = f'.{approach}'
            
            # Display categories
            st.info(f"**Selected Approach:** {approach.title()}")
            
            if approach == 'merged':
                categories = [
                    "Business", "Entertainment", "General News", "Health",
                    "Politics", "Science & Tech", "Sports", "Technology",
                    "Travel", "Food", "Lifestyle", "World", "India"
                ]
            else:
                categories = [
                    "CRIME", "DOMESTIC MARKETS", "EQUITY MARKET OPERATIONS",
                    "FOREX", "MONEY MARKET", "MUTUAL FUNDS", "GENERAL FINANCE",
                    "TECH", "ENTERTAINMENT", "SPORTS", "INDIA", "WORLD", "POLITICS",
                    "HEALTH", "BUSINESS", "SCIENCE", "EDUCATION", "TRAVEL",
                    "LIFESTYLE", "FOOD", "AUTO", "REAL ESTATE", "OTHERS"
                ]
            
            # Show as columns
            cols = st.columns(3)
            for i, cat in enumerate(categories):
                with cols[i % 3]:
                    st.write(f"• {cat}")
        
        except Exception as e:
            st.warning(f"Could not load categories: {e}")
    
    with doc_tab3:
        st.markdown("### Settings & Configuration")
        
        st.markdown("""
        **Current Configuration**
        """)
        
        # Display key settings
        settings_data = {
            'Data Directory': str(get_config_path('files.raw_data')),
            'Model Directory': str(get_config_path('model_artifacts.original.svm_model')),
        }
        
        for key, value in settings_data.items():
            st.write(f"- **{key}:** `{value}`")
        
        st.markdown("""
        **Advanced Options**
        - Model selection: SVM or XGBoost (merged only)
        - Classification approach: Original (42 cats) or Merged (13 cats)
        - Batch processing for multiple articles
        """)


def main():
    """Main dashboard application."""
    # Header with model selection
    approach, model_type = header_section()
    
    st.divider()
    
    # Load classifier
    try:
        classifier = load_classifier_cached(approach, model_type)
        if not classifier:
            st.error("Failed to load classifier. Please check if models are trained.")
            return
    except Exception as e:
        st.error(f"Error loading classifier: {e}")
        return
    
    # Main tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Predictor",
        "Batch",
        "Analytics",
        "Scraper",
        "Documentation"
    ])
    
    with tab1:
        tab1_predictor(classifier)
    
    with tab2:
        tab2_batch_processor(classifier)
    
    with tab3:
        tab3_model_analytics(approach, model_type, classifier)
    
    with tab4:
        tab4_live_scraper()
    
    with tab5:
        tab5_documentation()
    
    # Footer
    st.divider()
    st.markdown("""
    ---
    **Multi-Class News Classification Dashboard** | Built with Streamlit | 
    [GitHub](https://github.com) | Powered by scikit-learn, XGBoost, and NLTK
    """)


if __name__ == "__main__":
    main()
