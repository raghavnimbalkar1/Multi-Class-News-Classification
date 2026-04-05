"""
Multi-Class News Classification - Inference & Prediction Utilities
Provides functions for classifying news articles using trained models
"""

import pickle
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, Optional

sys.path.insert(0, str(Path(__file__).parent))
from config_loader import get_config_path


class NewsClassifier:
    """Main classifier for news articles."""
    
    def __init__(self, approach: str = "merged", model_type: str = "svm"):
        """
        Initialize the classifier.
        
        Args:
            approach: 'original' (42 categories) or 'merged' (13 categories)
            model_type: 'svm' or 'xgboost' (only for merged approach)
        """
        self.approach = approach
        self.model_type = model_type
        self.tfidf = None
        self.selector = None
        self.model = None
        self.encoder = None
        self.classes = None
        
        self._load_models()
    
    def _load_models(self):
        """Load all necessary model artifacts."""
        suffix = f".{self.approach}"
        
        try:
            # Load TF-IDF vectorizer
            tfidf_path = get_config_path(f'model_artifacts{suffix}.tfidf_vectorizer')
            with open(tfidf_path, 'rb') as f:
                self.tfidf = pickle.load(f)
            
            # Load feature selector
            selector_path = get_config_path(f'model_artifacts{suffix}.chi2_selector')
            with open(selector_path, 'rb') as f:
                self.selector = pickle.load(f)
            
            # Load classifier
            if self.approach == "original" or self.model_type == "svm":
                model_path = get_config_path(f'model_artifacts{suffix}.svm_model')
            else:  # merged + xgboost
                model_path = get_config_path(f'model_artifacts{suffix}.xgboost_model')
            
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            
            # For merged approach with xgboost, load encoder
            if self.approach == "merged" and self.model_type == "xgboost":
                encoder_path = get_config_path(f'model_artifacts{suffix}.label_encoder')
                with open(encoder_path, 'rb') as f:
                    self.encoder = pickle.load(f)
                self.classes = self.encoder.classes_
            else:
                self.classes = self.model.classes_
                
        except Exception as e:
            raise RuntimeError(f"Failed to load models: {e}")
    
    def predict(self, text: str) -> Dict:
        """
        Predict news category for given text.
        
        Args:
            text: News headline or article text
            
        Returns:
            Dictionary with prediction results
        """
        if not text or not isinstance(text, str):
            return {
                'success': False,
                'error': 'Invalid text input',
                'prediction': None,
                'confidence': None,
                'probabilities': None
            }
        
        try:
            # Vectorize
            X_tfidf = self.tfidf.transform([text])
            
            # Select features
            X_selected = self.selector.transform(X_tfidf)
            
            # Predict
            if self.approach == "merged" and self.model_type == "xgboost":
                # XGBoost predicts encoded labels
                pred_encoded = self.model.predict(X_selected)
                prediction = self.encoder.inverse_transform(pred_encoded)[0]
            else:
                # LinearSVC predicts directly
                prediction = self.model.predict(X_selected)[0]
            
            # Get confidence scores
            if hasattr(self.model, 'decision_function'):
                # LinearSVC has decision_function
                decision_scores = self.model.decision_function(X_selected)[0]
                probabilities = self._softmax(decision_scores)
            else:
                # XGBoost has predict_proba
                probabilities = self.model.predict_proba(X_selected)[0]
            
            # Create probability dict
            prob_dict = {
                self.classes[i]: float(probabilities[i])
                for i in range(len(self.classes))
            }
            
            # Get top probabilities
            sorted_probs = sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)[:5]
            
            return {
                'success': True,
                'prediction': prediction,
                'confidence': float(prob_dict[prediction]),
                'probabilities': dict(sorted_probs),
                'all_classes': list(self.classes),
                'text_preview': text[:100] + '...' if len(text) > 100 else text
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'prediction': None,
                'confidence': None,
                'probabilities': None
            }
    
    def batch_predict(self, texts: list) -> pd.DataFrame:
        """
        Predict categories for multiple texts.
        
        Args:
            texts: List of text strings
            
        Returns:
            DataFrame with predictions
        """
        results = []
        for text in texts:
            result = self.predict(text)
            results.append({
                'text': text[:100] + '...' if len(text) > 100 else text,
                'prediction': result.get('prediction', 'ERROR'),
                'confidence': result.get('confidence', 0),
                'success': result.get('success', False)
            })
        
        return pd.DataFrame(results)
    
    @staticmethod
    def _softmax(x):
        """Compute softmax of decision scores."""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()
    
    def get_classes(self) -> list:
        """Get list of all classification categories."""
        return list(self.classes)


def get_classifier(approach: str = "merged", model_type: str = "svm") -> Optional[NewsClassifier]:
    """
    Factory function to get a classifier instance.
    
    Args:
        approach: 'original' or 'merged'
        model_type: 'svm' or 'xgboost'
        
    Returns:
        NewsClassifier instance or None if loading fails
    """
    try:
        return NewsClassifier(approach=approach, model_type=model_type)
    except Exception as e:
        print(f"Error loading classifier: {e}")
        return None
