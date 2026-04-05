"""
Model Analysis and Comparison Utilities
Compares performance across different approaches and models
Generates comprehensive reports and visualizations
"""

import os
import sys
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple
import logging

sys.path.insert(0, str(Path(__file__).parent))
from config_loader import get_config_path

logger = logging.getLogger(__name__)


# ============================================================================
# MODEL LOADING UTILITIES
# ============================================================================

class ModelAnalyzer:
    """Analyze and compare trained models."""
    
    def __init__(self, log_enabled: bool = True):
        """Initialize the analyzer."""
        self.log_enabled = log_enabled
        self.models = {}
        self.reports = {}
        self.visualizations_dir = "visualizations"
        os.makedirs(self.visualizations_dir, exist_ok=True)
    
    def _log(self, message: str):
        """Log if enabled."""
        if self.log_enabled:
            print(f"[ModelAnalyzer] {message}")
    
    def load_classification_report(self, report_path: str) -> Dict:
        """Load classification report from CSV."""
        try:
            df = pd.read_csv(report_path, index_col=0)
            return df.to_dict()
        except Exception as e:
            self._log(f"Error loading report from {report_path}: {e}")
            return {}
    
    def load_confusion_matrix(self, matrix_path: str) -> np.ndarray:
        """Load confusion matrix from CSV."""
        try:
            df = pd.read_csv(matrix_path, index_col=0)
            return df.values
        except Exception as e:
            self._log(f"Error loading matrix from {matrix_path}: {e}")
            return np.array([])
    
    def load_all_reports(self):
        """Load all available reports for comparison."""
        self._log("Loading all reports...")
        
        approaches = ["original", "merged"]
        
        for approach in approaches:
            self.reports[approach] = {}
            
            # Load classification reports
            try:
                report_path = get_config_path(f'model_artifacts.{approach}.classification_report')
                self.reports[approach]['classification'] = self.load_classification_report(report_path)
                self._log(f"Loaded {approach} classification report")
            except Exception as e:
                self._log(f"Could not load {approach} classification report: {e}")
            
            # For merged, also load XGBoost report
            if approach == "merged":
                try:
                    xgb_report_path = get_config_path('model_artifacts.merged.xgboost_classification_report')
                    self.reports[approach]['xgboost'] = self.load_classification_report(xgb_report_path)
                    self._log("Loaded XGBoost classification report")
                except Exception as e:
                    self._log(f"Could not load XGBoost report: {e}")
    
    def extract_accuracy_comparison(self) -> pd.DataFrame:
        """Extract and compare accuracy across models."""
        accuracies = {}
        
        if 'original' in self.reports and 'classification' in self.reports['original']:
            report = self.reports['original']['classification']
            if 'accuracy' in report:
                accuracies['Original (42 categories)'] = report['accuracy'].get('precision', 0)
        
        if 'merged' in self.reports:
            if 'classification' in self.reports['merged']:
                report = self.reports['merged']['classification']
                if 'accuracy' in report:
                    accuracies['Merged SVM (13 categories)'] = report['accuracy'].get('precision', 0)
            
            if 'xgboost' in self.reports['merged']:
                report = self.reports['merged']['xgboost']
                if 'accuracy' in report:
                    accuracies['Merged XGBoost (13 categories)'] = report['accuracy'].get('precision', 0)
        
        return pd.DataFrame(list(accuracies.items()), columns=['Model', 'Accuracy'])
    
    def generate_accuracy_comparison_chart(self):
        """Generate accuracy comparison visualization."""
        df = self.extract_accuracy_comparison()
        
        if df.empty:
            self._log("No accuracy data available for comparison")
            return
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(df['Model'], df['Accuracy'], color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
        plt.ylabel('Accuracy', fontsize=12)
        plt.title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.ylim(0, 1)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.4f}',
                    ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        output_path = os.path.join(self.visualizations_dir, 'accuracy_comparison.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        self._log(f"Saved accuracy comparison chart to {output_path}")
        plt.close()
    
    def extract_per_class_metrics(self) -> Dict:
        """Extract per-class metrics for all models."""
        metrics = {}
        
        if 'original' in self.reports and 'classification' in self.reports['original']:
            report = self.reports['original']['classification']
            metrics['Original (42 categories)'] = report
        
        if 'merged' in self.reports and 'classification' in self.reports['merged']:
            report = self.reports['merged']['classification']
            metrics['Merged SVM (13 categories)'] = report
        
        if 'merged' in self.reports and 'xgboost' in self.reports['merged']:
            report = self.reports['merged']['xgboost']
            metrics['Merged XGBoost (13 categories)'] = report
        
        return metrics
    
    def extract_macro_metrics(self) -> pd.DataFrame:
        """Extract macro-averaged metrics (precision, recall, f1-score)."""
        macro_metrics = {}
        
        for model_name, metrics in self.extract_per_class_metrics().items():
            if 'macro avg' in metrics:
                macro_metrics[model_name] = metrics['macro avg']
            else:
                # Try alternative keys
                for key in metrics:
                    if 'macro' in key.lower() or 'avg' in key.lower():
                        macro_metrics[model_name] = metrics[key]
                        break
        
        if macro_metrics:
            return pd.DataFrame(macro_metrics).T
        return pd.DataFrame()
    
    def generate_comparison_report(self) -> str:
        """Generate a text-based comparison report."""
        report_text = []
        report_text.append("="*80)
        report_text.append("MULTI-CLASS NEWS CLASSIFICATION - MODEL COMPARISON REPORT")
        report_text.append("="*80)
        report_text.append("")
        
        # Load reports
        self.load_all_reports()
        
        # Accuracy comparison
        report_text.append("1. ACCURACY COMPARISON")
        report_text.append("-" * 80)
        accuracy_df = self.extract_accuracy_comparison()
        if not accuracy_df.empty:
            report_text.append(accuracy_df.to_string(index=False))
            report_text.append("")
        
        # Macro metrics
        report_text.append("2. MACRO-AVERAGED METRICS")
        report_text.append("-" * 80)
        macro_df = self.extract_macro_metrics()
        if not macro_df.empty:
            report_text.append(macro_df.to_string())
            report_text.append("")
        
        # Per-class metrics summary
        report_text.append("3. PER-CLASS METRICS SUMMARY")
        report_text.append("-" * 80)
        
        for model_name, metrics in self.extract_per_class_metrics().items():
            report_text.append(f"\n{model_name}:")
            report_text.append("-" * 40)
            
            # Extract precision, recall, f1-score for each class
            class_metrics = {k: v for k, v in metrics.items() 
                           if k not in ['accuracy', 'macro avg', 'weighted avg']}
            
            if class_metrics:
                df = pd.DataFrame(class_metrics).T
                report_text.append(df.to_string())
            report_text.append("")
        
        report_text.append("="*80)
        report_text.append("END OF REPORT")
        report_text.append("="*80)
        
        return "\n".join(report_text)
    
    def save_comparison_report(self, filename: str = "model_comparison.txt"):
        """Generate and save comparison report."""
        report_text = self.generate_comparison_report()
        
        with open(filename, 'w') as f:
            f.write(report_text)
        
        self._log(f"Comparison report saved to {filename}")
        return report_text
    
    def generate_all_visualizations(self):
        """Generate all comparison visualizations."""
        self._log("Generating visualizations...")
        self.load_all_reports()
        self.generate_accuracy_comparison_chart()
        self._log("Visualizations generation complete")


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """Run model analysis and generate reports."""
    analyzer = ModelAnalyzer(log_enabled=True)
    
    # Generate and save comparison report
    report_text = analyzer.save_comparison_report(
        "model_comparison_report.txt"
    )
    print("\n" + report_text)
    
    # Generate visualizations
    analyzer.generate_all_visualizations()


if __name__ == "__main__":
    main()
