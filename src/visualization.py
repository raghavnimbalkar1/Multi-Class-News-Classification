"""
Visualization utilities for Streamlit dashboard
Creates charts, plots, and reports for model analytics
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from config_loader import get_config_path


def plot_confusion_matrix(approach: str, model_type: str = 'svm') -> plt.Figure:
    """
    Load and plot confusion matrix for a model.
    
    Args:
        approach: 'original' or 'merged'
        model_type: 'svm' or 'xgboost'
        
    Returns:
        matplotlib figure
    """
    try:
        # Determine file suffix
        if approach == 'merged' and model_type == 'xgboost':
            filename = 'xgboost_confusion_matrix_numeric.csv'
        else:
            filename = 'confusion_matrix_numeric.csv'
        
        # Load matrix
        suffix = f'.{approach}'
        cm_path = get_config_path(f'model_artifacts{suffix}.confusion_matrix')
        cm_path = Path(cm_path).parent / filename
        
        cm = pd.read_csv(cm_path, index_col=0)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(cm, annot=False, fmt='d', cmap='Blues', ax=ax, 
                    cbar_kws={'label': 'Count'})
        ax.set_title(f'Confusion Matrix - {approach.title()} ({model_type.upper()})', 
                     fontsize=14, fontweight='bold')
        ax.set_ylabel('True Label', fontsize=12)
        ax.set_xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        
        return fig
    except Exception as e:
        # Return empty figure with error message
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, f'Error loading confusion matrix:\n{str(e)}', 
                ha='center', va='center', fontsize=12)
        return fig


def plot_accuracy_comparison() -> plt.Figure:
    """
    Create comparison chart of model accuracies.
    
    Returns:
        matplotlib figure
    """
    try:
        # Load classification reports
        original_path = get_config_path('model_artifacts.original.classification_report')
        merged_path = get_config_path('model_artifacts.merged.classification_report')
        xgboost_path = get_config_path('model_artifacts.merged.xgboost_classification_report')
        
        original_path = Path(original_path).parent / 'classification_report.csv'
        merged_path = Path(merged_path).parent / 'classification_report.csv'
        xgboost_path = Path(xgboost_path).parent / 'xgboost_classification_report.csv'
        
        # Load and extract accuracy
        original_df = pd.read_csv(original_path)
        merged_df = pd.read_csv(merged_path)
        xgboost_df = pd.read_csv(xgboost_path)
        
        # Extract macro avg accuracy (usually last row)
        original_acc = original_df.iloc[-1]['recall'] if 'recall' in original_df.columns else 0
        merged_acc = merged_df.iloc[-1]['recall'] if 'recall' in merged_df.columns else 0
        xgboost_acc = xgboost_df.iloc[-1]['recall'] if 'recall' in xgboost_df.columns else 0
        
        # Create bar plot
        models = ['Original\n(42 Categories)', 'Merged SVM\n(13 Categories)', 'Merged XGBoost\n(13 Categories)']
        accuracies = [original_acc, merged_acc, xgboost_acc]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        bars = ax.bar(models, accuracies, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        
        # Add value labels on bars
        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{acc:.2%}', ha='center', va='bottom', fontweight='bold', fontsize=11)
        
        ax.set_ylabel('Macro Avg Recall', fontsize=12, fontweight='bold')
        ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
        ax.set_ylim(0, 1.0)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        plt.tight_layout()
        
        return fig
    except Exception as e:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, f'Error loading accuracy data:\n{str(e)}', 
                ha='center', va='center', fontsize=12)
        return fig


def plot_top_categories_performance(approach: str) -> plt.Figure:
    """
    Plot per-category performance metrics.
    
    Args:
        approach: 'original' or 'merged'
        
    Returns:
        matplotlib figure
    """
    try:
        # Load classification report
        suffix = f'.{approach}'
        report_path = get_config_path(f'model_artifacts{suffix}.classification_report')
        report_path = Path(report_path).parent / 'classification_report.csv'
        
        df = pd.read_csv(report_path)
        
        # Remove summary rows
        df = df[~df.iloc[:, 0].isin(['accuracy', 'macro avg', 'weighted avg'])]
        
        # Sort by f1-score and get top 10
        if 'f1-score' in df.columns:
            df = df.sort_values('f1-score', ascending=True).tail(10)
            metric = 'f1-score'
        else:
            df = df.sort_values('recall', ascending=True).tail(10)
            metric = 'recall'
        
        # Create horizontal bar chart
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.barh(range(len(df)), df[metric], color='steelblue', alpha=0.8, edgecolor='black')
        ax.set_yticks(range(len(df)))
        ax.set_yticklabels(df.iloc[:, 0], fontsize=9)
        ax.set_xlabel(f'{metric.title()}', fontsize=11, fontweight='bold')
        ax.set_title(f'Top 10 Categories by {metric.title()} - {approach.title()}', 
                     fontsize=12, fontweight='bold')
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        plt.tight_layout()
        
        return fig
    except Exception as e:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, f'Error loading category performance:\n{str(e)}', 
                ha='center', va='center', fontsize=12)
        return fig


def create_model_summary_table(approach: str, model_type: str = 'svm') -> pd.DataFrame:
    """
    Create summary metrics table for a model.
    
    Args:
        approach: 'original' or 'merged'
        model_type: 'svm' or 'xgboost'
        
    Returns:
        DataFrame with summary metrics
    """
    try:
        # Determine file path
        if approach == 'merged' and model_type == 'xgboost':
            filename = 'xgboost_classification_report.csv'
        else:
            filename = 'classification_report.csv'
        
        # Load report
        suffix = f'.{approach}'
        report_path = get_config_path(f'model_artifacts{suffix}.classification_report')
        report_path = Path(report_path).parent / filename
        
        df = pd.read_csv(report_path)
        
        # Extract summary metrics (macro avg row)
        summary = df[df.iloc[:, 0] == 'macro avg']
        
        if summary.empty:
            summary = df.iloc[-2:]  # Last 2 rows
        
        # Rename for display
        summary = summary.copy()
        summary.columns = ['Metric', 'Precision', 'Recall', 'F1-Score', 'Support (N/A)']
        
        return summary
    except Exception as e:
        return pd.DataFrame({'Error': [str(e)]})


def get_category_mapping(approach: str) -> dict:
    """
    Get category mapping from config.
    
    Args:
        approach: 'original' or 'merged'
        
    Returns:
        Dictionary mapping category codes to names
    """
    try:
        from config_loader import config
        suffix = f'.{approach}'
        
        # Try to get from category_mapping
        mapping_key = f'category_mapping{suffix}'
        category_map = config.get(mapping_key)
        
        if category_map:
            return category_map
        
        # Fallback: return empty dict
        return {}
    except Exception:
        return {}
