"""
Unified Pipeline Orchestration Script
Coordinates all steps: preprocessing, feature engineering, training, and evaluation
Supports both original (42 categories) and merged (13 categories) approaches
"""

import sys
import os
import time
import logging
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))
from config_loader import get_config_path, ensure_dir


# ============================================================================
# LOGGING SETUP
# ============================================================================

def setup_logging(log_dir: str = "logs") -> logging.Logger:
    """Set up logging with both file and console handlers."""
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"pipeline_{timestamp}.log")
    
    logger = logging.getLogger("Pipeline")
    logger.setLevel(logging.DEBUG)
    
    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


logger = setup_logging()


# ============================================================================
# PIPELINE STEPS
# ============================================================================

def run_step(step_name: str, script_path: str, description: str) -> Tuple[bool, str]:
    """
    Run a pipeline step and return success status.
    
    Args:
        step_name: Name of the step (e.g., "Original Preprocessing")
        script_path: Path to the Python script to run
        description: Description of what the step does
    
    Returns:
        Tuple of (success: bool, message: str)
    """
    logger.info(f"\n{'='*70}")
    logger.info(f"Starting: {step_name}")
    logger.info(f"Description: {description}")
    logger.info(f"Script: {script_path}")
    logger.info(f"{'='*70}\n")
    
    try:
        # Check if script exists
        if not os.path.exists(script_path):
            msg = f"Script not found: {script_path}"
            logger.error(msg)
            return False, msg
        
        # Run the script
        start_time = time.time()
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour timeout
        )
        elapsed_time = time.time() - start_time
        
        # Log output
        if result.stdout:
            logger.info("STDOUT:\n" + result.stdout)
        if result.stderr:
            logger.warning("STDERR:\n" + result.stderr)
        
        # Check result
        if result.returncode == 0:
            msg = f"✓ {step_name} completed successfully in {elapsed_time:.2f}s"
            logger.info(msg)
            return True, msg
        else:
            msg = f"✗ {step_name} failed with return code {result.returncode}"
            logger.error(msg)
            return False, msg
            
    except subprocess.TimeoutExpired:
        msg = f"✗ {step_name} timed out (exceeded 1 hour)"
        logger.error(msg)
        return False, msg
    except Exception as e:
        msg = f"✗ {step_name} raised exception: {str(e)}"
        logger.error(msg)
        return False, msg


def run_original_pipeline() -> bool:
    """Run the original 42-category classification pipeline."""
    logger.info("\n" + "="*70)
    logger.info("ORIGINAL APPROACH (42 Categories)")
    logger.info("="*70)
    
    steps = [
        ("Original Preprocessing", 
         "src/original/preprocessing.py",
         "Clean and preprocess raw news data"),
        
        ("Original Feature Engineering",
         "src/original/feature_engineering.py",
         "TF-IDF vectorization and feature selection"),
        
        ("Original Model Training",
         "src/original/train_model.py",
         "Train LinearSVC classifier"),
        
        ("Original Model Evaluation",
         "src/original/evaluate.py",
         "Generate evaluation metrics and reports"),
    ]
    
    for step_name, script_path, description in steps:
        success, message = run_step(step_name, script_path, description)
        if not success:
            logger.error(f"Pipeline stopped at {step_name}")
            return False
        logger.info(message)
    
    return True


def run_merged_pipeline() -> bool:
    """Run the merged 13-category classification pipeline."""
    logger.info("\n" + "="*70)
    logger.info("MERGED APPROACH (13 SuperClasses)")
    logger.info("="*70)
    
    steps = [
        ("Merged Preprocessing",
         "src/merged/preprocessing.py",
         "Merge 42 categories into 13 and preprocess data"),
        
        ("Merged Feature Engineering",
         "src/merged/feature_engineering.py",
         "TF-IDF vectorization and feature selection"),
        
        ("Merged SVM Training",
         "src/merged/train_model.py",
         "Train LinearSVC classifier on merged categories"),
        
        ("Merged XGBoost Training",
         "src/merged/train_xgboost.py",
         "Train XGBoost classifier on merged categories"),
        
        ("Merged SVM Evaluation",
         "src/merged/evaluate.py",
         "Generate evaluation metrics for SVM"),
        
        ("Merged XGBoost Evaluation",
         "src/merged/evaluate_xgboost.py",
         "Generate evaluation metrics for XGBoost"),
    ]
    
    for step_name, script_path, description in steps:
        success, message = run_step(step_name, script_path, description)
        if not success:
            logger.error(f"Pipeline stopped at {step_name}")
            return False
        logger.info(message)
    
    return True


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main(approach: str = "all", skip_steps: Optional[list] = None):
    """
    Run the complete pipeline or specific components.
    
    Args:
        approach: One of ["all", "original", "merged"]
        skip_steps: List of step names to skip (for resuming interrupted runs)
    """
    logger.info(f"\n{'#'*70}")
    logger.info(f"# MULTI-CLASS NEWS CLASSIFICATION PIPELINE")
    logger.info(f"# Approach: {approach.upper()}")
    logger.info(f"# Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"{'#'*70}\n")
    
    # Ensure model directories exist
    ensure_dir('paths.models.base')
    ensure_dir('paths.models.original')
    ensure_dir('paths.models.merged')
    
    all_success = True
    pipeline_results = {}
    
    try:
        # Run original pipeline
        if approach in ["all", "original"]:
            success = run_original_pipeline()
            pipeline_results["original"] = success
            all_success = all_success and success
        
        # Run merged pipeline
        if approach in ["all", "merged"]:
            success = run_merged_pipeline()
            pipeline_results["merged"] = success
            all_success = all_success and success
        
    except KeyboardInterrupt:
        logger.warning("\n\nPipeline interrupted by user")
        all_success = False
    except Exception as e:
        logger.error(f"\n\nUnexpected error in pipeline: {str(e)}")
        all_success = False
    
    # Print summary
    logger.info(f"\n{'='*70}")
    logger.info("PIPELINE SUMMARY")
    logger.info(f"{'='*70}")
    
    for approach_name, success in pipeline_results.items():
        status = "✓ COMPLETED" if success else "✗ FAILED"
        logger.info(f"{approach_name.upper()}: {status}")
    
    if all_success:
        logger.info("\n✓ All pipeline steps completed successfully!")
        logger.info(f"\nResults are saved in:")
        logger.info(f"  - Original models: {get_config_path('paths.models.original')}")
        logger.info(f"  - Merged models: {get_config_path('paths.models.merged')}")
    else:
        logger.error("\n✗ Pipeline completed with errors. Check logs above for details.")
    
    logger.info(f"{'='*70}\n")
    
    return 0 if all_success else 1


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Run the complete news classification pipeline"
    )
    parser.add_argument(
        "--approach",
        choices=["all", "original", "merged"],
        default="all",
        help="Which pipeline to run (default: all)"
    )
    parser.add_argument(
        "--skip",
        nargs="+",
        help="Specific steps to skip"
    )
    
    args = parser.parse_args()
    exit_code = main(approach=args.approach, skip_steps=args.skip)
    sys.exit(exit_code)
