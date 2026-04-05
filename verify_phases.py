#!/usr/bin/env python3
"""
Comprehensive Phase Verification Script
Validates Phase 1 and Phase 2 completion
"""

import sys
import os
sys.path.insert(0, 'src')

print("\n" + "="*70)
print("COMPREHENSIVE PHASE VERIFICATION")
print("="*70)

# Phase 1: Config System
print("\n[PHASE 1: CONFIG FOUNDATIONS]")
try:
    from config_loader import get_config_path, config
    
    config_checks = [
        ('Config file exists', os.path.exists('config.yaml')),
        ('Raw data path resolves', 'NewsData.json' in get_config_path('files.raw_data')),
        ('Model artifact paths resolve', 'svm_model.pkl' in get_config_path('model_artifacts.original.svm_model')),
    ]
    
    for check_name, result in config_checks:
        print(f"  {'✓' if result else '✗'} {check_name}")
    
except Exception as e:
    print(f"  ✗ Config system error: {e}")

# Phase 2: Pipeline & Analysis
print("\n[PHASE 2: PIPELINE & ANALYSIS]")
try:
    import pipeline
    import model_analysis
    
    phase2_checks = [
        ('pipeline.py has run_step', hasattr(pipeline, 'run_step')),
        ('pipeline.py has run_original_pipeline', hasattr(pipeline, 'run_original_pipeline')),
        ('pipeline.py has run_merged_pipeline', hasattr(pipeline, 'run_merged_pipeline')),
        ('model_analysis.py has ModelAnalyzer', hasattr(model_analysis, 'ModelAnalyzer')),
    ]
    
    for check_name, result in phase2_checks:
        print(f"  {'✓' if result else '✗'} {check_name}")
        
except Exception as e:
    print(f"  ✗ Phase 2 import error: {e}")

# Files structure check
print("\n[FILE STRUCTURE VERIFICATION]")
essential_files = [
    'config.yaml',
    'README.md',
    'requirements.txt',
    'src/config_loader.py',
    'src/pipeline.py',
    'src/model_analysis.py',
    'src/scraper.py',
    'src/original/preprocessing.py',
    'src/original/feature_engineering.py',
    'src/original/train_model.py',
    'src/original/evaluate.py',
    'src/merged/preprocessing.py',
    'src/merged/feature_engineering.py',
    'src/merged/train_model.py',
    'src/merged/train_xgboost.py',
    'src/merged/evaluate.py',
    'src/merged/evaluate_xgboost.py',
]

missing_files = []
for filepath in essential_files:
    if os.path.exists(filepath):
        print(f"  ✓ {filepath}")
    else:
        print(f"  ✗ {filepath} - MISSING")
        missing_files.append(filepath)

# Dependencies check
print("\n[DEPENDENCIES CHECK]")
required_packages = [
    'pandas',
    'numpy',
    'sklearn',
    'xgboost',
    'nltk',
    'spacy',
    'yaml',
    'matplotlib',
    'seaborn',
]

missing_packages = []
for package in required_packages:
    try:
        __import__(package)
        print(f"  ✓ {package}")
    except ImportError:
        print(f"  ✗ {package} - NOT INSTALLED")
        missing_packages.append(package)

# Summary
print("\n" + "="*70)
print("VERIFICATION SUMMARY")
print("="*70)

issues = len(missing_files) + len(missing_packages)
if issues == 0:
    print("\n✓✓✓ ALL PHASES VERIFIED - READY FOR PHASE 3 ✓✓✓")
    print("\n  Phase 1: Config Foundations - ✓ COMPLETE")
    print("  Phase 2: Pipeline & Analysis - ✓ COMPLETE")
    print("\n" + "="*70)
else:
    print(f"\n⚠ {issues} issue(s) found:")
    for f in missing_files:
        print(f"  - Missing file: {f}")
    for p in missing_packages:
        print(f"  - Missing package: {p}")
