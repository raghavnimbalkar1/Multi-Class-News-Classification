#!/usr/bin/env python3
"""
Launcher script for the Streamlit dashboard
Handles environment setup and starts the web application
"""

import subprocess
import sys
from pathlib import Path


def main():
    """Launch the Streamlit dashboard."""
    
    # Get the project root directory
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    app_file = script_dir / "app.py"
    
    print("=" * 70)
    print("Multi-Class News Classification - Streamlit Dashboard")
    print("=" * 70)
    print()
    
    # Check if Streamlit is installed
    try:
        import streamlit
        print(f"✓ Streamlit {streamlit.__version__} is installed")
    except ImportError:
        print("✗ Streamlit is not installed")
        print("\nInstalling Streamlit...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "streamlit>=1.28.0"
        ])
        print("✓ Streamlit installed successfully")
    
    # Check other dependencies
    dependencies = {
        'pandas': 'pandas',
        'numpy': 'numpy',
        'sklearn': 'scikit-learn',
        'matplotlib': 'matplotlib',
        'seaborn': 'seaborn',
    }
    
    missing = []
    for module, package in dependencies.items():
        try:
            __import__(module)
            print(f"✓ {package} is installed")
        except ImportError:
            print(f"✗ {package} is not installed")
            missing.append(package)
    
    if missing:
        print(f"\nInstalling missing dependencies: {', '.join(missing)}")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", *missing
        ])
        print("✓ All dependencies installed")
    
    print()
    print("=" * 70)
    print("Starting Streamlit dashboard...")
    print("=" * 70)
    print()
    print("The dashboard will open in your default browser.")
    print("If it doesn't, navigate to: http://localhost:8501")
    print()
    print("Press Ctrl+C to stop the server")
    print()
    
    # Start Streamlit app
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run",
            str(app_file),
            "--logger.level=info",
            "--client.showErrorDetails=true"
        ])
    except KeyboardInterrupt:
        print("\n\nShutdown complete. Goodbye!")
        sys.exit(0)


if __name__ == "__main__":
    main()
