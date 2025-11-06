#!/usr/bin/env python3
"""Quick test to verify setup."""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

def test_imports():
    try:
        import pandas
        import numpy
        import sklearn
        import xgboost
        import streamlit
        import shap
        print("✅ All packages imported")
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

def test_structure():
    required = ["data", "src", "models", "outputs", "app"]
    for directory in required:
        if Path(directory).exists():
            print(f"✅ {directory}/ exists")
        else:
            print(f"❌ {directory}/ missing")
            return False
    return True

if __name__ == "__main__":
    print("="*60)
    print("🧪 TESTING PROJECT SETUP")
    print("="*60)
    
    all_passed = test_imports() and test_structure()
    
    print("\n" + "="*60)
    if all_passed:
        print("✅ SETUP COMPLETE!")
        print("\nNext steps:")
        print("  1. Add data to data/raw/")
        print("  2. Run: python src/cleaning.py")
        print("  3. Run: python src/clustering.py")
    else:
        print("❌ Setup incomplete")
    print("="*60)
