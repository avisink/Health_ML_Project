import os
from pathlib import Path
import subprocess
import sys

def create_directory_structure():
    """Create all project directories."""
    
    dirs = [
        "data/raw",
        "data/processed",
        "src",
        "notebooks",
        "models",
        "outputs/figures",
        "outputs/reports",
        "app/pages",
        "app/assets",
        "tests",
        "docs",
        "scripts"
    ]
    
    print("📁 Creating directory structure...")
    for directory in dirs:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"   ✓ {directory}/")
    
    # Create .gitkeep files for empty directories
    for directory in ["models", "outputs", "data/raw"]:
        gitkeep = Path(directory) / ".gitkeep"
        gitkeep.touch()


def create_gitignore():
    """Create .gitignore file."""
    
    gitignore_content = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
ENV/
build/
dist/
*.egg-info/

# Jupyter
.ipynb_checkpoints

# Data files (large)
data/raw/*.csv
data/raw/*.xlsx
data/processed/*.csv
*.h5

# Models (large)
models/*.pkl
models/*.joblib
*.pkl

# IDE
.vscode/
.idea/
*.swp
.DS_Store

# Outputs
outputs/figures/*.png
outputs/figures/*.pdf

# Environment
.env

# Logs
*.log
logs/
"""
    
    print("\n📝 Creating .gitignore...")
    with open(".gitignore", "w", encoding="utf-8") as f:
        f.write(gitignore_content)
    print("   ✓ .gitignore created")


def create_requirements():
    """Create requirements.txt file."""
    
    requirements = """# Core Data Science
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
scipy>=1.11.0

# Machine Learning
xgboost>=2.0.0
imbalanced-learn>=0.11.0

# Visualization
matplotlib>=3.7.0
seaborn>=0.12.0
plotly>=5.14.0

# Geospatial
geopandas>=0.13.0
folium>=0.14.0

# Explainability
shap>=0.42.0

# Clustering
umap-learn>=0.5.3

# Web App
streamlit>=1.28.0

# Utilities
joblib>=1.3.0
tqdm>=4.65.0
python-dotenv>=1.0.0
"""
    
    print("\n📦 Creating requirements.txt...")
    with open("requirements.txt", "w", encoding="utf-8") as f:
        f.write(requirements)
    print("   ✓ requirements.txt created")


def create_readme():
    """Create basic README.md."""
    
    readme = """# 🏥 Health Risk Intelligence Platform

**Predicting disease risk and discovering hidden health patterns**

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run pipeline
python src/cleaning.py
python src/clustering.py
python src/modeling.py

# Launch app
streamlit run app/app.py
```

## 📊 Project Structure

```
├── data/              # Datasets
├── src/               # Source code
├── models/            # Trained models
├── outputs/           # Results
├── app/               # Streamlit app
└── notebooks/         # Analysis notebooks
```

## 📝 TODO

- [ ] Add data to `data/raw/`
- [ ] Run cleaning pipeline
- [ ] Train models
- [ ] Test Streamlit app
- [ ] Create presentation

## 👤 Author

**Ayomide Isinkaye** - [GitHub](https://github.com/avisink)
"""
    
    print("\n📄 Creating README.md...")
    with open("README.md", "w", encoding="utf-8") as f:
        f.write(readme)
    print("   ✓ README.md created")


def create_config():
    """Create src/config.py file."""
    
    config = """\"\"\"
Configuration settings for the project.
\"\"\"

from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).parent.parent

# Data paths
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

RAW_HEART_DATA = RAW_DATA_DIR / "heart_2022.csv"
CLEANED_HEART_DATA = PROCESSED_DATA_DIR / "heart_cleaned.csv"
CLUSTERED_DATA = PROCESSED_DATA_DIR / "heart_with_clusters.csv"

# Model paths
MODELS_DIR = PROJECT_ROOT / "models"
DIABETES_MODEL = MODELS_DIR / "diabetes_xgboost.pkl"
HEART_MODEL = MODELS_DIR / "heart_xgboost.pkl"
CLUSTER_MODEL = MODELS_DIR / "kmeans_model.pkl"
SCALER = MODELS_DIR / "model_scaler.pkl"

# Output paths
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
REPORTS_DIR = OUTPUTS_DIR / "reports"

# Parameters
RANDOM_STATE = 42
TEST_SIZE = 0.2
N_CLUSTERS = 5

# Create directories
for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, FIGURES_DIR, REPORTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)
"""
    
    print("\n⚙️ Creating src/config.py...")
    with open("src/config.py", "w", encoding="utf-8") as f:
        f.write(config)
    
    # Create __init__.py
    with open("src/__init__.py", "w", encoding="utf-8") as f:
        f.write("")
    
    print("   ✓ src/config.py created")
    print("   ✓ src/__init__.py created")


def create_test_script():
    """Create quick test script."""
    
    test_script = """#!/usr/bin/env python3
\"\"\"Quick test to verify setup.\"\"\"

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
    
    print("\\n" + "="*60)
    if all_passed:
        print("✅ SETUP COMPLETE!")
        print("\\nNext steps:")
        print("  1. Add data to data/raw/")
        print("  2. Run: python src/cleaning.py")
        print("  3. Run: python src/clustering.py")
    else:
        print("❌ Setup incomplete")
    print("="*60)
"""
    
    print("\n🧪 Creating test script...")
    test_path = Path("scripts/test_setup.py")
    with open(test_path, "w", encoding="utf-8") as f:
        f.write(test_script)
    print("   ✓ scripts/test_setup.py created")


def initialize_git():
    """Initialize git repository."""
    
    if Path(".git").exists():
        print("\n🐙 Git already initialized")
        return
    
    print("\n🐙 Initializing git repository...")
    try:
        subprocess.run(["git", "init"], check=True, capture_output=True)
        subprocess.run(["git", "branch", "-M", "main"], check=True, capture_output=True)
        print("   ✓ Git initialized")
        print("   ✓ Default branch set to 'main'")
    except subprocess.CalledProcessError as e:
        print(f"   ⚠️ Git initialization failed: {e}")
        print("   → Install git first: https://git-scm.com/downloads")
    except FileNotFoundError:
        print("   ⚠️ Git not found")
        print("   → Install git first: https://git-scm.com/downloads")


def create_venv():
    """Create virtual environment."""
    
    if Path("venv").exists():
        print("\n🐍 Virtual environment already exists")
        return
    
    print("\n🐍 Creating virtual environment...")
    try:
        subprocess.run([sys.executable, "-m", "venv", "venv"], check=True)
        print("   ✓ Virtual environment created")
        
        # Determine activation command based on OS
        if sys.platform == "win32":
            activate_cmd = "venv\\Scripts\\activate"
        else:
            activate_cmd = "source venv/bin/activate"
        
        print(f"\n   Next: Activate with: {activate_cmd}")
        print(f"   Then: pip install -r requirements.txt")
    except Exception as e:
        print(f"   ⚠️ Virtual environment creation failed: {e}")


def print_next_steps():
    """Print what to do next."""
    
    print("\n" + "="*70)
    print("✅ PROJECT SETUP COMPLETE!")
    print("="*70)
    
    print("\n📋 Next Steps:")
    print("\n1️⃣  Activate virtual environment:")
    if sys.platform == "win32":
        print("    venv\\Scripts\\activate")
    else:
        print("    source venv/bin/activate")
    
    print("\n2️⃣  Install dependencies:")
    print("    pip install -r requirements.txt")
    
    print("\n3️⃣  Add your data:")
    print("    Copy heart_2022.csv → data/raw/")
    print("    Copy POP_EST2022.csv → data/raw/")
    
    print("\n4️⃣  Run test:")
    print("    python scripts/test_setup.py")
    
    print("\n5️⃣  Start working:")
    print("    python src/cleaning.py")
    
    print("\n6️⃣  Push to GitHub:")
    print("    git add .")
    print("    git commit -m 'Initial commit'")
    print("    git remote add origin <your-repo-url>")
    print("    git push -u origin main")
    
    print("\n" + "="*70)
    print("📚 Documentation:")
    print("   - README.md for overview")
    print("   - requirements.txt for dependencies")
    print("   - src/config.py for paths")
    print("="*70)


def main():
    """Run complete setup."""
    
    print("="*70)
    print("🚀 HEALTH ML PROJECT - AUTOMATED SETUP")
    print("="*70)
    print("\nThis will create a complete project structure.")
    
    # Check if already set up
    if Path("src/config.py").exists():
        response = input("\n⚠️  Project already set up. Overwrite? (y/N): ")
        if response.lower() != 'y':
            print("Aborted.")
            return
    
    # Run setup steps
    create_directory_structure()
    create_gitignore()
    create_requirements()
    create_readme()
    create_config()
    create_test_script()
    initialize_git()
    create_venv()
    
    # Final instructions
    print_next_steps()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Setup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Setup failed: {e}")
        sys.exit(1)