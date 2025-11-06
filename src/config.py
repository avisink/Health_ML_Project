"""
Configuration settings for the project.
"""

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
