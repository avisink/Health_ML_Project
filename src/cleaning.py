import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))
from src.config import RAW_HEART_DATA, RAW_POPULATION_DATA, CLEANED_HEART_DATA

def clean_strings(heart):
    """Trim whitespace and lowercase."""
    for c in heart.select_dtypes(include=["object"]).columns:
        heart[c] = heart[c].astype(str).str.strip().str.lower()
    return heart

def apply_mappings(heart):
    """Applying all categorical mappings."""
    mappings = {
        "GeneralHealth": {
        "poor": 1, "fair": 2, "good": 3, "very good": 4, "excellent": 5
    },
    "HadDiabetes": {
        "no": 0,
        "no, pre-diabetes or borderline diabetes": 1,
        "yes, but only during pregnancy (female)": 2,
        "yes": 3
    },
    "CovidPos": {
        "yes": 1,
        "tested positive using home test without a health professional": 1,
        "no": 0
    },
    "TetanusLast10Tdap": {
        "no, did not receive any tetanus shot in the past 10 years": "none",
        "yes, received tetanus shot but not sure what type": "unsure",
        "yes, received tdap": "tdap",
        "yes, received tetanus shot, but not tdap": "tetanus_only"
    },
    "RemovedTeeth": {
        "none of them": 0,
        "1 to 5": 1,
        "6 or more, but not all": 2,
        "all": 3
    },
    "SmokerStatus": {
        "never smoked": 0,
        "former smoker": 1,
        "current smoker - now smokes some days": 2,
        "current smoker - now smokes every day": 3
    },
    "LastCheckupTime": {
        "within past year (anytime less than 12 months ago)": 0,
        "within past 2 years (1 year but less than 2 years ago)": 1,
        "within past 5 years (2 years but less than 5 years ago)": 2,
        "5 or more years ago": 3
    },
    "ECigaretteUsage": {
        "never used e-cigarettes in my entire life": 0,
        "not at all (right now)": 1,
        "use them some days": 2,
        "use them every day": 3
    },
    "RaceEthnicityCategory": {
        "white only, non-hispanic": 0,
        "hispanic": 1,
        "black only, non-hispanic": 2,
        "other race only, non-hispanic": 3,
        "multiracial, non-hispanic": 4
    },
    "AgeCategory": {
        "age 18 to 24": 0,
        "age 25 to 29": 1,
        "age 30 to 34": 2,
        "age 35 to 39": 3,
        "age 40 to 44": 4,
        "age 45 to 49": 5,
        "age 50 to 54": 6,
        "age 55 to 59": 7,
        "age 60 to 64": 8,
        "age 65 to 69": 9,
        "age 70 to 74": 10,
        "age 75 to 79": 11,
        "age 80 or older": 12
        }
    }
    
    for col, mapping in mappings.items():
        heart[col] = heart[col].map(mapping)
    
    return heart

def binary_mappings(heart):
    binary_map ={
        "yes": 1,
        "no": 0
    }

    binary_cols = [
    'PhysicalActivities','HadHeartAttack','HadAngina','HadStroke','HadAsthma','HadSkinCancer',
    'HadCOPD','HadDepressiveDisorder','HadKidneyDisease','HadArthritis',
    'DeafOrHardOfHearing','BlindOrVisionDifficulty','DifficultyConcentrating','DifficultyWalking',
    'DifficultyDressingBathing','DifficultyErrands','ChestScan','AlcoholDrinkers','HIVTesting','FluVaxLast12',
    'PneumoVaxEver','HighRiskLastYear'
    ]

    for col in binary_cols:
        if col in heart.columns:
            heart[col] = heart[col].map(binary_map)
    return heart


def main():
    """Main cleaning pipeline."""
    print("="*80)
    print("DATA CLEANING PIPELINE")
    print("="*80)
    
    # Load
    print("\n📂 Loading data...")
    heart = pd.read_csv(RAW_HEART_DATA)
    popbystate_22 = pd.read_csv(RAW_POPULATION_DATA)
    print(f"✓ Loaded {len(heart):,} samples")
    
    # Clean
    print("\n🧹 Cleaning...")
    heart = clean_strings(heart)
    heart = apply_mappings(heart)
    heart = binary_mappings(heart)
    
    # to check for realistic values, i will be checking the maximum values of
    # fields with numerical value
    for col in heart.select_dtypes(include=["float64"]).columns:
        print(f"{col}: {heart[col].max()}")

    # print(heart.head())
    for col in heart.columns:
        print(heart[col].value_counts())
        
    # Save
    print("\n💾 Saving cleaned data...")
    heart.to_csv(CLEANED_HEART_DATA, index=False)
    print(f"✓ Saved to {CLEANED_HEART_DATA}")
    
    return heart

if __name__ == "__main__":
    heart = main()