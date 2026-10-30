# 🏥 Health Risk Intelligence Platform

**Predicting disease risk and discovering hidden health patterns**

### 🔗 [Live app → avisink-health-analysis-with-ml.streamlit.app](https://avisink-health-analysis-with-ml.streamlit.app/)

An interactive Streamlit dashboard built on the CDC BRFSS 2022 survey (246,022 respondents after cleaning):

- **📊 Overview:** disease prevalence and lifestyle patterns for any filtered population
- **🎯 Risk Calculator:** compare your profile to similar respondents and see "what if" lifestyle scenarios
- **🧬 Health Personas:** find the ML cluster you're closest to, see yourself on a PCA map, and dig into the model
- **🗺️ Geography:** U.S. choropleth map plus state-by-state rankings
- **📈 Trends:** how health changes with age

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run pipeline
python src/cleaning.py
python src/clustering.py
python src/modeling.py

# Launch app
streamlit run app/dashboard.py
```

## 📊 Project Structure

```
├── data/              # Datasets
├── src/               # Source code
├── models/            # Trained models
├── outputs/           # Results
├── app/               # Streamlit app (dashboard.py, cluster_explorer.py, ui.py design system)
└── notebooks/         # Analysis notebooks
```

## 🔬 Clustering: Health Persona Discovery

### Overview
Unsupervised learning (`src/clustering.py`) groups respondents into **health personas** from 14 features:
age, BMI, sleep, physical activity, smoking, alcohol, self-rated health, physically/mentally unwell days,
time since last checkup, flu shot, and diabetes / heart attack / depression history.

### Method
- Median imputation → `StandardScaler`
- `KMeans`, with K chosen by best silhouette score over K=3–8 → **K=7**
- Silhouette ≈ 0.13 (8k-row sample), Davies-Bouldin ≈ 1.94

### The personas
| Persona | Share | What defines it |
|---|---|---|
| High Symptom Burden | 8% | ~26 physically unwell days/month, lowest self-rated health, least active |
| Mental Health Focus | 13% | ~99% depression diagnosis, but mostly still active |
| Overdue for a Checkup | 9% | Younger, healthy, no checkup in 2+ years |
| Healthy, Skipping the Flu Shot | 25% | Active, healthy, no flu vaccine |
| Managing Diabetes | 10% | Older, highest BMI, ~96% diabetes |
| Thriving & Engaged | 29% | Active, healthy, fewest smokers, all vaccinated |
| Heart Attack Survivors | 5% | Oldest; all have had a heart attack |

### In the app (🧬 Health Personas tab)
- **Your persona:** enter your profile and get matched to the nearest centroid (same rule as `KMeans.predict`), with a radar chart of you vs. your persona vs. everyone and a closeness score for every persona
- **Persona map:** 2D/3D PCA projection of 8,000 respondents with persona centers and a star marking you
- **Under the hood:** centroid z-score heatmap, per-cluster silhouette distributions, PCA loadings, your per-feature distance breakdown, and downloadable profiles

### Known limitations
Binary diagnosis flags are standardised alongside continuous features, so a single yes/no can dominate distance.
Three personas are largely defined by one diagnosis, and the low silhouette means boundaries are fuzzy.

### Output Files
- `data/processed/heart_with_clusters.csv`: dataset with cluster assignments (used by the app)
- `outputs/reports/cluster_profiles.csv`: per-cluster summary stats
- `outputs/presentation_visuals/`: static charts from `src/simple_analysis.py`

## 📝 TODO

- [x] Add data to `data/raw/`
- [x] Run cleaning pipeline
- [x] Implement clustering pipeline
- [ ] Train prediction models
- [x] Build & deploy Streamlit app
- [x] Interactive cluster / persona explorer
- [ ] Create presentation

## 👤 Author

**Ayomide Isinkaye** - [GitHub](https://github.com/avisink)
