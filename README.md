# 🏥 Health Risk Intelligence Platform

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

## 🔬 Clustering: Health Persona Discovery

### Overview
Unsupervised learning pipeline to discover distinct health personas from behavioral and lifestyle patterns. Focuses on **preventive health** by identifying modifiable risk factors rather than disease outcomes.

### Key Features
- **11 Behavioral Features**: Demographics, lifestyle factors (BMI, sleep, physical activity, smoking, alcohol), health engagement, and self-reported health status
- **Advanced Preprocessing**: 
  - PCA dimensionality reduction (14 components explaining 96.9% variance)
  - Yeo-Johnson power transformation for skewed features
  - Variance-based feature selection
- **Multi-Algorithm Clustering**: KMeans with automatic memory optimization for large datasets (246k+ samples)
- **Comprehensive Evaluation**: Silhouette Score (0.1202), Calinski-Harabasz, and Davies-Bouldin metrics
- **Optimal K Selection**: Automated selection of K=6 clusters via exhaustive search (K=3-7)

### Results
- **Best Configuration**: KMeans with K=6 clusters
- **Silhouette Score**: 0.1202 (improved from initial 0.097-0.104)
- **Dataset**: 246,022 samples processed
- **Outputs**: Cluster profiles, radar charts, geographic heatmaps, saved models

### Output Files
- `outputs/figures/optimal_k.png` - K selection visualization
- `outputs/figures/cluster_radar_charts.png` - Cluster profile comparisons
- `outputs/figures/cluster_by_state_heatmap.png` - Geographic distribution
- `data/processed/heart_with_clusters.csv` - Dataset with cluster assignments
- `models/kmeans_model.pkl` - Trained clustering model
- `models/pca_model.pkl` - PCA transformer
- `models/model_scaler.pkl` - Feature scaler

## 📝 TODO

- [x] Add data to `data/raw/`
- [x] Run cleaning pipeline
- [x] Implement clustering pipeline
- [ ] Train prediction models
- [ ] Test Streamlit app
- [ ] Create presentation

## 👤 Author

**Ayomide Isinkaye** - [GitHub](https://github.com/avisink)
