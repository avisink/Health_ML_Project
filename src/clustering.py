"""
CLUSTERING PIPELINE - Week 1 Days 3-5
Goal: Discover 5 health personas for killer presentation visuals
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, RobustScaler, PowerTransformer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.feature_selection import VarianceThreshold
from pathlib import Path
import sys
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))
from src.config import (
    CLEANED_HEART_DATA, 
    CLUSTERED_DATA, 
    CLUSTER_MODEL, 
    SCALER, 
    FIGURES_DIR,
    MODELS_DIR,
    RANDOM_STATE,
    N_CLUSTERS
)

# ============================================================================
# STEP 1: FEATURE SELECTION FOR CLUSTERING
# ============================================================================

def select_clustering_features(heart):
    """
    Choosing features that define health behavior/status profiles.
    """
    
    # OPTION 1: Behavioral/Lifestyle Clusters 
    behavioral_features = [
        # Demographics
        'AgeCategory',  # 0-12 range (well distributed)
        # Note: Sex already encoded as binary during cleaning
        # Lifestyle (highest impact modifiable factors)
        'BMI',                    # Continuous, good spread
        'SleepHours',             # Peaked at 7-8, there's quite the outliers but good variation
        'PhysicalActivities',     # Binary: 78% yes, 22% no
        'SmokerStatus',           # 0-3: 60% never, 28% former, 12% current
        'AlcoholDrinkers',        # Binary: 55% yes, 45% no
        
        # Health engagement (shows healthcare access patterns)
        'LastCheckupTime',        # 0-3: 81% within year
        'FluVaxLast12',           # Binary: 53% yes
        
        # Self-reported health status
        'GeneralHealth',          # 1-5: good spread, peaked at 3-4
        'PhysicalHealthDays',     # 0-30: 62% report 0 days (right-skewed)
        'MentalHealthDays'        # 0-30: 61% report 0 days (right-skewed)
    ]
    
    # OPTION 2: We can add outcomes for disease-pattern clusters
    # We want clusters defined BY disease co-occurrence
    outcome_features = [
        'HadDiabetes',            # 14% prevalence (use binary 0 or 3)
        'HadHeartAttack',         # 5.5% prevalence
        'HadDepressiveDisorder',  # 21% prevalence
        'HadArthritis'            # 35% prevalence (age-related)
    ]

    behavioral_features.extend(outcome_features)
    
    print(f"Selected {len(behavioral_features)} features for clustering")
    return behavioral_features


# ============================================================================
# STEP 2: PREPARE DATA FOR CLUSTERING
# ============================================================================

def prepare_clustering_data(heart, features, use_pca=True, pca_components=None, 
                           handle_skew=True, use_robust_scaling=False):
    """
    Handle missing values, encode categorical variables, and apply transformations.
    
    Parameters:
    - use_pca: Apply PCA for dimensionality reduction
    - pca_components: Number of PCA components (None = auto-select to explain 95% variance)
    - handle_skew: Apply power transform to skewed features
    - use_robust_scaling: Use RobustScaler instead of StandardScaler (better for outliers)
    """
    
    df = heart[features].copy()
    
    # One-hot encode 'Sex' if needed
    if 'Sex' in df.columns:
        df = pd.get_dummies(df, columns=['Sex'], drop_first=True)
    
    # Fill missing values
    df = df.fillna(df.median())
    
    # Handle skewed features (PhysicalHealthDays, MentalHealthDays)
    skewed_features = ['PhysicalHealthDays', 'MentalHealthDays']
    if handle_skew:
        for feat in skewed_features:
            if feat in df.columns:
                # Apply Yeo-Johnson transformation (handles zeros and negatives)
                power_transformer = PowerTransformer(method='yeo-johnson', standardize=False)
                df[feat] = power_transformer.fit_transform(df[[feat]]).flatten()
    
    # Remove low-variance features (likely not useful for clustering)
    variance_selector = VarianceThreshold(threshold=0.01)
    df_array = variance_selector.fit_transform(df)
    selected_features = [df.columns[i] for i in variance_selector.get_support(indices=True)]
    
    # Scale features
    if use_robust_scaling:
        scaler = RobustScaler()  # Better for outliers
    else:
        scaler = StandardScaler()
    
    df_scaled = scaler.fit_transform(df_array)
    
    # Apply PCA if requested
    pca = None
    if use_pca:
        if pca_components is None:
            # Auto-select components to explain 95% variance
            pca_temp = PCA()
            pca_temp.fit(df_scaled)
            cumsum_variance = np.cumsum(pca_temp.explained_variance_ratio_)
            pca_components = np.argmax(cumsum_variance >= 0.95) + 1
            pca_components = min(pca_components, df_scaled.shape[1] - 1)  # Don't exceed n_features-1
        
        pca = PCA(n_components=pca_components, random_state=RANDOM_STATE)
        df_scaled = pca.fit_transform(df_scaled)
        print(f"  Applied PCA: {df_scaled.shape[1]} components explain "
              f"{pca.explained_variance_ratio_.sum():.1%} of variance")
    
    return df_scaled, selected_features, scaler, pca


# ============================================================================
# STEP 3: FIND OPTIMAL NUMBER OF CLUSTERS
# ============================================================================

def find_optimal_clusters(data, k_range=range(3, 8), try_multiple_algorithms=True):
    """
    Use multiple metrics and algorithms to find best clustering approach.
    
    Returns:
    - optimal_k: Best number of clusters
    - best_method: Best algorithm ('KMeans', 'Agglomerative')
    - best_score: Best silhouette score achieved
    """
    
    results = []
    
    # Test KMeans
    print("\nTesting KMeans:")
    kmeans_scores = []
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=20, max_iter=500)
        labels = kmeans.fit_predict(data)
        sil_score = silhouette_score(data, labels)
        ch_score = calinski_harabasz_score(data, labels)
        db_score = davies_bouldin_score(data, labels)
        kmeans_scores.append(sil_score)
        results.append({
            'method': 'KMeans',
            'k': k,
            'silhouette': sil_score,
            'calinski_harabasz': ch_score,
            'davies_bouldin': db_score
        })
        print(f"  K={k}: Silhouette={sil_score:.4f}, CH={ch_score:.1f}, DB={db_score:.4f}")
    
    # Test Agglomerative Clustering
    # Note: Agglomerative Clustering has O(n²) memory complexity
    # Skip for large datasets (>50k samples) to avoid memory issues
    MAX_SAMPLES_FOR_AGGLOMERATIVE = 50000
    if try_multiple_algorithms and len(data) <= MAX_SAMPLES_FOR_AGGLOMERATIVE:
        print("\nTesting Agglomerative Clustering:")
        agglo_scores = []
        for k in k_range:
            agglo = AgglomerativeClustering(n_clusters=k)
            labels = agglo.fit_predict(data)
            sil_score = silhouette_score(data, labels)
            ch_score = calinski_harabasz_score(data, labels)
            db_score = davies_bouldin_score(data, labels)
            agglo_scores.append(sil_score)
            results.append({
                'method': 'Agglomerative',
                'k': k,
                'silhouette': sil_score,
                'calinski_harabasz': ch_score,
                'davies_bouldin': db_score
            })
            print(f"  K={k}: Silhouette={sil_score:.4f}, CH={ch_score:.1f}, DB={db_score:.4f}")
    elif try_multiple_algorithms and len(data) > MAX_SAMPLES_FOR_AGGLOMERATIVE:
        print(f"\n⚠ Skipping Agglomerative Clustering: dataset too large ({len(data):,} samples)")
        print(f"  Agglomerative Clustering requires O(n²) memory and would need ~{len(data)**2 * 8 / 1e9:.1f} GB")
        print(f"  Using KMeans only (more memory-efficient for large datasets)")
    
    # Find best result
    results_df = pd.DataFrame(results)
    best_idx = results_df['silhouette'].idxmax()
    best_result = results_df.loc[best_idx]
    
    optimal_k = int(best_result['k'])
    best_method = best_result['method']
    best_score = best_result['silhouette']
    
    print(f"\n{'='*60}")
    print(f"✓ BEST RESULT: {best_method} with K={optimal_k}")
    print(f"  Silhouette Score: {best_score:.4f}")
    print(f"  Calinski-Harabasz: {best_result['calinski_harabasz']:.1f}")
    print(f"  Davies-Bouldin: {best_result['davies_bouldin']:.4f}")
    print(f"{'='*60}")
    
    # Plot comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Silhouette scores
    ax1 = axes[0]
    for method in results_df['method'].unique():
        method_data = results_df[results_df['method'] == method]
        ax1.plot(method_data['k'], method_data['silhouette'], 
                marker='o', linewidth=2, label=method)
    ax1.set_xlabel('Number of Clusters (K)')
    ax1.set_ylabel('Silhouette Score')
    ax1.set_title('Silhouette Score by Method')
    ax1.legend()
    ax1.grid(alpha=0.3)
    ax1.axhline(y=best_score, color='r', linestyle='--', alpha=0.5, label=f'Best: {best_score:.4f}')
    
    # All metrics comparison
    ax2 = axes[1]
    kmeans_data = results_df[results_df['method'] == 'KMeans']
    ax2_twin = ax2.twinx()
    
    ax2.plot(kmeans_data['k'], kmeans_data['silhouette'], 'o-', label='Silhouette', color='blue')
    ax2_twin.plot(kmeans_data['k'], kmeans_data['calinski_harabasz'], 's-', 
                  label='Calinski-Harabasz', color='green')
    ax2.plot(kmeans_data['k'], kmeans_data['davies_bouldin'], '^-', 
             label='Davies-Bouldin (lower is better)', color='red')
    
    ax2.set_xlabel('Number of Clusters (K)')
    ax2.set_ylabel('Silhouette / Davies-Bouldin', color='black')
    ax2_twin.set_ylabel('Calinski-Harabasz', color='green')
    ax2.set_title('KMeans: All Metrics')
    ax2.legend(loc='upper left')
    ax2_twin.legend(loc='upper right')
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'optimal_k.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return optimal_k, best_method, best_score


# ============================================================================
# STEP 4: TRAIN FINAL CLUSTERING MODEL
# ============================================================================

def train_clustering(data, k=5, method='KMeans'):
    """
    Train clustering model with specified algorithm.
    
    Parameters:
    - data: Preprocessed data
    - k: Number of clusters
    - method: 'KMeans' or 'Agglomerative'
    """
    
    if method == 'KMeans':
        model = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=20, max_iter=500)
        labels = model.fit_predict(data)
    elif method == 'Agglomerative':
        model = AgglomerativeClustering(n_clusters=k)
        labels = model.fit_predict(data)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Calculate final metrics
    sil_score = silhouette_score(data, labels)
    ch_score = calinski_harabasz_score(data, labels)
    db_score = davies_bouldin_score(data, labels)
    
    print(f"\nFinal Model: {method} with K={k}")
    print(f"  Silhouette Score: {sil_score:.4f}")
    print(f"  Calinski-Harabasz: {ch_score:.1f}")
    print(f"  Davies-Bouldin: {db_score:.4f}")
    
    print(f"\nCluster sizes:")
    unique, counts = np.unique(labels, return_counts=True)
    for cluster, count in zip(unique, counts):
        print(f"  Cluster {cluster}: {count:,} ({count/len(labels)*100:.1f}%)")
    
    return model, labels


# ============================================================================
# STEP 5: PROFILE CLUSTERS (THE MONEY SHOT!)
# ============================================================================

def profile_clusters(heart, labels, features):
    """Generate human-readable cluster descriptions."""
    
    heart_with_clusters = heart.copy()
    heart_with_clusters['Cluster'] = labels
    
    profiles = []
    
    for cluster_id in sorted(heart_with_clusters['Cluster'].unique()):
        cluster_data = heart_with_clusters[heart_with_clusters['Cluster'] == cluster_id]
        
        profile = {
            'Cluster': cluster_id,
            'Size': len(cluster_data),
            'Avg_Age': cluster_data['AgeCategory'].mean(),
            'Avg_BMI': cluster_data['BMI'].mean(),
            'Avg_Sleep': cluster_data['SleepHours'].mean(),
            'Pct_PhysActive': cluster_data['PhysicalActivities'].mean() * 100,
            'Pct_Smokers': (cluster_data['SmokerStatus'] >= 2).mean() * 100,
            'Pct_Diabetes': cluster_data['HadDiabetes'].mean() * 100,
            'Pct_HeartAttack': cluster_data['HadHeartAttack'].mean() * 100,
            'Pct_Depression': cluster_data['HadDepressiveDisorder'].mean() * 100,
            'Avg_PhysHealthDays': cluster_data['PhysicalHealthDays'].mean(),
            'Avg_MentHealthDays': cluster_data['MentalHealthDays'].mean()
        }
        profiles.append(profile)
    
    profile_df = pd.DataFrame(profiles)
    print("\n" + "="*80)
    print("CLUSTER PROFILES")
    print("="*80)
    print(profile_df.to_string(index=False))
    
    return profile_df, heart_with_clusters


# ============================================================================
# STEP 6: NAME THE CLUSTERS (MAKE IT MEMORABLE!)
# ============================================================================

def assign_cluster_names(profile_df):
    """
    Manually assign catchy names based on profiles.
    Adjust these after you see the data!
    """
    
    # Example naming logic (customize after seeing your results)
    names = {}
    
    for idx, row in profile_df.iterrows():
        cluster_id = int(row['Cluster'])
        
        # Decision tree for naming
        if row['Avg_Age'] < 4 and row['Pct_PhysActive'] > 70:
            names[cluster_id] = "💪 Healthy Young Actives"
        elif row['Pct_Smokers'] > 30:
            names[cluster_id] = "🚬 Struggling Smokers"
        elif row['Avg_Age'] > 8 and row['Pct_Diabetes'] > 20:
            names[cluster_id] = "🧓 Aging with Challenges"
        elif row['Pct_PhysActive'] < 50 and row['Avg_BMI'] > 28:
            names[cluster_id] = "🛋️ Sedentary High-Risk"
        else:
            names[cluster_id] = f"📊 Cluster {cluster_id}"
    
    profile_df['Name'] = profile_df['Cluster'].map(names)
    return profile_df, names


# ============================================================================
# STEP 7: VISUALIZATION - RADAR CHARTS
# ============================================================================

def create_radar_chart(profile_df, cluster_names):
    """Create radar chart for each cluster - PRESENTATION GOLD!"""
    
    # Select key features for radar
    features_for_radar = [
        'Avg_BMI', 'Avg_Sleep', 'Pct_PhysActive', 
        'Pct_Diabetes', 'Pct_Depression', 'Avg_PhysHealthDays'
    ]
    
    # Normalize to 0-1 scale for radar
    radar_data = profile_df[features_for_radar].copy()
    for col in radar_data.columns:
        min_val = radar_data[col].min()
        max_val = radar_data[col].max()
        radar_data[col] = (radar_data[col] - min_val) / (max_val - min_val)
    
    # Create subplot for each cluster
    n_clusters = len(profile_df)
    fig, axes = plt.subplots(1, n_clusters, figsize=(5*n_clusters, 5), 
                             subplot_kw=dict(projection='polar'))
    
    if n_clusters == 1:
        axes = [axes]
    
    categories = ['BMI', 'Sleep', 'Active%', 'Diabetes%', 'Depress%', 'PhysHealth']
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]
    
    for idx, (ax, (_, row)) in enumerate(zip(axes, profile_df.iterrows())):
        values = radar_data.iloc[idx].tolist()
        values += values[:1]
        
        ax.plot(angles, values, 'o-', linewidth=2, label=row['Name'])
        ax.fill(angles, values, alpha=0.25)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_ylim(0, 1)
        ax.set_title(f"{row['Name']}\n({row['Size']:,} people)", 
                     fontsize=10, pad=20)
        ax.grid(True)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'cluster_radar_charts.png', dpi=300, bbox_inches='tight')
    plt.show()


# ============================================================================
# STEP 8: GEOGRAPHIC DISTRIBUTION
# ============================================================================

def map_clusters_by_state(heart_with_clusters):
    """Show which states have which cluster concentrations."""
    
    state_cluster_dist = pd.crosstab(
        heart_with_clusters['State'], 
        heart_with_clusters['Cluster'],
        normalize='index'
    ) * 100
    
    # Find dominant cluster per state
    dominant_cluster = state_cluster_dist.idxmax(axis=1)
    
    print("\nTop 10 states by cluster dominance:")
    print(dominant_cluster.head(10))
    
    # Heatmap
    plt.figure(figsize=(10, 12))
    sns.heatmap(state_cluster_dist.iloc[:20], annot=True, fmt='.1f', 
                cmap='YlOrRd', cbar_kws={'label': '% of state population'})
    plt.title('Cluster Distribution by State (Top 20)')
    plt.xlabel('Cluster ID')
    plt.ylabel('State')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'cluster_by_state_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return state_cluster_dist, dominant_cluster


# ============================================================================
# MAIN EXECUTION PIPELINE
# ============================================================================

def run_clustering_pipeline(heart, use_pca=True, use_robust_scaling=False, 
                           handle_skew=True, try_multiple_algorithms=True):
    """
    Run complete clustering analysis with improved preprocessing.
    
    Parameters:
    - use_pca: Apply PCA for dimensionality reduction (recommended for better scores)
    - use_robust_scaling: Use RobustScaler instead of StandardScaler (better for outliers)
    - handle_skew: Apply power transform to skewed features
    - try_multiple_algorithms: Test both KMeans and Agglomerative clustering
    """
    
    print("="*80)
    print("CLUSTERING PIPELINE - Discovering Health Personas")
    print("="*80)
    
    # Step 1: Select features
    features = select_clustering_features(heart)
    print(f"\n✓ Selected {len(features)} features for clustering")
    
    # Step 2: Prepare data with improved preprocessing
    print("\n" + "-"*80)
    print("Preparing data (with PCA and transformations)...")
    print("-"*80)
    data_scaled, feature_names, scaler, pca = prepare_clustering_data(
        heart, features, 
        use_pca=use_pca,
        handle_skew=handle_skew,
        use_robust_scaling=use_robust_scaling
    )
    print(f"✓ Prepared {data_scaled.shape[0]:,} samples, {data_scaled.shape[1]} dimensions")
    
    # Step 3: Find optimal K and method
    print("\n" + "-"*80)
    print("Finding optimal number of clusters and algorithm...")
    print("-"*80)
    optimal_k, best_method, best_score = find_optimal_clusters(
        data_scaled, 
        try_multiple_algorithms=try_multiple_algorithms
    )
    
    # Step 4: Train final model
    print("\n" + "-"*80)
    print(f"Training final model: {best_method} with K={optimal_k}...")
    print("-"*80)
    model, labels = train_clustering(data_scaled, k=optimal_k, method=best_method)
    
    # Step 5: Profile clusters
    print("\n" + "-"*80)
    print("Profiling clusters...")
    print("-"*80)
    profile_df, heart_with_clusters = profile_clusters(heart, labels, features)
    
    # Step 6: Name clusters
    profile_df, cluster_names = assign_cluster_names(profile_df)
    print("\n✓ Cluster names assigned:")
    for cid, name in cluster_names.items():
        print(f"  {name}")
    
    # Step 7: Create visualizations
    print("\n" + "-"*80)
    print("Creating visualizations...")
    print("-"*80)
    create_radar_chart(profile_df, cluster_names)
    
    # Step 8: Geographic analysis
    state_dist, dominant = map_clusters_by_state(heart_with_clusters)
    
    print("\n" + "="*80)
    print("✓ CLUSTERING COMPLETE!")
    print("="*80)
    print(f"Final Silhouette Score: {best_score:.4f}")
    print(f"Outputs saved:")
    print(f"  - {FIGURES_DIR / 'optimal_k.png'}")
    print(f"  - {FIGURES_DIR / 'cluster_radar_charts.png'}")
    print(f"  - {FIGURES_DIR / 'cluster_by_state_heatmap.png'}")
    
    # Save cluster assignments
    heart_with_clusters.to_csv(CLUSTERED_DATA, index=False)
    print(f"  - {CLUSTERED_DATA}")
    
    return heart_with_clusters, profile_df, cluster_names, model, scaler, pca


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    # Load your cleaned data
    print("="*80)
    print("CLUSTERING PIPELINE")
    print("="*80)
    print(f"\n📂 Loading cleaned data from {CLEANED_HEART_DATA}...")
    heart = pd.read_csv(CLEANED_HEART_DATA)
    print(f"✓ Loaded {len(heart):,} samples")
    
    # Run pipeline with improved settings
    # use_pca=True: Apply PCA for better clustering (recommended)
    # use_robust_scaling=False: Use StandardScaler (can try True if many outliers)
    # handle_skew=True: Transform skewed features (recommended)
    # try_multiple_algorithms=True: Test both KMeans and Agglomerative
    heart_clustered, profiles, names, model, scaler, pca = run_clustering_pipeline(
        heart,
        use_pca=True,
        use_robust_scaling=False,
        handle_skew=True,
        try_multiple_algorithms=True
    )
    
    # Save model for later use in calculator
    import joblib
    joblib.dump(model, CLUSTER_MODEL)
    joblib.dump(scaler, SCALER)
    if pca is not None:
        joblib.dump(pca, MODELS_DIR / 'pca_model.pkl')
    print(f"\n💾 Models saved:")
    print(f"  - {CLUSTER_MODEL}")
    print(f"  - {SCALER}")
    if pca is not None:
        print(f"  - {MODELS_DIR / 'pca_model.pkl'}")
    
    print("\n✅ All done!")