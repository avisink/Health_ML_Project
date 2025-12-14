"""
SIMPLIFIED CLUSTERING PIPELINE
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from pathlib import Path
import sys
import joblib

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))
from src.config import (
    CLEANED_HEART_DATA, 
    CLUSTERED_DATA, 
    CLUSTER_MODEL, 
    SCALER, 
    FIGURES_DIR,
    MODELS_DIR,
    RANDOM_STATE
)

# ============================================================================
# STEP 1: SELECT FEATURES
# ============================================================================

def select_features():
    """Pick features that make sense for health profiles."""
    return [
        'AgeCategory', 'BMI', 'SleepHours', 
        'PhysicalActivities', 'SmokerStatus', 'AlcoholDrinkers',
        'GeneralHealth', 'PhysicalHealthDays', 'MentalHealthDays',
        'LastCheckupTime', 'FluVaxLast12',
        # Add disease outcomes for richer profiles
        'HadDiabetes', 'HadHeartAttack', 'HadDepressiveDisorder'
    ]


# ============================================================================
# STEP 2: PREPARE DATA
# ============================================================================

def prepare_data(heart, features):
    """Clean and scale data. That's it."""
    
    df = heart[features].copy()
    
    # Fill missing (shouldn't be many after cleaning)
    df = df.fillna(df.median())
    
    # Scale everything
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)
    
    print(f"✓ Prepared {df.shape[0]:,} samples × {df.shape[1]} features")
    return df_scaled, df.columns.tolist(), scaler


# ============================================================================
# STEP 3: FIND BEST K (Quick version)
# ============================================================================

def find_best_k(data, k_range=range(3, 9)):
    """Test K=3 to 8, pick best silhouette score."""
    
    print("\nTesting different K values...")
    scores = []
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels = kmeans.fit_predict(data)
        score = silhouette_score(data, labels)
        scores.append(score)
        print(f"  K={k}: Silhouette = {score:.3f}")
    
    # Plot
    plt.figure(figsize=(8, 5))
    plt.plot(k_range, scores, 'o-', linewidth=2, markersize=8)
    plt.xlabel('Number of Clusters (K)', fontsize=12)
    plt.ylabel('Silhouette Score', fontsize=12)
    plt.title('Finding Optimal K', fontsize=14, fontweight='bold')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'optimal_k.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    best_k = list(k_range)[np.argmax(scores)]
    best_score = max(scores)
    print(f"\n✓ Best K = {best_k} (Silhouette = {best_score:.3f})")
    
    return best_k, best_score


# ============================================================================
# STEP 4: TRAIN FINAL MODEL
# ============================================================================

def train_model(data, k):
    """Train KMeans with best K."""
    
    print(f"\nTraining final model with K={k}...")
    kmeans = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=20)
    labels = kmeans.fit_predict(data)
    
    # Show cluster sizes
    print("\nCluster sizes:")
    for i in range(k):
        count = (labels == i).sum()
        pct = count / len(labels) * 100
        print(f"  Cluster {i}: {count:,} ({pct:.1f}%)")
    
    return kmeans, labels


# ============================================================================
# STEP 5: PROFILE CLUSTERS
# ============================================================================

def profile_clusters(heart, labels):
    """Calculate average stats for each cluster."""
    
    heart_clustered = heart.copy()
    heart_clustered['Cluster'] = labels
    
    profiles = []
    for i in sorted(np.unique(labels)):
        cluster_data = heart_clustered[heart_clustered['Cluster'] == i]
        
        # Convert diabetes to binary for cleaner stats
        diabetes_pct = (cluster_data['HadDiabetes'] == 3).mean() * 100
        
        profile = {
            'Cluster': i,
            'Size': len(cluster_data),
            'Pct': len(cluster_data) / len(heart_clustered) * 100,
            'Avg_Age': cluster_data['AgeCategory'].mean(),
            'Avg_BMI': cluster_data['BMI'].mean(),
            'Avg_Sleep': cluster_data['SleepHours'].mean(),
            'Pct_Active': cluster_data['PhysicalActivities'].mean() * 100,
            'Pct_Smokers': (cluster_data['SmokerStatus'] >= 2).mean() * 100,
            'Pct_Diabetes': diabetes_pct,
            'Pct_HeartAttack': cluster_data['HadHeartAttack'].mean() * 100,
            'Pct_Depression': cluster_data['HadDepressiveDisorder'].mean() * 100,
            'Avg_PhysHealthDays': cluster_data['PhysicalHealthDays'].mean(),
            'Avg_MentHealthDays': cluster_data['MentalHealthDays'].mean(),
        }
        profiles.append(profile)
    
    profile_df = pd.DataFrame(profiles)
    
    print("\n" + "="*100)
    print("CLUSTER PROFILES")
    print("="*100)
    print(profile_df.round(1).to_string(index=False))
    
    return profile_df, heart_clustered


# ============================================================================
# STEP 6: NAME CLUSTERS (Manual - customize after seeing results!)
# ============================================================================

def name_clusters(profile_df):
    """
    Give clusters catchy names based on their characteristics.
    CUSTOMIZE THIS after you see your actual profiles!
    """
    
    names = {}
    
    for _, row in profile_df.iterrows():
        cid = int(row['Cluster'])
        
        # Simple naming logic - ADJUST BASED ON YOUR RESULTS
        if row['Avg_Age'] < 4 and row['Pct_Active'] > 75:
            names[cid] = "💪 Healthy Young Actives"
        
        elif row['Pct_Smokers'] > 25 and row['Avg_MentHealthDays'] > 5:
            names[cid] = "🚬 Struggling Smokers"
        
        elif row['Avg_Age'] > 8 and row['Pct_Diabetes'] > 20:
            names[cid] = "🧓 Aging with Challenges"
        
        elif row['Pct_Active'] < 60 and row['Avg_BMI'] > 28:
            names[cid] = "🛋️ Sedentary High-Risk"
        
        elif row['Pct_Active'] > 75 and row['Avg_Age'] > 4:
            names[cid] = "🏃 Active Middle-Agers"
        
        else:
            # Fallback - describe key traits
            age_label = "Young" if row['Avg_Age'] < 5 else "Middle" if row['Avg_Age'] < 9 else "Senior"
            health_label = "Healthy" if row['Pct_Diabetes'] < 10 else "At-Risk"
            names[cid] = f"📊 {age_label} {health_label}"
    
    profile_df['Name'] = profile_df['Cluster'].map(names)
    
    print("\n✓ Cluster names:")
    for cid, name in names.items():
        pct = profile_df[profile_df['Cluster'] == cid]['Pct'].values[0]
        size = profile_df[profile_df['Cluster'] == cid]['Size'].values[0]
        print(f"  {name} - {size:,} people ({pct:.1f}%)")
    
    return profile_df, names


# ============================================================================
# STEP 7: CREATE VISUALIZATIONS
# ============================================================================

def create_visualizations(profile_df):
    """Make radar charts and other visuals."""
    
    # 1. RADAR CHART (Presentation Gold!)
    features_radar = ['Avg_BMI', 'Avg_Sleep', 'Pct_Active', 
                      'Pct_Diabetes', 'Pct_Depression']
    
    # Normalize to 0-1
    radar_data = profile_df[features_radar].copy()
    for col in radar_data.columns:
        min_val, max_val = radar_data[col].min(), radar_data[col].max()
        if max_val > min_val:  # Avoid division by zero
            radar_data[col] = (radar_data[col] - min_val) / (max_val - min_val)
        else:
            radar_data[col] = 0
    
    # Plot
    n_clusters = len(profile_df)
    fig, axes = plt.subplots(1, n_clusters, figsize=(5*n_clusters, 5),
                             subplot_kw=dict(projection='polar'))
    
    if n_clusters == 1:
        axes = [axes]
    
    categories = ['BMI', 'Sleep', 'Active%', 'Diabetes%', 'Depression%']
    angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]
    
    for idx, (ax, (_, row)) in enumerate(zip(axes, profile_df.iterrows())):
        values = radar_data.iloc[idx].tolist()
        values += values[:1]
        
        ax.plot(angles, values, 'o-', linewidth=2)
        ax.fill(angles, values, alpha=0.25)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, size=10)
        ax.set_ylim(0, 1)
        ax.set_title(f"{row['Name']}\n({row['Size']:,} people)", 
                     size=11, pad=20, fontweight='bold')
        ax.grid(True)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'cluster_radar_charts.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: cluster_radar_charts.png")
    
    
    # 2. SIMPLE BAR CHART (Disease prevalence comparison)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    conditions = [
        ('Pct_Diabetes', 'Diabetes', axes[0]),
        ('Pct_HeartAttack', 'Heart Attack', axes[1]),
        ('Pct_Depression', 'Depression', axes[2])
    ]
    
    for col, title, ax in conditions:
        profile_df.plot(x='Name', y=col, kind='bar', ax=ax, 
                       color='steelblue', legend=False)
        ax.set_title(f'{title} Prevalence by Cluster', fontweight='bold')
        ax.set_ylabel('Prevalence (%)')
        ax.set_xlabel('')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'cluster_disease_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: cluster_disease_comparison.png")


# ============================================================================
# STEP 8: GEOGRAPHIC DISTRIBUTION
# ============================================================================

def map_by_state(heart_clustered, profile_df):
    """Show cluster distribution by state."""
    
    # Get cluster names
    cluster_names = dict(zip(profile_df['Cluster'], profile_df['Name']))
    
    # Calculate distribution
    state_dist = pd.crosstab(
        heart_clustered['State'],
        heart_clustered['Cluster'],
        normalize='index'
    ) * 100
    
    # Show top 20 states
    top_states = state_dist.iloc[:20]
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(top_states, annot=True, fmt='.1f', cmap='YlOrRd',
                cbar_kws={'label': '% of State Population'})
    plt.title('Cluster Distribution by State (Top 20)', fontsize=14, fontweight='bold')
    plt.xlabel('Cluster ID')
    plt.ylabel('State')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'cluster_by_state.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: cluster_by_state.png")
    
    return state_dist


# ============================================================================
# MAIN PIPELINE (Simple!)
# ============================================================================

def run_clustering_pipeline(heart):
    """
    Complete clustering in one function.
    No fancy options - just works!
    """
    
    print("="*80)
    print("CLUSTERING PIPELINE - Finding Health Personas")
    print("="*80)
    
    # 1. Select features
    features = select_features()
    print(f"\n✓ Using {len(features)} features")
    
    # 2. Prepare data
    print("\nPreparing data...")
    data_scaled, feature_names, scaler = prepare_data(heart, features)
    
    # 3. Find best K
    best_k, best_score = find_best_k(data_scaled)
    
    # 4. Train model
    model, labels = train_model(data_scaled, k=best_k)
    
    # 5. Profile clusters
    print("\nAnalyzing clusters...")
    profile_df, heart_clustered = profile_clusters(heart, labels)
    
    # 6. Name clusters (CUSTOMIZE THIS!)
    profile_df, names = name_clusters(profile_df)
    
    # 7. Create visuals
    print("\nCreating visualizations...")
    create_visualizations(profile_df)
    
    # 8. Map by state
    state_dist = map_by_state(heart_clustered, profile_df)
    
    # Save everything
    print("\nSaving outputs...")
    heart_clustered.to_csv(CLUSTERED_DATA, index=False)
    profile_df.to_csv(FIGURES_DIR.parent / 'reports' / 'cluster_profiles.csv', index=False)
    joblib.dump(model, CLUSTER_MODEL)
    joblib.dump(scaler, SCALER)
    
    print("\n" + "="*80)
    print("✅ CLUSTERING COMPLETE!")
    print("="*80)
    print(f"Silhouette Score: {best_score:.3f}")
    print(f"Number of Clusters: {best_k}")
    print(f"\nOutputs:")
    print(f"  📊 Figures: {FIGURES_DIR}")
    print(f"  💾 Data: {CLUSTERED_DATA}")
    print(f"  🤖 Model: {CLUSTER_MODEL}")
    
    print("\n🧪 Quick sanity checks:")
    print(f"✓ All clusters have >1000 people: {(profiles['Size'] > 1000).all()}")
    print(f"✓ Silhouette score >0.2: {best_score > 0.2}")
    print(f"✓ Avg BMI reasonable (20-35): {profiles['Avg_BMI'].between(20, 35).all()}")
    print(f"✓ Files saved: {CLUSTER_MODEL.exists()}")
    return heart_clustered, profile_df, names, model, scaler


# ============================================================================
# RUN IT!
# ============================================================================

if __name__ == "__main__":
    print("\n📂 Loading data...")
    heart = pd.read_csv(CLEANED_HEART_DATA)
    print(f"✓ Loaded {len(heart):,} samples\n")
    
    # Run the pipeline
    heart_clustered, profiles, names, model, scaler = run_clustering_pipeline(heart)
    
    print("\n🎉 Done! Check outputs/ folder for visuals.")
    print("\nNext steps:")
    print("  1. Review cluster_profiles.csv")
    print("  2. Customize cluster names in name_clusters() function")
    print("  3. Run modeling pipeline: python src/modeling.py")