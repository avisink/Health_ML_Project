"""
HEALTH PERSONAS TAB
Lets a user enter their profile, see which KMeans cluster (health persona) they
are closest to, and gives data-science readers a look under the hood.

The clusters in data/processed/heart_with_clusters.csv come from
src/clustering.py: StandardScaler on the 14 features below, then KMeans.
KMeans assigns a point to its nearest centroid, so refitting the scaler and
taking per-cluster means in scaled space reproduces the model's predictions
without needing the pickled model on Streamlit Cloud.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.decomposition import PCA
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_samples,
)
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
from src.config import CLUSTERED_DATA, RANDOM_STATE

# Same features (and order) as src/clustering.py select_features()
CLUSTER_FEATURES = [
    'AgeCategory', 'BMI', 'SleepHours',
    'PhysicalActivities', 'SmokerStatus', 'AlcoholDrinkers',
    'GeneralHealth', 'PhysicalHealthDays', 'MentalHealthDays',
    'LastCheckupTime', 'FluVaxLast12',
    'HadDiabetes', 'HadHeartAttack', 'HadDepressiveDisorder'
]

FEATURE_LABELS = {
    'AgeCategory': 'Age group',
    'BMI': 'BMI',
    'SleepHours': 'Sleep (hrs)',
    'PhysicalActivities': 'Physically active',
    'SmokerStatus': 'Smoking',
    'AlcoholDrinkers': 'Drinks alcohol',
    'GeneralHealth': 'Self-rated health',
    'PhysicalHealthDays': 'Physically unwell days',
    'MentalHealthDays': 'Mentally unwell days',
    'LastCheckupTime': 'Time since checkup',
    'FluVaxLast12': 'Flu shot (12 mo)',
    'HadDiabetes': 'Diabetes',
    'HadHeartAttack': 'Heart attack',
    'HadDepressiveDisorder': 'Depression',
}

AGE_LABELS = ["18-24", "25-29", "30-34", "35-39", "40-44", "45-49",
              "50-54", "55-59", "60-64", "65-69", "70-74", "75-79", "80+"]

# Categorical palette (dark-surface steps), assigned to cluster ids in fixed order
PERSONA_COLORS = ['#3987e5', '#d95926', '#199e70', '#c98500',
                  '#d55181', '#008300', '#9085e9', '#e66767']
# Secondary encoding so identity never relies on color alone
PERSONA_SYMBOLS = ['circle', 'square', 'diamond', 'triangle-up',
                   'pentagon', 'hexagon', 'triangle-down', 'cross']

# Names written from outputs/reports/cluster_profiles.csv (re-check if clustering is re-run)
PERSONAS = {
    0: ("🩹", "High Symptom Burden",
        "About 26 physically unwell days a month, the lowest self-rated health, and the "
        "least active group (42%). Diabetes and depression are both above average."),
    1: ("🧠", "Mental Health Focus",
        "Nearly everyone here has a depression diagnosis, with about 10 mentally unwell "
        "days a month, yet most stay physically active."),
    2: ("🩺", "Overdue for a Checkup",
        "Younger and generally healthy, but most haven't seen a doctor in 2+ years "
        "and few got a flu shot."),
    3: ("🌱", "Healthy, Skipping the Flu Shot",
        "Active, good self-rated health, very few chronic conditions, and no flu "
        "vaccine in the past year."),
    4: ("🩸", "Managing Diabetes",
        "Older with the highest average BMI. Almost all have diabetes, and they "
        "keep up with regular checkups."),
    5: ("⭐", "Thriving & Engaged",
        "The largest group. Active, healthy, the fewest smokers, and everyone "
        "got a flu shot this year."),
    6: ("❤️", "Heart Attack Survivors",
        "The oldest group, and every member has had a heart attack. More smokers, "
        "less activity, and about 9 physically unwell days a month."),
}

PLOTLY_CONFIG = {
    "displayModeBar": True,
    "modeBarButtonsToAdd": ["resetScale2d"],
    "displaylogo": False,
    "responsive": True
}

CARD_STYLE = ("background-color: rgba(38, 39, 48, 0.6); padding: 20px; border-radius: 12px; "
              "border: 1px solid rgba(255, 255, 255, 0.1); box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);")


def persona_name(cid):
    emoji, name, _ = PERSONAS.get(cid, ("📊", f"Persona {cid}", ""))
    return f"{emoji} {name}"


def persona_color(cid):
    return PERSONA_COLORS[cid % len(PERSONA_COLORS)]


# ============================================================================
# DATA + MODEL
# ============================================================================

@st.cache_data
def load_clustered_data():
    return pd.read_csv(str(CLUSTERED_DATA), usecols=CLUSTER_FEATURES + ['Cluster'])


@st.cache_resource
def build_model():
    """Refit scaler, centroids, PCA, and evaluation metrics once per server."""
    df = load_clustered_data()
    X = df[CLUSTER_FEATURES].fillna(df[CLUSTER_FEATURES].median())
    labels = df['Cluster'].to_numpy()

    scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X)

    cluster_ids = np.sort(np.unique(labels))
    centroids = np.vstack([X_scaled[labels == c].mean(axis=0) for c in cluster_ids])

    pca = PCA(n_components=3, random_state=RANDOM_STATE).fit(X_scaled)

    # Stratified-ish random sample for plots and O(n^2) metrics
    rng = np.random.default_rng(RANDOM_STATE)
    sample_idx = rng.choice(len(X_scaled), size=min(8000, len(X_scaled)), replace=False)
    X_sample = X_scaled[sample_idx]
    y_sample = labels[sample_idx]

    sil = silhouette_samples(X_sample, y_sample)
    metrics = {
        'silhouette': float(sil.mean()),
        'calinski_harabasz': float(calinski_harabasz_score(X_sample, y_sample)),
        'davies_bouldin': float(davies_bouldin_score(X_sample, y_sample)),
    }

    return {
        'scaler': scaler,
        'cluster_ids': cluster_ids,
        'centroids': centroids,
        'pca': pca,
        'sample_pcs': pca.transform(X_sample),
        'sample_labels': y_sample,
        'sample_silhouette': sil,
        'metrics': metrics,
        'sizes': pd.Series(labels).value_counts().sort_index(),
        'n': len(df),
    }


@st.cache_data
def persona_stats():
    """Human-readable stats per persona plus an 'Everyone' row."""
    df = load_clustered_data()

    def stats(g):
        return pd.Series({
            'Avg age': AGE_LABELS[int(round(g['AgeCategory'].mean()))],
            'Avg BMI': g['BMI'].mean(),
            'Avg sleep (hrs)': g['SleepHours'].mean(),
            '% active': g['PhysicalActivities'].mean() * 100,
            '% current smokers': (g['SmokerStatus'] >= 2).mean() * 100,
            '% drink alcohol': g['AlcoholDrinkers'].mean() * 100,
            'Self-rated health (1-5)': g['GeneralHealth'].mean(),
            'Physically unwell days': g['PhysicalHealthDays'].mean(),
            'Mentally unwell days': g['MentalHealthDays'].mean(),
            '% checkup in past year': (g['LastCheckupTime'] == 0).mean() * 100,
            '% flu shot': g['FluVaxLast12'].mean() * 100,
            '% diabetes': (g['HadDiabetes'] == 3).mean() * 100,
            '% heart attack': g['HadHeartAttack'].mean() * 100,
            '% depression': g['HadDepressiveDisorder'].mean() * 100,
        })

    table = df.groupby('Cluster').apply(stats)
    table.index = [persona_name(c) for c in table.index]
    table.loc['Everyone'] = stats(df)
    return table


def match_user(model, user_values):
    """Return assigned cluster, distances, and relative closeness for one user."""
    user_scaled = model['scaler'].transform(pd.DataFrame([user_values], columns=CLUSTER_FEATURES))[0]
    distances = np.linalg.norm(model['centroids'] - user_scaled, axis=1)
    inv = 1 / np.maximum(distances, 1e-6) ** 2
    closeness = inv / inv.sum() * 100
    best = int(model['cluster_ids'][np.argmin(distances)])
    return best, user_scaled, distances, closeness


# ============================================================================
# UI PIECES
# ============================================================================

def profile_form():
    """Collect the 14 clustering features. Returns a dict in CLUSTER_FEATURES order."""
    with st.form("persona_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**About you**")
            age = st.selectbox("Age group", AGE_LABELS, index=5, key="p_age")
            bmi = st.slider("BMI", 15.0, 50.0, 27.0, 0.5, key="p_bmi",
                            help="Not sure? The Risk Calculator tab can work it out from height and weight.")
            general = st.select_slider("How would you rate your health?",
                                       ["Poor", "Fair", "Good", "Very good", "Excellent"],
                                       value="Very good", key="p_general")
            phys_days = st.slider("Days physically unwell (last 30)", 0, 30, 0, key="p_phys")
            ment_days = st.slider("Days mentally unwell (last 30)", 0, 30, 0, key="p_ment")

        with col2:
            st.markdown("**Lifestyle**")
            sleep = st.slider("Sleep hours/night", 3.0, 12.0, 7.0, 0.5, key="p_sleep")
            active = st.radio("Physically active in the past month?", ["Yes", "No"],
                              horizontal=True, key="p_active")
            smoker = st.selectbox("Smoking", ["Never smoked", "Former smoker",
                                              "Smoke some days", "Smoke every day"], key="p_smoker")
            alcohol = st.radio("Had a drink in the past month?", ["Yes", "No"],
                               horizontal=True, key="p_alcohol")

        with col3:
            st.markdown("**Health & care**")
            checkup = st.selectbox("Last routine checkup", ["Within the past year", "1-2 years ago",
                                                            "2-5 years ago", "5+ years ago"], key="p_checkup")
            flu = st.radio("Flu shot in the past 12 months?", ["Yes", "No"],
                           horizontal=True, key="p_flu")
            diabetes = st.selectbox("Ever told you have diabetes?",
                                    ["No", "Pre-diabetes / borderline", "Only during pregnancy", "Yes"],
                                    key="p_diabetes")
            heart = st.radio("Ever had a heart attack?", ["No", "Yes"], horizontal=True, key="p_heart")
            depression = st.radio("Ever diagnosed with depression?", ["No", "Yes"],
                                  horizontal=True, key="p_depression")

        st.form_submit_button("🔎 Find my persona", type="primary", use_container_width=True)

    yes = {"Yes": 1, "No": 0}
    return {
        'AgeCategory': AGE_LABELS.index(age),
        'BMI': bmi,
        'SleepHours': sleep,
        'PhysicalActivities': yes[active],
        'SmokerStatus': ["Never smoked", "Former smoker", "Smoke some days", "Smoke every day"].index(smoker),
        'AlcoholDrinkers': yes[alcohol],
        'GeneralHealth': ["Poor", "Fair", "Good", "Very good", "Excellent"].index(general) + 1,
        'PhysicalHealthDays': phys_days,
        'MentalHealthDays': ment_days,
        'LastCheckupTime': ["Within the past year", "1-2 years ago", "2-5 years ago", "5+ years ago"].index(checkup),
        'FluVaxLast12': yes[flu],
        'HadDiabetes': ["No", "Pre-diabetes / borderline", "Only during pregnancy", "Yes"].index(diabetes),
        'HadHeartAttack': yes[heart],
        'HadDepressiveDisorder': yes[depression],
    }


def hero_card(cid, closeness, share):
    emoji, name, desc = PERSONAS.get(cid, ("📊", f"Persona {cid}", ""))
    color = persona_color(cid)
    return f"""
    <div style="{CARD_STYLE} border-left: 6px solid {color}; margin-bottom: 16px;">
        <div style="font-size: 13px; letter-spacing: 0.08em; text-transform: uppercase; color: #b0b0b0;">
            Your closest health persona
        </div>
        <div style="display: flex; align-items: baseline; justify-content: space-between; flex-wrap: wrap; gap: 8px; margin-top: 6px;">
            <span style="font-size: 30px; font-weight: 700; color: #ffffff;">{emoji} {name}</span>
            <span style="font-size: 15px; color: #e0e0e0;"><b>{closeness:.0f}%</b> relative closeness</span>
        </div>
        <div style="font-size: 15px; color: #d0d0d0; margin-top: 10px; line-height: 1.5;">{desc}</div>
        <div style="font-size: 13px; color: #b0b0b0; margin-top: 10px;">
            {share:.1f}% of the 246,000+ survey respondents belong to this persona.
        </div>
    </div>
    """


def persona_gallery(model):
    """Small cards for every persona so users see the whole landscape."""
    sizes = model['sizes']
    cards = []
    for cid in model['cluster_ids']:
        emoji, name, desc = PERSONAS.get(int(cid), ("📊", f"Persona {cid}", ""))
        pct = sizes[cid] / model['n'] * 100
        cards.append(f"""
        <div style="{CARD_STYLE} padding: 14px; border-top: 4px solid {persona_color(int(cid))};">
            <div style="font-size: 22px;">{emoji}</div>
            <div style="font-size: 15px; font-weight: 600; color: #ffffff; margin: 4px 0;">{name}</div>
            <div style="font-size: 12px; color: #b0b0b0; margin-bottom: 6px;">{pct:.1f}% of respondents</div>
            <div style="font-size: 12.5px; color: #d0d0d0; line-height: 1.4;">{desc}</div>
        </div>""")
    st.markdown(
        f"""<div style="display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
                        gap: 12px; margin-bottom: 8px;">{''.join(cards)}</div>""",
        unsafe_allow_html=True
    )


def radar_chart(model, user_values, cid):
    """You vs your persona vs everyone on 8 lifestyle/wellbeing axes, scaled 0-1."""
    axes = ['BMI', 'SleepHours', 'PhysicalActivities', 'SmokerStatus', 'AlcoholDrinkers',
            'GeneralHealth', 'PhysicalHealthDays', 'MentalHealthDays']
    ranges = {'BMI': (15, 50), 'SleepHours': (3, 12), 'PhysicalActivities': (0, 1),
              'SmokerStatus': (0, 3), 'AlcoholDrinkers': (0, 1), 'GeneralHealth': (1, 5),
              'PhysicalHealthDays': (0, 30), 'MentalHealthDays': (0, 30)}
    scaler = model['scaler']
    idx = [CLUSTER_FEATURES.index(a) for a in axes]
    centroid_raw = scaler.inverse_transform(model['centroids'])[list(model['cluster_ids']).index(cid)]
    population_raw = scaler.mean_

    def norm(values):
        return [(v - ranges[a][0]) / (ranges[a][1] - ranges[a][0]) for a, v in zip(axes, values)]

    theta = [FEATURE_LABELS[a] for a in axes] + [FEATURE_LABELS[axes[0]]]
    series = [
        ("Everyone", norm(population_raw[idx]), '#898781', 'dot', None),
        (persona_name(cid), norm(centroid_raw[idx]), persona_color(cid), 'solid', persona_color(cid)),
        ("You", norm([user_values[a] for a in axes]), '#f0efec', 'solid', None),
    ]

    fig = go.Figure()
    for name, r, color, dash, fill in series:
        fig.add_trace(go.Scatterpolar(
            r=r + r[:1], theta=theta, name=name,
            line=dict(color=color, width=2, dash=dash),
            marker=dict(size=8),
            fill='toself' if fill else None,
            fillcolor=f"rgba{tuple(int(fill[i:i + 2], 16) for i in (1, 3, 5)) + (0.18,)}" if fill else None,
            hovertemplate='%{theta}: %{r:.2f}<extra>' + name + '</extra>'
        ))
    fig.update_layout(
        polar=dict(bgcolor='rgba(0,0,0,0)',
                   radialaxis=dict(range=[0, 1], showticklabels=False, gridcolor='#383835', linecolor='#383835'),
                   angularaxis=dict(gridcolor='#383835', linecolor='#383835')),
        legend=dict(orientation='h', yanchor='bottom', y=-0.2, xanchor='center', x=0.5),
        height=440, margin=dict(l=40, r=40, t=30, b=40)
    )
    return fig


def closeness_chart(model, closeness, cid):
    ids = list(model['cluster_ids'])
    order = np.argsort(closeness)
    fig = go.Figure(go.Bar(
        x=closeness[order],
        y=[persona_name(int(ids[i])) for i in order],
        orientation='h',
        marker=dict(color=[persona_color(int(ids[i])) for i in order],
                    opacity=[1.0 if ids[i] == cid else 0.45 for i in order],
                    line=dict(width=0)),
        text=[f"{v:.0f}%" for v in closeness[order]],
        textposition='outside',
        hovertemplate='%{y}: %{x:.1f}% relative closeness<extra></extra>'
    ))
    fig.update_layout(height=360, xaxis_title="Relative closeness (%)",
                      xaxis=dict(range=[0, max(closeness) * 1.2]),
                      margin=dict(l=10, r=20, t=40, b=40), showlegend=False)
    return fig


def persona_map(model, user_scaled, cid, dims=2):
    """PCA projection of a sample of respondents, persona centroids, and the user."""
    pcs = model['sample_pcs']
    labels = model['sample_labels']
    pca = model['pca']
    centroid_pcs = pca.transform(model['centroids'])
    user_pc = pca.transform(user_scaled.reshape(1, -1))[0]
    ids = list(model['cluster_ids'])

    scatter = go.Scatter3d if dims == 3 else go.Scatter
    fig = go.Figure()
    for i, c in enumerate(ids):
        mask = labels == c
        coords = dict(x=pcs[mask, 0], y=pcs[mask, 1])
        if dims == 3:
            coords['z'] = pcs[mask, 2]
        fig.add_trace(scatter(
            **coords, mode='markers', name=persona_name(int(c)), legendgroup=str(c),
            marker=dict(size=3 if dims == 3 else 5, color=persona_color(int(c)),
                        symbol=PERSONA_SYMBOLS[i % len(PERSONA_SYMBOLS)] if dims == 2 else 'circle',
                        opacity=0.35 if dims == 2 else 0.5),
            hovertemplate=persona_name(int(c)) + '<extra></extra>'
        ))

    centroid_coords = dict(x=centroid_pcs[:, 0], y=centroid_pcs[:, 1])
    user_coords = dict(x=[user_pc[0]], y=[user_pc[1]])
    if dims == 3:
        centroid_coords['z'] = centroid_pcs[:, 2]
        user_coords['z'] = [user_pc[2]]

    fig.add_trace(scatter(
        **centroid_coords, mode='markers+text' if dims == 2 else 'markers', name='Persona centers',
        text=[PERSONAS.get(int(c), ("📊",))[0] for c in ids], textposition='top center',
        textfont=dict(size=18),
        marker=dict(size=16 if dims == 2 else 8, color=[persona_color(int(c)) for c in ids],
                    line=dict(color='#f0efec', width=2)),
        hovertext=[persona_name(int(c)) + ' (center)' for c in ids], hoverinfo='text'
    ))
    fig.add_trace(scatter(
        **user_coords, mode='markers+text', name='You', text=['You'], textposition='bottom center',
        textfont=dict(size=14, color='#f0efec'),
        marker=dict(size=24 if dims == 2 else 10, symbol='star' if dims == 2 else 'diamond',
                    color='#f0efec', line=dict(color=persona_color(cid), width=3)),
        hovertemplate='You → ' + persona_name(cid) + '<extra></extra>'
    ))

    var = pca.explained_variance_ratio_
    if dims == 3:
        fig.update_layout(scene=dict(xaxis_title=f"PC1 ({var[0]:.0%})", yaxis_title=f"PC2 ({var[1]:.0%})",
                                     zaxis_title=f"PC3 ({var[2]:.0%})"),
                          height=620, margin=dict(l=0, r=0, t=10, b=0))
    else:
        fig.update_layout(xaxis_title=f"PC1 ({var[0]:.0%} of variance)",
                          yaxis_title=f"PC2 ({var[1]:.0%} of variance)",
                          height=640, margin=dict(l=20, r=20, t=40, b=20))
    fig.update_layout(legend=dict(itemsizing='constant', orientation='h', yanchor='top', y=-0.12,
                                  xanchor='center', x=0.5))
    return fig


def biggest_differences(model, user_scaled, cid, top=3):
    """Features where the user sits furthest from their persona's center (in std units)."""
    centroid = model['centroids'][list(model['cluster_ids']).index(cid)]
    diff = user_scaled - centroid
    order = np.argsort(-np.abs(diff))[:top]
    return [(CLUSTER_FEATURES[i], diff[i]) for i in order]


# ============================================================================
# DATA SCIENCE VIEW
# ============================================================================

def centroid_heatmap(model):
    z = model['centroids']
    names = [persona_name(int(c)) for c in model['cluster_ids']]
    labels = [FEATURE_LABELS[f] for f in CLUSTER_FEATURES]
    limit = float(np.abs(z).max())
    fig = go.Figure(go.Heatmap(
        z=z.T, x=names, y=labels, zmid=0, zmin=-limit, zmax=limit,
        colorscale=[[0, '#2a78d6'], [0.5, '#383835'], [1, '#e34948']],
        text=np.round(z.T, 2), texttemplate='%{text}', textfont=dict(size=11),
        xgap=2, ygap=2,
        colorbar=dict(title='z-score'),
        hovertemplate='%{x}<br>%{y}: %{z:.2f} std from average<extra></extra>'
    ))
    fig.update_layout(height=620, margin=dict(l=10, r=10, t=40, b=10),
                      xaxis=dict(tickangle=-30), yaxis=dict(autorange='reversed'))
    return fig


def silhouette_chart(model):
    ids = list(model['cluster_ids'])
    fig = go.Figure()
    for c in ids:
        vals = model['sample_silhouette'][model['sample_labels'] == c]
        fig.add_trace(go.Box(x=vals, name=persona_name(int(c)), marker_color=persona_color(int(c)),
                             boxmean=True, orientation='h', showlegend=False))
    fig.add_vline(x=0, line=dict(color='#898781', width=1, dash='dot'))
    fig.update_layout(height=420, xaxis_title="Silhouette coefficient",
                      margin=dict(l=10, r=20, t=10, b=40))
    return fig


def loadings_chart(model):
    pca = model['pca']
    labels = [FEATURE_LABELS[f] for f in CLUSTER_FEATURES]
    fig = go.Figure()
    for k, color in zip(range(2), ['#3987e5', '#d95926']):
        fig.add_trace(go.Bar(x=pca.components_[k], y=labels, orientation='h', name=f"PC{k + 1}",
                             marker_color=color))
    fig.update_layout(barmode='group', height=520, xaxis_title="Loading",
                      legend=dict(orientation='h', y=1.05, x=0),
                      margin=dict(l=10, r=20, t=30, b=40))
    return fig


def distance_contributions(model, user_scaled, cid):
    centroid = model['centroids'][list(model['cluster_ids']).index(cid)]
    sq = (user_scaled - centroid) ** 2
    order = np.argsort(sq)
    fig = go.Figure(go.Bar(
        x=sq[order], y=[FEATURE_LABELS[CLUSTER_FEATURES[i]] for i in order], orientation='h',
        marker_color=persona_color(cid),
        hovertemplate='%{y}: %{x:.2f}<extra></extra>'
    ))
    fig.update_layout(height=460, xaxis_title="Squared distance (std units²)",
                      margin=dict(l=10, r=20, t=10, b=40))
    return fig


def render_data_science(model, user_scaled, distances, cid):
    m = model['metrics']
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Respondents", f"{model['n']:,}")
    c2.metric("Clusters (K)", len(model['cluster_ids']))
    c3.metric("Silhouette", f"{m['silhouette']:.3f}", help="Mean over an 8,000-row random sample. Range -1 to 1, higher is better.")
    c4.metric("Davies-Bouldin", f"{m['davies_bouldin']:.2f}", help="Lower is better. Same 8,000-row sample.")

    with st.expander("⚙️ Pipeline", expanded=False):
        st.markdown(f"""
        1. **Features (14):** {', '.join(FEATURE_LABELS[f] for f in CLUSTER_FEATURES)}
        2. **Preprocessing:** median imputation, then `StandardScaler` (z-scores)
        3. **Model:** `KMeans` (n_init=20, random_state={RANDOM_STATE}); K chosen by best silhouette over K=3-8 (`src/clustering.py`)
        4. **Assigning you:** your answers go through the same scaler, then you're matched to the nearest centroid
           (Euclidean), which is exactly how `KMeans.predict` works.
           *Relative closeness* is inverse squared distance, normalised to 100% across personas.
        5. **Visual:** PCA fit on all scaled rows; the map plots a random 8,000-row sample.
        """)

    st.subheader("Centroid fingerprint")
    st.caption("Each cell is how far a persona's average sits from the overall average, in standard deviations. "
               "Red = above average, blue = below.")
    st.plotly_chart(centroid_heatmap(model), use_container_width=True, config=PLOTLY_CONFIG)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Silhouette by persona")
        st.caption("How well each sampled respondent fits its own cluster vs. the next-closest one. "
                   "Values near 0 mean the boundary between personas is fuzzy.")
        st.plotly_chart(silhouette_chart(model), use_container_width=True, config=PLOTLY_CONFIG)
    with col2:
        st.subheader("What drives the map axes")
        var = model['pca'].explained_variance_ratio_
        st.caption(f"PCA loadings. PC1 and PC2 together explain {var[:2].sum():.0%} of the variance, "
                   "so the 2-D map is a simplification. Overlap there doesn't always mean overlap in 14-D.")
        st.plotly_chart(loadings_chart(model), use_container_width=True, config=PLOTLY_CONFIG)

    st.subheader("Your distance breakdown")
    col1, col2 = st.columns([3, 2])
    with col1:
        st.caption(f"Which of your answers pull you away from the {persona_name(cid)} center. "
                   "The bars sum to your squared Euclidean distance.")
        st.plotly_chart(distance_contributions(model, user_scaled, cid), use_container_width=True, config=PLOTLY_CONFIG)
    with col2:
        st.caption("Your Euclidean distance to every centroid (scaled space)")
        dist_df = pd.DataFrame({
            'Persona': [persona_name(int(c)) for c in model['cluster_ids']],
            'Distance': distances,
        }).sort_values('Distance').reset_index(drop=True)
        st.dataframe(dist_df.style.format({'Distance': '{:.2f}'}), use_container_width=True, hide_index=True)

    st.subheader("Persona profiles")
    stats = persona_stats()
    st.dataframe(stats.style.format(precision=1), use_container_width=True)
    st.download_button("⬇️ Download profiles (CSV)", stats.to_csv().encode(), "persona_profiles.csv", "text/csv")

    st.info(
        "**Caveats worth knowing.** The silhouette score is low, so these personas are useful summaries rather "
        "than sharply separated groups. Because binary diagnosis flags (heart attack, depression, diabetes) are "
        "standardised alongside continuous features, a single yes/no can move someone several standard "
        "deviations, so three personas are largely defined by one diagnosis. Re-weighting those flags, "
        "or clustering on behaviour only and profiling outcomes afterwards, would be a good next experiment."
    )


# ============================================================================
# ENTRY POINT
# ============================================================================

def render_cluster_tab():
    st.header("🧬 Health Personas")
    st.markdown("Machine learning (KMeans clustering) grouped 246,000+ Americans into **health personas**: "
                "people with similar habits, wellbeing, and health history. Tell us about yourself to see which one "
                "you're closest to.")

    with st.spinner("Loading clustering model..."):
        model = build_model()

    with st.expander("👀 Meet the personas", expanded=False):
        persona_gallery(model)

    user_values = profile_form()
    cid, user_scaled, distances, closeness = match_user(model, user_values)
    best_pos = list(model['cluster_ids']).index(cid)
    share = model['sizes'][cid] / model['n'] * 100

    view_you, view_map, view_ds = st.tabs(["✨ Your persona", "🗺️ Persona map", "🔬 Under the hood"])

    with view_you:
        st.markdown(hero_card(cid, closeness[best_pos], share), unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("You vs. your persona")
            st.caption("Each axis is scaled from its minimum to maximum possible value.")
            st.plotly_chart(radar_chart(model, user_values, cid), use_container_width=True, config=PLOTLY_CONFIG)
        with col2:
            st.subheader("How close you are to each persona")
            st.caption("Your answers can resemble several personas. Here's how close you are to each one.")
            st.plotly_chart(closeness_chart(model, closeness, cid), use_container_width=True, config=PLOTLY_CONFIG)

        st.markdown("**Where you differ most from this persona**")
        diff_cols = st.columns(3)
        for col, (feature, d) in zip(diff_cols, biggest_differences(model, user_scaled, cid)):
            direction = "higher" if d > 0 else "lower"
            col.markdown(
                f"""<div style="{CARD_STYLE} padding: 14px;">
                    <div style="font-size: 14px; color: #ffffff; font-weight: 600;">{FEATURE_LABELS[feature]}</div>
                    <div style="font-size: 13px; color: #b0b0b0; margin-top: 4px;">
                        Yours is <b style="color:#ffffff;">{direction}</b> than this persona's average
                        by {abs(d):.1f} standard deviations
                    </div></div>""",
                unsafe_allow_html=True
            )

    with view_map:
        st.subheader("Where you sit among everyone")
        dims = st.radio("View", ["2D", "3D"], horizontal=True, key="persona_map_dims")
        st.caption("Each dot is a real survey respondent (random sample of 8,000), projected from 14 dimensions down "
                   "with PCA. Large outlined markers are persona centers; the ⭐ is you. Click legend items to hide personas.")
        st.plotly_chart(persona_map(model, user_scaled, cid, dims=3 if dims == "3D" else 2),
                        use_container_width=True, config=PLOTLY_CONFIG)

    with view_ds:
        render_data_science(model, user_scaled, distances, cid)
