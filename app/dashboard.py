"""
SIMPLE INTERACTIVE HEALTH DASHBOARD
No ML required - just interactive data exploration!

Run with: streamlit run app/dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))
from src.config import CLEANED_HEART_DATA, RANDOM_STATE
from app.cluster_explorer import render_cluster_tab
from app import ui

# PAGE CONFIG

st.set_page_config(
    page_title="Health Risk Explorer",
    page_icon=":material/monitor_heart:",
    layout="wide",
    initial_sidebar_state="collapsed"
)
ui.inject_css()

# HELPER FUNCTIONS

def get_bmi_category(bmi):
    """Get BMI category label."""
    if bmi < 18.5:
        return "Underweight"
    elif bmi < 25:
        return "Normal"
    elif bmi < 30:
        return "Overweight"
    elif bmi < 35:
        return "Obese"
    else:
        return "Severely Obese"

def calculate_bmi(weight_lbs, height_ft, height_in):
    """Calculate BMI from weight and height."""
    height_inches = height_ft * 12 + height_in
    bmi = (weight_lbs / (height_inches ** 2)) * 703
    return round(bmi, 1)

def safe_percentage(series_or_value, multiplier=100):
    """Safely calculate percentage, handling NaN and Inf values."""
    if isinstance(series_or_value, pd.Series):
        result = series_or_value.mean() * multiplier
    else:
        result = series_or_value * multiplier
    
    # Replace NaN or Inf with 0
    if pd.isna(result) or np.isinf(result):
        return 0.0
    return float(result)

PLOTLY_CONFIG = ui.PLOTLY_CONFIG

STATE_CODES = {
    'Alabama': 'AL', 'Alaska': 'AK', 'Arizona': 'AZ', 'Arkansas': 'AR', 'California': 'CA', 'Colorado': 'CO',
    'Connecticut': 'CT', 'Delaware': 'DE', 'District Of Columbia': 'DC', 'Florida': 'FL', 'Georgia': 'GA',
    'Hawaii': 'HI', 'Idaho': 'ID', 'Illinois': 'IL', 'Indiana': 'IN', 'Iowa': 'IA', 'Kansas': 'KS',
    'Kentucky': 'KY', 'Louisiana': 'LA', 'Maine': 'ME', 'Maryland': 'MD', 'Massachusetts': 'MA',
    'Michigan': 'MI', 'Minnesota': 'MN', 'Mississippi': 'MS', 'Missouri': 'MO', 'Montana': 'MT',
    'Nebraska': 'NE', 'Nevada': 'NV', 'New Hampshire': 'NH', 'New Jersey': 'NJ', 'New Mexico': 'NM',
    'New York': 'NY', 'North Carolina': 'NC', 'North Dakota': 'ND', 'Ohio': 'OH', 'Oklahoma': 'OK',
    'Oregon': 'OR', 'Pennsylvania': 'PA', 'Rhode Island': 'RI', 'South Carolina': 'SC', 'South Dakota': 'SD',
    'Tennessee': 'TN', 'Texas': 'TX', 'Utah': 'UT', 'Vermont': 'VT', 'Virginia': 'VA', 'Washington': 'WA',
    'West Virginia': 'WV', 'Wisconsin': 'WI', 'Wyoming': 'WY'
}

# LOAD DATA

@st.cache_data
def load_data():
    """Load and prepare data once."""
    heart = pd.read_csv(str(CLEANED_HEART_DATA))
    
    # Add helpful categories
    age_labels = ["18-24", "25-29", "30-34", "35-39", "40-44", "45-49",
                  "50-54", "55-59", "60-64", "65-69", "70-74", "75-79", "80+"]
    heart['Age_Label'] = heart['AgeCategory'].map(dict(enumerate(age_labels)))
    
    heart['BMI_Category'] = pd.cut(heart['BMI'], 
                                    bins=[0, 18.5, 25, 30, 35, 100],
                                    labels=['Underweight', 'Normal', 'Overweight', 'Obese', 'Severe Obese'])
    
    heart['Sleep_Category'] = pd.cut(heart['SleepHours'],
                                      bins=[0, 5, 7, 9, 24],
                                      labels=['<5 hrs', '5-6 hrs', '7-8 hrs', '>9 hrs'])
    
    heart['Diabetes_Binary'] = (heart['HadDiabetes'] == 3).astype(int)
    
    return heart

heart = load_data()

# MAIN CONTENT - WELCOME SECTION

ui.hero(
    "Health Risk Intelligence Platform",
    "How do everyday habits shape health across America?",
    "Explore CDC survey data from 246,000+ adults, estimate your own risk, and find the health persona "
    "machine learning places you in.",
    [("respondents", "246,022"), ("states & territories", f"{heart['State'].nunique()}"),
     ("health personas", "7"), ("source", "CDC BRFSS 2022")]
)

# Welcome/Onboarding section (collapsible)
with st.expander("**Getting started**", expanded=False, icon=":material/info:"):
    st.markdown("""
    **Welcome!** This dashboard helps you explore health patterns and understand your personal risk factors.
    
    **How to use:**
    1. **Overview**: Start here to see disease prevalence and lifestyle patterns
    2. **Risk calculator**: Enter your profile to see personalized risk estimates
    3. **Health personas**: Find the health persona (ML cluster) you're closest to
    4. **Geography**: Compare health outcomes across U.S. states
    5. **Trends**: Explore how health changes with age
    
    **Tips:**
    - Use the filters below to focus on specific populations
    - All filters apply across all tabs
    - Click the "Reset All Filters" button to start fresh
    - Hover over charts for detailed information
    
    **Important:** This tool is for educational purposes only and is not medical advice. 
    Always consult healthcare professionals for medical decisions.
    """)

# Initialize age_labels for use in filter panel
age_labels = ["18-24", "25-29", "30-34", "35-39", "40-44", "45-49",
              "50-54", "55-59", "60-64", "65-69", "70-74", "75-79", "80+"]

# Filter panel (expandable)
with st.expander("**Filter the population**", expanded=False, icon=":material/tune:"):
    # Header with title and reset button
    header_col1, header_col2 = st.columns([3, 1])
    with header_col1:
        st.markdown("**Use filters to explore specific populations. Changes apply to all tabs.**")
    with header_col2:
        if st.button("Reset all filters", icon=":material/restart_alt:", use_container_width=True):
            st.rerun()
    
    st.markdown("---")
    
    # Three column layout
    col1, col2, col3 = st.columns([1, 1, 1])
    
    # Column 1: Age Groups with checkboxes
    with col1:
        st.markdown("**Age groups**")
        selected_ages = []
        for age in age_labels:
            if st.checkbox(age, value=True, key=f"age_{age}"):
                selected_ages.append(age)
    
    # Column 2: Lifestyle Filters and State
    with col2:
        st.markdown("**Lifestyle**")
        
        smoking_filter = st.radio(
            "Smoking Status",
            ["All", "Smokers Only", "Non-Smokers Only"],
            help="Filter by smoking status",
            index=0,
            key="smoking_filter"
        )
        
        activity_filter = st.radio(
            "Physical Activity",
            ["All", "Active Only", "Inactive Only"],
            help="Filter by physical activity level",
            index=0,
            key="activity_filter"
        )
        
        st.markdown("---")
        st.markdown("**State (optional)**")
        all_states = sorted(heart['State'].str.title().unique())
        selected_states = st.multiselect(
            "Choose options...",
            options=all_states,
            default=[],  # Empty = all states
            help="Leave empty to see all states, or select specific states",
            label_visibility="collapsed",
            key="state_filter"
        )
    
    # Column 3: Filtered Sample Size
    with col3:
        st.markdown("**Filtered sample size**")
        
        # Apply filters to calculate sample size (will be recalculated below)
        temp_filtered = heart.copy()
        
        # Age filter
        if not selected_ages:
            selected_ages = age_labels  # Default to all if empty
        temp_filtered = temp_filtered[temp_filtered['Age_Label'].isin(selected_ages)]
        
        # State filter
        if selected_states:
            temp_filtered = temp_filtered[temp_filtered['State'].str.title().isin(selected_states)]
        
        # Lifestyle filters - clearer logic
        smoker_condition = temp_filtered['SmokerStatus'] >= 2
        if smoking_filter == "Smokers Only":
            temp_filtered = temp_filtered[smoker_condition]
        elif smoking_filter == "Non-Smokers Only":
            temp_filtered = temp_filtered[~smoker_condition]
        
        active_condition = temp_filtered['PhysicalActivities'] == 1
        if activity_filter == "Active Only":
            temp_filtered = temp_filtered[active_condition]
        elif activity_filter == "Inactive Only":
            temp_filtered = temp_filtered[~active_condition]
        
        # Display sample size
        sample_size = len(temp_filtered)
        st.metric("Filtered sample size", f"{sample_size:,}", label_visibility="collapsed")
        if sample_size < 100:
            st.warning("Small sample size", icon=":material/warning:")

# Apply filters to the full dataset (outside expander for use in tabs)
# Get filter values from session state or use defaults
selected_ages = []
for age in age_labels:
    if st.session_state.get(f"age_{age}", True):
        selected_ages.append(age)

if not selected_ages:
    selected_ages = age_labels  # Default to all if empty

smoking_filter = st.session_state.get("smoking_filter", "All")
activity_filter = st.session_state.get("activity_filter", "All")
selected_states = st.session_state.get("state_filter", [])

filtered_data = heart.copy()

# Age filter
filtered_data = filtered_data[filtered_data['Age_Label'].isin(selected_ages)]

# State filter
if selected_states:
    filtered_data = filtered_data[filtered_data['State'].str.title().isin(selected_states)]

# Lifestyle filters - clearer logic
smoker_condition = filtered_data['SmokerStatus'] >= 2
if smoking_filter == "Smokers Only":
    filtered_data = filtered_data[smoker_condition]
elif smoking_filter == "Non-Smokers Only":
    filtered_data = filtered_data[~smoker_condition]

active_condition = filtered_data['PhysicalActivities'] == 1
if activity_filter == "Active Only":
    filtered_data = filtered_data[active_condition]
elif activity_filter == "Inactive Only":
    filtered_data = filtered_data[~active_condition]

# Check for empty filtered data
if len(filtered_data) == 0:
    st.error("**No data matches your current filters.** Please adjust your filter selections above.", icon=":material/error:")
    st.stop()

# Create tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    ":material/insights: Overview",
    ":material/calculate: Risk calculator",
    ":material/hub: Health personas",
    ":material/map: Geography",
    ":material/trending_up: Trends",
])

# TAB 1: OVERVIEW

with tab1:
    ui.section("Disease prevalence", "Key health metrics for your selected population", "activity")
    
    # Key metrics row with progress bars
    col1, col2, col3, col4 = st.columns(4)
    
    # Calculate percentages
    diabetes_pct = safe_percentage(filtered_data['Diabetes_Binary'] == 1)
    heart_pct = safe_percentage(filtered_data['HadHeartAttack'])
    depression_pct = safe_percentage(filtered_data['HadDepressiveDisorder'])
    obesity_pct = safe_percentage(filtered_data['BMI'] >= 30)
    
    with col1:
        st.markdown(ui.stat_card("Diabetes", f"{diabetes_pct:.1f}%", diabetes_pct, ui.SERIES[0], "droplet"),
                    unsafe_allow_html=True)
    with col2:
        st.markdown(ui.stat_card("Heart attack", f"{heart_pct:.1f}%", heart_pct, ui.SERIES[0], "heart-pulse"),
                    unsafe_allow_html=True)
    with col3:
        st.markdown(ui.stat_card("Depression", f"{depression_pct:.1f}%", depression_pct, ui.SERIES[0], "brain"),
                    unsafe_allow_html=True)
    with col4:
        st.markdown(ui.stat_card("Obesity (BMI 30+)", f"{obesity_pct:.1f}%", obesity_pct, ui.SERIES[0], "scale"),
                    unsafe_allow_html=True)

    st.write("")
    
    # Interactive disease comparison
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Disease prevalence")
        st.caption("Compare prevalence rates across different conditions")
        
        diseases = {
            'Depression': safe_percentage(filtered_data['HadDepressiveDisorder']),
            'Arthritis': safe_percentage(filtered_data['HadArthritis']),
            'Diabetes': diabetes_pct,
            'Heart Attack': heart_pct,
            'Stroke': safe_percentage(filtered_data['HadStroke']),
            'COPD': safe_percentage(filtered_data['HadCOPD'])
        }
        
        # Ensure all values are finite (handle NaN/Inf)
        diseases_clean = {k: (v if np.isfinite(v) else 0.0) for k, v in diseases.items()}
        
        if diseases_clean:
            fig = go.Figure(data=[
                go.Bar(x=list(diseases_clean.values()), 
                       y=list(diseases_clean.keys()),
                       orientation='h',
                       marker_color=ui.SERIES[0],
                       text=[f'{v:.1f}%' for v in diseases_clean.values()],
                       textposition='outside', cliponaxis=False)
            ])
            fig.update_layout(
                xaxis_title="Prevalence (%)",
                height=400,
                showlegend=False,
                margin=dict(l=20, r=20, t=20, b=20)
            )
            ui.chart(fig)
        else:
            st.info("No data available for disease prevalence with current filters.")
    
    with col2:
        st.subheader("Lifestyle snapshot")
        st.caption("Share of the selected population with each lifestyle factor")
        
        lifestyle_data = {
            'Physically Active': safe_percentage(filtered_data['PhysicalActivities']),
            'Get 7-9hrs Sleep': safe_percentage((filtered_data['SleepHours'] >= 7) & 
                                (filtered_data['SleepHours'] <= 9)),
            'Current Smokers': safe_percentage(filtered_data['SmokerStatus'] >= 2),
            'Overweight/Obese': safe_percentage(filtered_data['BMI'] >= 25),
            'Recent Checkup': safe_percentage(filtered_data['LastCheckupTime'] == 0),
        }
        
        # Ensure all values are finite (handle NaN/Inf)
        lifestyle_data_clean = {k: (v if np.isfinite(v) else 0.0) for k, v in lifestyle_data.items()}
        
        if lifestyle_data_clean:
            fig = go.Figure(data=[
                go.Bar(x=list(lifestyle_data_clean.keys()), 
                       y=list(lifestyle_data_clean.values()),
                       marker_color=ui.SERIES[0],
                       text=[f'{v:.1f}%' for v in lifestyle_data_clean.values()],
                       textposition='outside', cliponaxis=False)
            ])
            fig.update_layout(
                yaxis_title="Percentage (%)",
                height=400,
                showlegend=False,
                margin=dict(l=20, r=20, t=20, b=20)
            )
            ui.chart(fig)
        else:
            st.info("No data available for lifestyle distribution with current filters.")

# TAB 2: RISK CALCULATOR

with tab2:
    ui.section("Personal risk calculator",
               "See how often people with a profile like yours report each condition", "stethoscope")
    ui.callout("<b>How it works:</b> we find respondents with your age group, a BMI within ±2, and the same activity "
               "and smoking status, then show how common each condition is among them.")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Your profile")
        
        # BMI Calculator option
        bmi_method = st.radio(
            "BMI Input Method",
            ["Enter BMI directly", "Calculate from height & weight"],
            help="BMI (Body Mass Index) is a measure of body fat based on height and weight"
        )
        
        if bmi_method == "Calculate from height & weight":
            col_h1, col_h2, col_w = st.columns(3)
            with col_h1:
                height_ft = st.number_input("Feet", min_value=4, max_value=7, value=5, step=1)
            with col_h2:
                height_in = st.number_input("Inches", min_value=0, max_value=11, value=10, step=1)
            with col_w:
                weight_lbs = st.number_input("Weight (lbs)", min_value=80, max_value=400, value=170, step=1)
            
            user_bmi = calculate_bmi(weight_lbs, height_ft, height_in)
            bmi_category = get_bmi_category(user_bmi)
            st.success(f"**Your BMI: {user_bmi}** ({bmi_category})", icon=":material/scale:")
            
            # BMI reference
            with st.expander("BMI reference guide", icon=":material/straighten:"):
                st.markdown("""
                - **Underweight**: BMI < 18.5
                - **Normal**: BMI 18.5 - 24.9
                - **Overweight**: BMI 25 - 29.9
                - **Obese**: BMI 30 - 34.9
                - **Severely Obese**: BMI ≥ 35
                """)
        else:
            user_bmi = st.slider("BMI", 15.0, 50.0, 27.0, 0.5,
                               help="Body Mass Index - a measure of body fat")
            bmi_category = get_bmi_category(user_bmi)
            st.caption(f"Category: {bmi_category}")
        
        user_age = st.selectbox("Age Group", options=age_labels, index=5,
                               help="Select your age range")
        user_sleep = st.slider("Sleep Hours/Night", 4.0, 10.0, 7.0, 0.5,
                              help="Average hours of sleep per night")
        user_active = st.radio("Physically Active?", ["Yes", "No"],
                              help="Do you engage in regular physical activity?")
        user_smoker = st.radio("Smoking Status", ["Never", "Former", "Current"],
                              help="Your current or past smoking status")
        
        st.markdown("---")
        
        # Calculate risk based on filters
        risk_profile = filtered_data.copy()
        
        # Filter to similar profiles
        age_idx = age_labels.index(user_age)
        risk_profile = risk_profile[risk_profile['AgeCategory'] == age_idx]
        risk_profile = risk_profile[risk_profile['BMI'].between(user_bmi - 2, user_bmi + 2)]
        
        # Additional filters based on user input
        if user_active == "Yes":
            risk_profile = risk_profile[risk_profile['PhysicalActivities'] == 1]
        else:
            risk_profile = risk_profile[risk_profile['PhysicalActivities'] == 0]
        
        smoker_map = {"Never": 0, "Former": 1, "Current": 2}
        if user_smoker in smoker_map:
            if user_smoker == "Current":
                risk_profile = risk_profile[risk_profile['SmokerStatus'] >= 2]
            elif user_smoker == "Former":
                risk_profile = risk_profile[risk_profile['SmokerStatus'] == 1]
            else:
                risk_profile = risk_profile[risk_profile['SmokerStatus'] == 0]
        
        if len(risk_profile) > 0:
            your_diabetes_risk = (risk_profile['Diabetes_Binary'] == 1).mean() * 100
            your_heart_risk = risk_profile['HadHeartAttack'].mean() * 100
            your_depression_risk = risk_profile['HadDepressiveDisorder'].mean() * 100
            sample_info = f"Based on {len(risk_profile):,} similar profiles"
        else:
            # Fallback to filtered data averages
            your_diabetes_risk = (filtered_data['Diabetes_Binary'] == 1).mean() * 100
            your_heart_risk = filtered_data['HadHeartAttack'].mean() * 100
            your_depression_risk = filtered_data['HadDepressiveDisorder'].mean() * 100
            sample_info = "Based on general population (limited similar profiles found)"
    
    with col2:
        st.subheader("Your risk profile")
        st.caption(sample_info)
        
        # Risk interpretation guide
        with st.expander("How to read your risk scores", icon=":material/help:"):
            st.markdown("""
            **Risk levels:**
            - **Low**: below average risk
            - **Moderate**: average to slightly elevated risk
            - **High**: above average risk
            
            **What the numbers mean:**
            - These percentages show the likelihood of having (or developing) each condition
            - Based on people with similar age, BMI, and lifestyle factors
            - Not a diagnosis - consult healthcare professionals for medical advice
            """)
        
        # Calculate average risks for comparison
        avg_diabetes = safe_percentage(filtered_data['Diabetes_Binary'] == 1)
        avg_heart = safe_percentage(filtered_data['HadHeartAttack'])
        
        # Calculate BMI/Obesity risk based on user's actual BMI category
        # Risk score: 0-100 scale where higher = higher risk
        if user_bmi < 18.5:
            # Underweight - HIGH risk
            your_bmi_risk = 75.0  # High risk score
            bmi_category_name = "Underweight"
        elif user_bmi < 25.0:
            # Normal weight - LOW risk
            your_bmi_risk = 15.0  # Low risk score
            bmi_category_name = "Normal Weight"
        elif user_bmi < 30.0:
            # Overweight - MODERATE risk
            your_bmi_risk = 50.0  # Moderate risk score
            bmi_category_name = "Overweight"
        else:
            # Obese - HIGH risk
            your_bmi_risk = 85.0  # High risk score
            bmi_category_name = "Obese"
        
        # Calculate average BMI risk for comparison (percentage of population in high-risk categories)
        avg_obesity_risk = safe_percentage((filtered_data['BMI'] < 18.5) | (filtered_data['BMI'] >= 25))
        
        # Calculate comparisons
        diabetes_diff = your_diabetes_risk - avg_diabetes
        heart_diff = your_heart_risk - avg_heart
        # For BMI, compare to average risk level
        bmi_diff = your_bmi_risk - 50.0  # Compare to middle point (50)
        
        def create_risk_card(title, value, risk_level, level_text, color, comparison_diff, similar_pct,
                             bar_value=None, icon_name="activity", unit="%", footnote=None):
            """Risk card: value, status pill (icon + label), meter, and comparison to average."""
            if bar_value is None:
                bar_value = value
            bar_value = min(max(bar_value, 0.0), 100.0)

            if "Obesity" in title:
                desc_text = {"LOW": "Healthy weight range",
                             "MODERATE": "Consider more physical activity",
                             "HIGH": "Consider talking with a healthcare provider"}[risk_level]
            else:
                desc_text = {"LOW": "Below average risk",
                             "MODERATE": "Moderate risk level",
                             "HIGH": "High risk level"}[risk_level]

            if footnote is None:
                trend = "trending-up" if comparison_diff > 0 else "trending-down"
                footnote = (f'{ui.icon(trend, 14)} {comparison_diff:+.1f} pts vs. average for your filters')

            return f"""
            <div class="hr-card">
                <div class="hr-row" style="justify-content:space-between;align-items:center;">
                    <span style="display:flex;gap:10px;align-items:center;font-size:14px;font-weight:500;color:{ui.TEXT_2};">
                        {ui.icon_badge(icon_name, ui.ACCENT, 32)}{title}</span>
                </div>
                <div class="hr-row" style="justify-content:space-between;align-items:center;margin:14px 0 12px;">
                    <span style="font-size:32px;font-weight:600;color:{ui.TEXT};">{value:.1f}{unit}</span>
                    {ui.status_pill(risk_level)}
                </div>
                <div class="hr-meter"><div style="width:{bar_value:.1f}%;background:{color};"></div></div>
                <div style="font-size:13px;color:{ui.TEXT_2};margin-top:12px;">{desc_text}</div>
                <div style="font-size:12.5px;color:{ui.MUTED};margin-top:6px;display:flex;gap:6px;align-items:center;">{footnote}</div>
            </div>
            """

        # Determine risk levels and colors based on BMI category
        if user_bmi < 18.5:
            # Underweight - HIGH risk
            bmi_level = "HIGH"
            bmi_color = ui.STATUS["HIGH"][0]
        elif user_bmi < 25.0:
            # Normal weight - LOW risk
            bmi_level = "LOW"
            bmi_color = ui.STATUS["LOW"][0]
        elif user_bmi < 30.0:
            # Overweight - MODERATE risk
            bmi_level = "MODERATE"
            bmi_color = ui.STATUS["MODERATE"][0]
        else:
            # Obese - HIGH risk
            bmi_level = "HIGH"
            bmi_color = ui.STATUS["HIGH"][0]
        
        # Diabetes thresholds: low < 10, moderate 10-20, high > 20
        if your_diabetes_risk < 10:
            diabetes_level = "LOW"
            diabetes_color = ui.STATUS["LOW"][0]
        elif your_diabetes_risk < 20:
            diabetes_level = "MODERATE"
            diabetes_color = ui.STATUS["MODERATE"][0]
        else:
            diabetes_level = "HIGH"
            diabetes_color = ui.STATUS["HIGH"][0]
        
        # Heart Attack thresholds: low < 5, moderate 5-10, high > 10
        if your_heart_risk < 5:
            heart_level = "LOW"
            heart_color = ui.STATUS["LOW"][0]
        elif your_heart_risk < 10:
            heart_level = "MODERATE"
            heart_color = ui.STATUS["MODERATE"][0]
        else:
            heart_level = "HIGH"
            heart_color = ui.STATUS["HIGH"][0]
        
        # Display risk cards horizontally (side by side)
        card_col1, card_col2, card_col3 = st.columns(3)
        
        with card_col1:
            # Display Obesity Risk card
            # Pass user_bmi for display, but use your_bmi_risk for progress bar
            # We need to modify the card to accept both values
            st.markdown(create_risk_card("Weight (BMI)", user_bmi, bmi_level, "", bmi_color, bmi_diff, your_bmi_risk,
                                         bar_value=your_bmi_risk, icon_name="scale", unit="",
                                         footnote=f"Category: {bmi_category_name}"),
                        unsafe_allow_html=True)
        
        with card_col2:
            st.markdown(create_risk_card("Heart attack", your_heart_risk, heart_level, "", heart_color, heart_diff, your_heart_risk,
                                         icon_name="heart-pulse"),
                       unsafe_allow_html=True)
        
        with card_col3:
            st.markdown(create_risk_card("Diabetes", your_diabetes_risk, diabetes_level, "", diabetes_color, diabetes_diff, your_diabetes_risk,
                                         icon_name="droplet"),
                       unsafe_allow_html=True)
        
        # What-if scenarios
        st.write("")
        st.subheader("What if you changed...")
        st.caption("Risk among people like you who already have the healthier habit")
        
        # Use a broader comparison dataset (same age and BMI range, but allow different lifestyle factors)
        comparison_profile = filtered_data.copy()
        age_idx = age_labels.index(user_age)
        comparison_profile = comparison_profile[comparison_profile['AgeCategory'] == age_idx]
        comparison_profile = comparison_profile[comparison_profile['BMI'].between(user_bmi - 2, user_bmi + 2)]
        
        scenarios = []
        
        def calculate_risk_reductions(improved_group):
            """Calculate risk reductions for all three metrics."""
            if len(improved_group) == 0:
                return None
            
            new_diabetes = safe_percentage(improved_group['Diabetes_Binary'] == 1)
            new_heart = safe_percentage(improved_group['HadHeartAttack'])
            
            # For BMI risk, calculate based on BMI distribution in improved group
            improved_bmi_avg = improved_group['BMI'].mean()
            if pd.isna(improved_bmi_avg):
                new_bmi_risk = your_bmi_risk
            elif improved_bmi_avg < 18.5:
                new_bmi_risk = 75.0
            elif improved_bmi_avg < 25.0:
                new_bmi_risk = 15.0
            elif improved_bmi_avg < 30.0:
                new_bmi_risk = 50.0
            else:
                new_bmi_risk = 85.0
            
            return {
                'diabetes': (your_diabetes_risk - new_diabetes, new_diabetes),
                'heart': (your_heart_risk - new_heart, new_heart),
                'bmi': (your_bmi_risk - new_bmi_risk, new_bmi_risk)
            }
        
        # Scenario 1: Improve sleep (if currently getting less than 7 hours)
        if user_sleep < 7:
            better_sleep = comparison_profile[comparison_profile['SleepHours'] >= 7]
            reductions = calculate_risk_reductions(better_sleep)
            if reductions is None:
                # Fallback to filtered_data if comparison_profile is too small
                better_sleep_fallback = filtered_data[filtered_data['SleepHours'] >= 7]
                reductions = calculate_risk_reductions(better_sleep_fallback)
            if reductions and (reductions['diabetes'][0] > 0 or reductions['heart'][0] > 0):
                scenarios.append((("moon", "Sleep 7-9 hours a night"), reductions))
        
        # Scenario 2: Start exercising (if currently inactive)
        if user_active == "No":
            active_group = comparison_profile[comparison_profile['PhysicalActivities'] == 1]
            reductions = calculate_risk_reductions(active_group)
            if reductions is None:
                # Fallback to filtered_data
                active_fallback = filtered_data[filtered_data['PhysicalActivities'] == 1]
                reductions = calculate_risk_reductions(active_fallback)
            if reductions and (reductions['diabetes'][0] > 0 or reductions['heart'][0] > 0):
                scenarios.append((("activity", "Exercise regularly"), reductions))
        
        # Scenario 3: Weight management based on BMI category
        if user_bmi < 18.5:
            # Underweight - advise healthy eating (no risk reduction calculation needed)
            scenarios.append((("leaf", "Focus on healthy eating and nutrition"), None))
        elif user_bmi >= 25 and user_bmi < 30:
            # Overweight - advise more physical activity
            active_normal_weight = comparison_profile[
                (comparison_profile['BMI'] < 25) & 
                (comparison_profile['PhysicalActivities'] == 1)
            ]
            reductions = calculate_risk_reductions(active_normal_weight)
            if reductions and (reductions['diabetes'][0] > 0 or reductions['heart'][0] > 0 or reductions['bmi'][0] > 0):
                scenarios.append((("activity", "Be more active and reach a healthy weight (BMI < 25)"), reductions))
            else:
                scenarios.append((("activity", "Be more physically active to reduce weight"), None))
        elif user_bmi >= 30:
            # Obese - advise weight loss
            healthy_bmi = comparison_profile[comparison_profile['BMI'] < 25]
            reductions = calculate_risk_reductions(healthy_bmi)
            if reductions is None:
                # Fallback: compare to people with lower BMI in same age group
                age_filtered = filtered_data[filtered_data['AgeCategory'] == age_idx]
                healthy_bmi_fallback = age_filtered[age_filtered['BMI'] < 25]
                reductions = calculate_risk_reductions(healthy_bmi_fallback)
            if reductions and (reductions['diabetes'][0] > 0 or reductions['heart'][0] > 0 or reductions['bmi'][0] > 0):
                scenarios.append((("scale", "Reach a healthy weight (BMI < 25)"), reductions))
        
        # Scenario 4: Quit smoking (if currently smoking)
        if user_smoker == "Current":
            nonsmoker = comparison_profile[comparison_profile['SmokerStatus'] == 0]
            reductions = calculate_risk_reductions(nonsmoker)
            if reductions is None:
                # Fallback to filtered_data
                nonsmoker_fallback = filtered_data[filtered_data['SmokerStatus'] == 0]
                reductions = calculate_risk_reductions(nonsmoker_fallback)
            if reductions and (reductions['diabetes'][0] > 0 or reductions['heart'][0] > 0):
                scenarios.append((("ban", "Quit smoking"), reductions))
        
        # Display scenarios
        def scenario_card(icon_name, change, changes_html):
            return f"""
            <div class="hr-card" style="padding:16px;margin-bottom:10px;">
                <div class="hr-row" style="align-items:center;">
                    {ui.icon_badge(icon_name, ui.STATUS["LOW"][0], 38)}
                    <div><div style="font-size:15px;font-weight:600;color:{ui.TEXT};">{change}</div>
                    <div style="font-size:13.5px;color:{ui.TEXT_2};margin-top:4px;">{changes_html}</div></div>
                </div>
            </div>"""

        if scenarios:
            for (icon_name, change), reductions in scenarios:
                if reductions is None:
                    st.markdown(scenario_card(icon_name, change,
                                              "This can help improve your overall health and reduce disease risk."),
                                unsafe_allow_html=True)
                    continue

                risk_changes = []
                if reductions['diabetes'][0] > 0:
                    risk_changes.append(f"Diabetes <b>{your_diabetes_risk:.1f}% → {reductions['diabetes'][1]:.1f}%</b>")
                if reductions['heart'][0] > 0:
                    risk_changes.append(f"Heart attack <b>{your_heart_risk:.1f}% → {reductions['heart'][1]:.1f}%</b>")
                if reductions['bmi'][0] > 0:
                    risk_changes.append(f"Weight risk score <b>{your_bmi_risk:.0f} → {reductions['bmi'][1]:.0f}</b>")
                if risk_changes:
                    st.markdown(scenario_card(icon_name, change, " &nbsp;·&nbsp; ".join(risk_changes)),
                                unsafe_allow_html=True)
        else:
            # Only show positive message if user actually has good habits
            has_issues = (user_sleep < 7) or (user_active == "No") or (user_bmi < 18.5) or (user_bmi >= 25) or (user_smoker == "Current")
            if has_issues:
                st.warning("**Unable to calculate improvement scenarios.** Try adjusting your filters.", icon=":material/warning:")
            else:
                ui.callout("<b>Great job!</b> You're already following healthy lifestyle habits. Keep it up!",
                           "check", ui.STATUS["LOW"][0])

# ============================================================================
# TAB 3: HEALTH PERSONAS (clustering)
# ============================================================================

with tab3:
    render_cluster_tab()

# ============================================================================
# TAB 4: GEOGRAPHIC
# ============================================================================

with tab4:
    ui.section("Geographic health patterns", "Compare health outcomes across U.S. states", "map-pin")

    # Select condition to map
    condition_map = {
        'Diabetes': 'Diabetes_Binary',
        'Heart Attack': 'HadHeartAttack',
        'Depression': 'HadDepressiveDisorder',
        'Arthritis': 'HadArthritis',
        'Stroke': 'HadStroke'
    }

    selected_condition = st.selectbox(
        "Condition",
        list(condition_map.keys()),
        help="Choose a health condition to see state-by-state comparisons"
    )

    # Calculate state-level prevalence
    state_prevalence = filtered_data.groupby('State')[condition_map[selected_condition]].mean() * 100
    state_prevalence = state_prevalence.reset_index()
    state_prevalence.columns = ['State', 'Prevalence']
    state_prevalence['State'] = state_prevalence['State'].str.title()
    state_prevalence = state_prevalence.sort_values('Prevalence', ascending=False)

    if len(state_prevalence) == 0:
        st.warning("No data available for the selected filters. Please adjust your filters.", icon=":material/warning:")
    else:
        highest_state = state_prevalence.iloc[0]
        lowest_state = state_prevalence.iloc[-1]
        gap = highest_state['Prevalence'] - lowest_state['Prevalence']

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(ui.stat_card("Highest", highest_state['State'], icon_name="trending-up",
                                     note=f"{highest_state['Prevalence']:.1f}% {selected_condition.lower()} prevalence"),
                        unsafe_allow_html=True)
        with col2:
            st.markdown(ui.stat_card("Lowest", lowest_state['State'], icon_name="trending-down",
                                     note=f"{lowest_state['Prevalence']:.1f}% {selected_condition.lower()} prevalence"),
                        unsafe_allow_html=True)
        with col3:
            st.markdown(ui.stat_card("Geographic gap", f"{gap:.1f} pts", icon_name="map-pin",
                                     note="Difference between highest and lowest state"),
                        unsafe_allow_html=True)

        # Choropleth (50 states + DC; territories listed in the bar charts below)
        map_df = state_prevalence.assign(Code=state_prevalence['State'].map(STATE_CODES)).dropna(subset=['Code'])
        fig = go.Figure(go.Choropleth(
            locations=map_df['Code'], z=map_df['Prevalence'], locationmode='USA-states',
            text=map_df['State'],
            colorscale=[[0, '#184f95'], [0.5, '#5598e7'], [1, '#cde2fb']],
            marker_line_color=ui.BG, marker_line_width=1,
            colorbar=dict(title=dict(text="%", side='top'), thickness=12, len=0.7, outlinewidth=0,
                          tickfont=dict(color=ui.TEXT_2)),
            hovertemplate='<b>%{text}</b><br>%{z:.1f}% ' + selected_condition.lower() + '<extra></extra>'
        ))
        fig.update_layout(
            geo=dict(scope='usa', bgcolor='rgba(0,0,0,0)', lakecolor='rgba(0,0,0,0)',
                     showlakes=False, landcolor=ui.SURFACE_2),
            height=520, margin=dict(l=0, r=0, t=10, b=0), dragmode=False
        )
        ui.chart(fig)
        st.caption("Lighter states have higher prevalence. Hover a state for its rate.")

        col1, col2 = st.columns(2)
        for col, title, data in [(col1, "Highest 15", state_prevalence.head(15)),
                                 (col2, "Lowest 15", state_prevalence.tail(15))]:
            with col:
                st.subheader(f"{title}: {selected_condition.lower()}")
                fig = go.Figure(go.Bar(
                    x=data['Prevalence'], y=data['State'], orientation='h',
                    marker_color=ui.SERIES[0],
                    text=[f"{v:.1f}%" for v in data['Prevalence']], textposition='outside', cliponaxis=False,
                    hovertemplate='%{y}: %{x:.1f}%<extra></extra>'
                ))
                fig.update_layout(height=500, xaxis_title="Prevalence (%)",
                                  yaxis=dict(autorange='reversed'), margin=dict(l=20, r=40, t=10, b=20))
                ui.chart(fig)

# ============================================================================
# TAB 5: TRENDS
# ============================================================================

with tab5:
    ui.section("Age & lifestyle trends", "How health metrics change across age groups", "trending-up")

    # Select metric to visualize
    trend_metric = st.selectbox(
        "Metric",
        ['Diabetes', 'Heart Attack', 'Depression', 'Average BMI', 'Physical Activity', 'Sleep Hours'],
        help="Choose a health metric to see how it varies by age"
    )

    # Calculate trend data
    if trend_metric == 'Diabetes':
        trend_data = filtered_data.groupby('AgeCategory')['Diabetes_Binary'].mean() * 100
        ylabel = "Diabetes prevalence (%)"
    elif trend_metric == 'Heart Attack':
        trend_data = filtered_data.groupby('AgeCategory')['HadHeartAttack'].mean() * 100
        ylabel = "Heart attack prevalence (%)"
    elif trend_metric == 'Depression':
        trend_data = filtered_data.groupby('AgeCategory')['HadDepressiveDisorder'].mean() * 100
        ylabel = "Depression prevalence (%)"
    elif trend_metric == 'Average BMI':
        trend_data = filtered_data.groupby('AgeCategory')['BMI'].mean()
        ylabel = "Average BMI"
    elif trend_metric == 'Physical Activity':
        trend_data = filtered_data.groupby('AgeCategory')['PhysicalActivities'].mean() * 100
        ylabel = "% physically active"
    else:  # Sleep Hours
        trend_data = filtered_data.groupby('AgeCategory')['SleepHours'].mean()
        ylabel = "Average sleep (hours)"

    # Plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[age_labels[int(i)] for i in trend_data.index],
        y=trend_data.values,
        mode='lines+markers',
        line=dict(width=2, color=ui.SERIES[0], shape='spline', smoothing=0.6),
        marker=dict(size=8, color=ui.SERIES[0], line=dict(color=ui.BG, width=2)),
        fill='tozeroy',
        fillcolor=ui.rgba(ui.SERIES[0], 0.12),
        name=trend_metric,
        hovertemplate='%{x}: %{y:.1f}<extra></extra>'
    ))

    fig.update_layout(
        xaxis_title="Age group",
        yaxis_title=ylabel,
        height=460,
        hovermode='x',
        margin=dict(l=20, r=20, t=20, b=20)
    )

    ui.chart(fig)

    # Comparative analysis
    st.subheader("Lifestyle impact")
    st.caption("How lifestyle factors relate to health outcomes")

    col1, col2 = st.columns(2)

    with col1:
        # BMI vs Diabetes
        st.markdown("**Diabetes by BMI category**")
        diabetes_by_bmi = filtered_data.groupby('BMI_Category', observed=False)['Diabetes_Binary'].mean() * 100
        fig = go.Figure(go.Bar(x=diabetes_by_bmi.index.astype(str), y=diabetes_by_bmi.values,
                               marker_color=ui.SERIES[0],
                               text=[f"{v:.1f}%" for v in diabetes_by_bmi.values], textposition='outside',
                               cliponaxis=False, hovertemplate='%{x}: %{y:.1f}%<extra></extra>'))
        fig.update_layout(
                          xaxis_title="BMI category", yaxis_title="Diabetes prevalence (%)",
                          height=400, margin=dict(l=20, r=20, t=40, b=20))
        ui.chart(fig)

    with col2:
        # Sleep vs Depression
        st.markdown("**Depression by nightly sleep**")
        depression_by_sleep = filtered_data.groupby('Sleep_Category', observed=False)['HadDepressiveDisorder'].mean() * 100
        fig = go.Figure(go.Bar(x=depression_by_sleep.index.astype(str), y=depression_by_sleep.values,
                               marker_color=ui.SERIES[0],
                               text=[f"{v:.1f}%" for v in depression_by_sleep.values], textposition='outside',
                               cliponaxis=False, hovertemplate='%{x}: %{y:.1f}%<extra></extra>'))
        fig.update_layout(
                          xaxis_title="Sleep", yaxis_title="Depression prevalence (%)",
                          height=400, margin=dict(l=20, r=20, t=40, b=20))
        ui.chart(fig)

# ============================================================================
# FOOTER
# ============================================================================

ui.footer(
    f"{ui.icon('database', 15)} <b>Data:</b> CDC BRFSS 2022 survey (445,132 original respondents; "
    "246,022 complete rows after cleaning)<br>"
    f"{ui.icon('info', 15)} <b>Disclaimer:</b> For educational purposes only. Not medical advice. "
    "Always consult healthcare professionals for medical decisions.<br>"
    "Built by <b>Ayomide Isinkaye</b> · "
    "<a href='https://github.com/avisink/Health_ML_Project' style='color:#be96e3;'>Source on GitHub</a>"
)
