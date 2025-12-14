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

# ============================================================================
# PAGE CONFIG
# ============================================================================

st.set_page_config(
    page_title="Health Risk Explorer",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

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

def get_risk_level(value, thresholds):
    """Get risk level based on value and thresholds."""
    if value <= thresholds[0]:
        return ("Low", "🟢", "lightgreen")
    elif value <= thresholds[1]:
        return ("Moderate", "🟡", "yellow")
    else:
        return ("High", "🔴", "lightcoral")

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

# ============================================================================
# LOAD DATA
# ============================================================================

@st.cache_data
def load_data():
    """Load and prepare data once."""
    heart = pd.read_csv("data/processed/heart_cleaned.csv")
    
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

# ============================================================================
# SIDEBAR - FILTERS
# ============================================================================

st.sidebar.header("🔍 Filter Data")
st.sidebar.markdown("Use filters to explore specific populations. Changes apply to all tabs.")

# Reset filters button
if st.sidebar.button("🔄 Reset All Filters", use_container_width=True):
    st.rerun()

st.sidebar.markdown("---")

# Age filter
age_labels = ["18-24", "25-29", "30-34", "35-39", "40-44", "45-49",
              "50-54", "55-59", "60-64", "65-69", "70-74", "75-79", "80+"]
selected_ages = st.sidebar.multiselect(
    "Age Groups",
    options=age_labels,
    default=age_labels,
    help="Select one or more age groups to analyze"
)

# State filter
all_states = sorted(heart['State'].str.title().unique())
selected_states = st.sidebar.multiselect(
    "States (Optional)",
    options=all_states,
    default=[],  # Empty = all states
    help="Leave empty to see all states, or select specific states"
)

# Lifestyle filters - improved UX with radio buttons
st.sidebar.subheader("Lifestyle Filters")

smoking_filter = st.sidebar.radio(
    "Smoking Status",
    ["All", "Smokers Only", "Non-Smokers Only"],
    help="Filter by smoking status"
)

activity_filter = st.sidebar.radio(
    "Physical Activity",
    ["All", "Active Only", "Inactive Only"],
    help="Filter by physical activity level"
)

# Apply filters
filtered_data = heart.copy()

# Age filter
if not selected_ages:
    st.sidebar.warning("⚠️ Please select at least one age group")
    selected_ages = age_labels  # Default to all if empty
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

st.sidebar.markdown("---")

# Show filtered sample size with warning if too small
sample_size = len(filtered_data)
st.sidebar.metric("Filtered Sample Size", f"{sample_size:,}")
if sample_size < 100:
    st.sidebar.warning("⚠️ Small sample size - results may be less reliable")

# ============================================================================
# MAIN CONTENT - WELCOME SECTION
# ============================================================================

st.title("🏥 U.S. Health Risk Explorer")
st.markdown("### *Interactive analysis of 246,000+ Americans' health data*")

# Welcome/Onboarding section (collapsible)
with st.expander("ℹ️ **Getting Started - Click to expand**", expanded=False):
    st.markdown("""
    **Welcome!** This dashboard helps you explore health patterns and understand your personal risk factors.
    
    **How to use:**
    1. **📊 Overview Tab**: Start here to see disease prevalence and lifestyle patterns
    2. **🎯 Risk Calculator**: Enter your profile to see personalized risk estimates
    3. **🗺️ Geographic Tab**: Compare health outcomes across U.S. states
    4. **📈 Trends Tab**: Explore how health changes with age
    
    **Tips:**
    - Use the sidebar filters to focus on specific populations
    - All filters apply across all tabs
    - Click the "Reset All Filters" button to start fresh
    - Hover over charts for detailed information
    
    **Important:** This tool is for educational purposes only and is not medical advice. 
    Always consult healthcare professionals for medical decisions.
    """)

st.markdown("---")

# Check for empty filtered data
if len(filtered_data) == 0:
    st.error("⚠️ **No data matches your current filters.** Please adjust your filter selections in the sidebar.")
    st.stop()

# Create tabs
tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🎯 Risk Calculator", "🗺️ Geographic", "📈 Trends"])

# ============================================================================
# TAB 1: OVERVIEW
# ============================================================================

with tab1:
    st.header("Disease Prevalence Overview")
    st.markdown("Key health metrics for your selected population")
    
    # Key metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        diabetes_pct = safe_percentage(filtered_data['Diabetes_Binary'] == 1)
        st.metric("Diabetes", f"{diabetes_pct:.1f}%", 
                 help="Percentage with diagnosed diabetes (Type 1 or Type 2)")
    
    with col2:
        heart_pct = safe_percentage(filtered_data['HadHeartAttack'])
        st.metric("Heart Attack", f"{heart_pct:.1f}%",
                 help="Percentage who have had a heart attack")
    
    with col3:
        depression_pct = safe_percentage(filtered_data['HadDepressiveDisorder'])
        st.metric("Depression", f"{depression_pct:.1f}%",
                 help="Percentage with diagnosed depression")
    
    with col4:
        avg_bmi = filtered_data['BMI'].mean()
        if pd.isna(avg_bmi) or np.isinf(avg_bmi):
            avg_bmi = 0.0
        bmi_category = get_bmi_category(avg_bmi)
        st.metric("Avg BMI", f"{avg_bmi:.1f}",
                 help=f"Average Body Mass Index ({bmi_category} range)")
    
    st.markdown("---")
    
    # Interactive disease comparison
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Disease Prevalence")
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
                       marker_color=['#e74c3c', '#e67e22', '#f39c12', '#3498db', '#2ecc71', '#9b59b6'][:len(diseases_clean)],
                       text=[f'{v:.1f}%' for v in diseases_clean.values()],
                       textposition='auto')
            ])
            fig.update_layout(
                xaxis_title="Prevalence (%)",
                height=400,
                showlegend=False,
                margin=dict(l=20, r=20, t=20, b=20)
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No data available for disease prevalence with current filters.")
    
    with col2:
        st.subheader("Lifestyle Distribution")
        st.caption("Percentage of population with healthy lifestyle factors")
        
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
                       marker_color=['#2ecc71', '#3498db', '#e74c3c', '#e67e22', '#9b59b6'][:len(lifestyle_data_clean)],
                       text=[f'{v:.1f}%' for v in lifestyle_data_clean.values()],
                       textposition='auto')
            ])
            fig.update_layout(
                yaxis_title="Percentage (%)",
                height=400,
                showlegend=False,
                margin=dict(l=20, r=20, t=20, b=20)
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No data available for lifestyle distribution with current filters.")

# ============================================================================
# TAB 2: RISK CALCULATOR
# ============================================================================

with tab2:
    st.header("🎯 Personal Risk Calculator")
    st.markdown("**Enter your information to see personalized risk estimates based on similar profiles in the data**")
    st.info("💡 **How it works:** We compare your profile to people with similar characteristics in our dataset to estimate your risk levels.")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Your Profile")
        
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
            st.success(f"**Your BMI: {user_bmi}** ({bmi_category})")
            
            # BMI reference
            with st.expander("📏 BMI Reference Guide"):
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
        st.subheader("Your Risk Profile")
        st.caption(sample_info)
        
        # Risk interpretation guide
        with st.expander("📊 How to Read Your Risk Scores"):
            st.markdown("""
            **Risk Levels:**
            - 🟢 **Low (Green)**: Below average risk
            - 🟡 **Moderate (Yellow)**: Average to slightly elevated risk
            - 🔴 **High (Red)**: Above average risk
            
            **What the numbers mean:**
            - These percentages show the likelihood of having (or developing) each condition
            - Based on people with similar age, BMI, and lifestyle factors
            - Not a diagnosis - consult healthcare professionals for medical advice
            """)
        
        # Gauge charts with better labels
        col_a, col_b, col_c = st.columns(3)
        
        with col_a:
            diabetes_level, diabetes_emoji, diabetes_color = get_risk_level(your_diabetes_risk, [10, 20])
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=your_diabetes_risk,
                title={'text': f"Diabetes Risk<br>{diabetes_emoji} {diabetes_level}"},
                gauge={'axis': {'range': [None, 30]},
                       'bar': {'color': diabetes_color},
                       'steps': [
                           {'range': [0, 10], 'color': "lightgreen"},
                           {'range': [10, 20], 'color': "yellow"},
                           {'range': [20, 30], 'color': "lightcoral"}],
                       'threshold': {'line': {'color': "red", 'width': 4}, 
                                   'thickness': 0.75, 'value': 25}}))
            fig.update_layout(height=250, margin=dict(l=10, r=10, t=60, b=10))
            st.plotly_chart(fig, use_container_width=True)
            st.caption(f"{your_diabetes_risk:.1f}% of similar profiles")
        
        with col_b:
            heart_level, heart_emoji, heart_color = get_risk_level(your_heart_risk, [5, 10])
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=your_heart_risk,
                title={'text': f"Heart Attack Risk<br>{heart_emoji} {heart_level}"},
                gauge={'axis': {'range': [None, 15]},
                       'bar': {'color': heart_color},
                       'steps': [
                           {'range': [0, 5], 'color': "lightgreen"},
                           {'range': [5, 10], 'color': "yellow"},
                           {'range': [10, 15], 'color': "lightcoral"}],
                       'threshold': {'line': {'color': "red", 'width': 4}, 
                                   'thickness': 0.75, 'value': 12}}))
            fig.update_layout(height=250, margin=dict(l=10, r=10, t=60, b=10))
            st.plotly_chart(fig, use_container_width=True)
            st.caption(f"{your_heart_risk:.1f}% of similar profiles")
        
        with col_c:
            depression_level, depression_emoji, depression_color = get_risk_level(your_depression_risk, [15, 25])
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=your_depression_risk,
                title={'text': f"Depression Risk<br>{depression_emoji} {depression_level}"},
                gauge={'axis': {'range': [None, 40]},
                       'bar': {'color': depression_color},
                       'steps': [
                           {'range': [0, 15], 'color': "lightgreen"},
                           {'range': [15, 25], 'color': "yellow"},
                           {'range': [25, 40], 'color': "lightcoral"}],
                       'threshold': {'line': {'color': "red", 'width': 4}, 
                                   'thickness': 0.75, 'value': 35}}))
            fig.update_layout(height=250, margin=dict(l=10, r=10, t=60, b=10))
            st.plotly_chart(fig, use_container_width=True)
            st.caption(f"{your_depression_risk:.1f}% of similar profiles")
        
        # What-if scenarios
        st.markdown("---")
        st.subheader("💡 What If You Changed...")
        st.caption("See how lifestyle changes could impact your diabetes risk")
        
        scenarios = []
        
        # Scenario 1: Improve sleep
        if user_sleep < 7:
            better_sleep = risk_profile[risk_profile['SleepHours'] >= 7]
            if len(better_sleep) > 0:
                new_risk = (better_sleep['Diabetes_Binary'] == 1).mean() * 100
                reduction = your_diabetes_risk - new_risk
                if reduction > 0:
                    scenarios.append(("😴 Increased sleep to 7-9 hours", reduction, new_risk))
        
        # Scenario 2: Start exercising
        if user_active == "No":
            active_group = risk_profile[risk_profile['PhysicalActivities'] == 1]
            if len(active_group) > 0:
                new_risk = (active_group['Diabetes_Binary'] == 1).mean() * 100
                reduction = your_diabetes_risk - new_risk
                if reduction > 0:
                    scenarios.append(("🏃 Started exercising regularly", reduction, new_risk))
        
        # Scenario 3: Lose weight
        if user_bmi >= 25:
            normal_weight = risk_profile[risk_profile['BMI'] < 25]
            if len(normal_weight) > 0:
                new_risk = (normal_weight['Diabetes_Binary'] == 1).mean() * 100
                reduction = your_diabetes_risk - new_risk
                if reduction > 0:
                    scenarios.append(("⚖️ Achieved healthy weight (BMI <25)", reduction, new_risk))
        
        # Scenario 4: Quit smoking
        if user_smoker == "Current":
            nonsmoker = risk_profile[risk_profile['SmokerStatus'] == 0]
            if len(nonsmoker) > 0:
                new_risk = (nonsmoker['Diabetes_Binary'] == 1).mean() * 100
                reduction = your_diabetes_risk - new_risk
                if reduction > 0:
                    scenarios.append(("🚭 Quit smoking", reduction, new_risk))
        
        if scenarios:
            for change, reduction, new_risk in scenarios:
                st.success(f"**{change}** → Risk drops from {your_diabetes_risk:.1f}% to {new_risk:.1f}% (**-{reduction:.1f}%** reduction)")
        else:
            st.info("🌟 **Great job!** You're already following healthy lifestyle habits. Keep it up!")

# ============================================================================
# TAB 3: GEOGRAPHIC
# ============================================================================

with tab3:
    st.header("🗺️ Geographic Health Patterns")
    st.markdown("Compare health outcomes across different U.S. states")
    
    # Select condition to map
    condition_map = {
        'Diabetes': 'Diabetes_Binary',
        'Heart Attack': 'HadHeartAttack',
        'Depression': 'HadDepressiveDisorder',
        'Arthritis': 'HadArthritis',
        'Stroke': 'HadStroke'
    }
    
    selected_condition = st.selectbox(
        "Select Condition to Map", 
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
        st.warning("No data available for the selected filters. Please adjust your filters.")
    else:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader(f"Highest {selected_condition} Rates")
            st.caption("States with the highest prevalence")
            top_15 = state_prevalence.head(15)
            fig = px.bar(top_15, y='State', x='Prevalence', orientation='h',
                         color='Prevalence', color_continuous_scale='Reds',
                         text='Prevalence')
            fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
            fig.update_layout(height=500, showlegend=False, 
                            xaxis_title="Prevalence (%)",
                            margin=dict(l=20, r=20, t=20, b=20))
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader(f"Lowest {selected_condition} Rates")
            st.caption("States with the lowest prevalence")
            bottom_15 = state_prevalence.tail(15)
            fig = px.bar(bottom_15, y='State', x='Prevalence', orientation='h',
                         color='Prevalence', color_continuous_scale='Greens',
                         text='Prevalence')
            fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
            fig.update_layout(height=500, showlegend=False,
                            xaxis_title="Prevalence (%)",
                            margin=dict(l=20, r=20, t=20, b=20))
            st.plotly_chart(fig, use_container_width=True)
        
        # Geographic insights
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            highest_state = state_prevalence.iloc[0]
            st.metric("Highest State", 
                     highest_state['State'], 
                     f"{highest_state['Prevalence']:.1f}%")
        
        with col2:
            lowest_state = state_prevalence.iloc[-1]
            st.metric("Lowest State", 
                     lowest_state['State'], 
                     f"{lowest_state['Prevalence']:.1f}%")
        
        with col3:
            gap = highest_state['Prevalence'] - lowest_state['Prevalence']
            st.metric("Geographic Gap", 
                     f"{gap:.1f}",
                     "percentage points")

# ============================================================================
# TAB 4: TRENDS
# ============================================================================

with tab4:
    st.header("📈 Age & Lifestyle Trends")
    st.markdown("Explore how health metrics change across different age groups")
    
    # Select metric to visualize
    trend_metric = st.selectbox(
        "Select Metric",
        ['Diabetes', 'Heart Attack', 'Depression', 'Average BMI', 'Physical Activity', 'Sleep Hours'],
        help="Choose a health metric to see how it varies by age"
    )
    
    # Calculate trend data
    if trend_metric == 'Diabetes':
        trend_data = filtered_data.groupby('AgeCategory')['Diabetes_Binary'].mean() * 100
        ylabel = "Diabetes Prevalence (%)"
    elif trend_metric == 'Heart Attack':
        trend_data = filtered_data.groupby('AgeCategory')['HadHeartAttack'].mean() * 100
        ylabel = "Heart Attack Prevalence (%)"
    elif trend_metric == 'Depression':
        trend_data = filtered_data.groupby('AgeCategory')['HadDepressiveDisorder'].mean() * 100
        ylabel = "Depression Prevalence (%)"
    elif trend_metric == 'Average BMI':
        trend_data = filtered_data.groupby('AgeCategory')['BMI'].mean()
        ylabel = "Average BMI"
    elif trend_metric == 'Physical Activity':
        trend_data = filtered_data.groupby('AgeCategory')['PhysicalActivities'].mean() * 100
        ylabel = "% Physically Active"
    else:  # Sleep Hours
        trend_data = filtered_data.groupby('AgeCategory')['SleepHours'].mean()
        ylabel = "Average Sleep Hours"
    
    # Plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=age_labels,
        y=trend_data.values,
        mode='lines+markers',
        line=dict(width=3, color='#3498db'),
        marker=dict(size=10),
        fill='tozeroy',
        fillcolor='rgba(52, 152, 219, 0.2)',
        name=trend_metric
    ))
    
    fig.update_layout(
        xaxis_title="Age Group",
        yaxis_title=ylabel,
        height=500,
        hovermode='x unified',
        margin=dict(l=20, r=20, t=20, b=20)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Comparative analysis
    st.markdown("---")
    st.subheader("Lifestyle Impact Comparison")
    st.caption("See how lifestyle factors relate to health outcomes")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # BMI vs Diabetes
        bmi_categories = ['Underweight', 'Normal', 'Overweight', 'Obese', 'Severe Obese']
        diabetes_by_bmi = filtered_data.groupby('BMI_Category')['Diabetes_Binary'].mean() * 100
        
        fig = px.bar(x=bmi_categories, y=diabetes_by_bmi.values,
                    labels={'x': 'BMI Category', 'y': 'Diabetes Prevalence (%)'},
                    title='BMI → Diabetes Risk',
                    color=diabetes_by_bmi.values,
                    color_continuous_scale='Reds')
        fig.update_layout(showlegend=False, height=400,
                         margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Sleep vs Depression
        sleep_categories = ['<5 hrs', '5-6 hrs', '7-8 hrs', '>9 hrs']
        depression_by_sleep = filtered_data.groupby('Sleep_Category')['HadDepressiveDisorder'].mean() * 100
        
        fig = px.bar(x=sleep_categories, y=depression_by_sleep.values,
                    labels={'x': 'Sleep Category', 'y': 'Depression Prevalence (%)'},
                    title='Sleep → Mental Health',
                    color=depression_by_sleep.values,
                    color_continuous_scale='Purples')
        fig.update_layout(showlegend=False, height=400,
                         margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666; padding: 20px;'>
    <p><strong>📊 Data Source:</strong> CDC BRFSS 2022 Survey (246,022 respondents)</p>
    <p><strong>⚠️ Disclaimer:</strong> For educational purposes only. Not medical advice. Always consult healthcare professionals for medical decisions.</p>
    <p style='font-size: 0.9em; margin-top: 10px;'>This dashboard compares your profile to similar individuals in the dataset to provide risk estimates.</p>
    </div>
    """,
    unsafe_allow_html=True
)
