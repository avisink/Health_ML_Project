"""
SIMPLE VISUAL HEALTH ANALYSIS
just great visuals that tell a story!

Goal: Create 5-8 killer visualizations for your presentation in 30 minutes
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
from src.config import CLEANED_HEART_DATA, FIGURES_DIR, RANDOM_STATE

sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 10

PRESENTATION_DIR = FIGURES_DIR.parent / "presentation_visuals"
PRESENTATION_DIR.mkdir(parents=True, exist_ok=True)

print("="*80)
print("SIMPLE VISUAL ANALYSIS")
print("="*80)
print(f"\n📂 Loading data from {CLEANED_HEART_DATA}...")
heart = pd.read_csv(CLEANED_HEART_DATA)
print(f"✓ Loaded {len(heart):,} samples\n")

print(f"📊 Saving visuals to {PRESENTATION_DIR}\n")
print("Creating visualizations...\n")

# ============================================================================
# 1. THE BIG PICTURE - Disease Prevalence Overview
# ============================================================================
print("1. Disease prevalence overview...")

fig, ax = plt.subplots(figsize=(10, 6))

diseases = {
    'Depression': (heart['HadDepressiveDisorder'] == 1).mean() * 100,
    'Arthritis': (heart['HadArthritis'] == 1).mean() * 100,
    'Diabetes': (heart['HadDiabetes'] == 3).mean() * 100,
    'Heart Attack': (heart['HadHeartAttack'] == 1).mean() * 100,
    'Stroke': (heart['HadStroke'] == 1).mean() * 100,
    'COPD': (heart['HadCOPD'] == 1).mean() * 100
}

# Sort by prevalence
diseases_sorted = dict(sorted(diseases.items(), key=lambda x: x[1], reverse=True))

colors = ['#e74c3c', '#e67e22', '#f39c12', '#3498db', '#2ecc71', '#9b59b6']
bars = ax.barh(list(diseases_sorted.keys()), list(diseases_sorted.values()), 
               color=colors, edgecolor='black', linewidth=1.5)

# Add percentage labels
for i, (disease, pct) in enumerate(diseases_sorted.items()):
    ax.text(pct + 0.5, i, f'{pct:.1f}%', va='center', fontweight='bold', fontsize=11)

ax.set_xlabel('Prevalence (%)', fontsize=12, fontweight='bold')
ax.set_title('Chronic Disease Prevalence in U.S. Adults', 
             fontsize=14, fontweight='bold', pad=20)
ax.set_xlim(0, max(diseases_sorted.values()) * 1.15)
plt.tight_layout()
plt.savefig(PRESENTATION_DIR / '1_disease_prevalence.png', bbox_inches='tight')
plt.close()
print("   ✓ Saved: 1_disease_prevalence.png")

# ============================================================================
# 2. AGE STORY - How health changes with age
# ============================================================================
print("2. Health across age groups...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

age_labels = ["18-24", "25-29", "30-34", "35-39", "40-44", "45-49",
              "50-54", "55-59", "60-64", "65-69", "70-74", "75-79", "80+"]

# Diabetes by age
ax1 = axes[0, 0]
diabetes_by_age = heart.groupby('AgeCategory', observed=False).apply(
    lambda x: (x['HadDiabetes'] == 3).mean() * 100, include_groups=False
)
ax1.plot(range(13), diabetes_by_age.values, 'o-', linewidth=3, 
         markersize=8, color='#e74c3c')
ax1.fill_between(range(13), diabetes_by_age.values, alpha=0.3, color='#e74c3c')
ax1.set_xticks(range(13))
ax1.set_xticklabels(age_labels, rotation=45, ha='right')
ax1.set_ylabel('Diabetes Prevalence (%)', fontweight='bold')
ax1.set_title('Diabetes Risk Increases with Age', fontweight='bold', fontsize=12)
ax1.grid(alpha=0.3)

# Physical activity by age
ax2 = axes[0, 1]
activity_by_age = heart.groupby('AgeCategory')['PhysicalActivities'].mean() * 100
ax2.plot(range(13), activity_by_age.values, 'o-', linewidth=3, 
         markersize=8, color='#2ecc71')
ax2.fill_between(range(13), activity_by_age.values, alpha=0.3, color='#2ecc71')
ax2.set_xticks(range(13))
ax2.set_xticklabels(age_labels, rotation=45, ha='right')
ax2.set_ylabel('Physically Active (%)', fontweight='bold')
ax2.set_title('Physical Activity Declines with Age', fontweight='bold', fontsize=12)
ax2.grid(alpha=0.3)

# BMI by age
ax3 = axes[1, 0]
bmi_by_age = heart.groupby('AgeCategory')['BMI'].mean()
ax3.plot(range(13), bmi_by_age.values, 'o-', linewidth=3, 
         markersize=8, color='#f39c12')
ax3.fill_between(range(13), bmi_by_age.values, alpha=0.3, color='#f39c12')
ax3.axhline(y=25, color='orange', linestyle='--', label='Overweight threshold')
ax3.axhline(y=30, color='red', linestyle='--', label='Obese threshold')
ax3.set_xticks(range(13))
ax3.set_xticklabels(age_labels, rotation=45, ha='right')
ax3.set_ylabel('Average BMI', fontweight='bold')
ax3.set_title('BMI Peaks in Middle Age', fontweight='bold', fontsize=12)
ax3.legend()
ax3.grid(alpha=0.3)

# Depression by age
ax4 = axes[1, 1]
depression_by_age = heart.groupby('AgeCategory')['HadDepressiveDisorder'].mean() * 100
ax4.plot(range(13), depression_by_age.values, 'o-', linewidth=3, 
         markersize=8, color='#9b59b6')
ax4.fill_between(range(13), depression_by_age.values, alpha=0.3, color='#9b59b6')
ax4.set_xticks(range(13))
ax4.set_xticklabels(age_labels, rotation=45, ha='right')
ax4.set_ylabel('Depression Prevalence (%)', fontweight='bold')
ax4.set_title('Depression Affects All Ages', fontweight='bold', fontsize=12)
ax4.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(PRESENTATION_DIR / '2_age_patterns.png', bbox_inches='tight')
plt.close()
print("   ✓ Saved: 2_age_patterns.png")

# ============================================================================
# 3. LIFESTYLE IMPACT - Show the "why"
# ============================================================================
print("3. Lifestyle impacts...")

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Smoking vs Heart Disease
ax1 = axes[0]
smoking_labels = ['Never', 'Former', 'Some Days', 'Every Day']
heart_by_smoking = heart.groupby('SmokerStatus')['HadHeartAttack'].mean() * 100
ax1.bar(smoking_labels, heart_by_smoking.values, 
        color=['#2ecc71', '#f39c12', '#e67e22', '#e74c3c'], 
        edgecolor='black', linewidth=1.5)
ax1.set_ylabel('Heart Attack Prevalence (%)', fontweight='bold')
ax1.set_title('Smoking → Heart Disease', fontweight='bold', fontsize=12)
ax1.set_ylim(0, max(heart_by_smoking.values) * 1.3)
for i, v in enumerate(heart_by_smoking.values):
    ax1.text(i, v + 0.3, f'{v:.1f}%', ha='center', fontweight='bold')

# BMI vs Diabetes
ax2 = axes[1]
bmi_bins = [0, 18.5, 25, 30, 35, 100]
bmi_labels = ['Underweight', 'Normal', 'Overweight', 'Obese', 'Severe Obese']
heart['BMI_Category'] = pd.cut(heart['BMI'], bins=bmi_bins, labels=bmi_labels)
diabetes_by_bmi = heart.groupby('BMI_Category', observed=True).apply(
    lambda x: (x['HadDiabetes'] == 3).mean() * 100, include_groups=False
)
colors_bmi = ['#3498db', '#2ecc71', '#f39c12', '#e67e22', '#e74c3c']
ax2.bar(bmi_labels, diabetes_by_bmi.values, color=colors_bmi, 
        edgecolor='black', linewidth=1.5)
ax2.set_ylabel('Diabetes Prevalence (%)', fontweight='bold')
ax2.set_title('BMI → Diabetes', fontweight='bold', fontsize=12)
ax2.tick_params(axis='x', rotation=45)
ax2.set_ylim(0, max(diabetes_by_bmi.values) * 1.3)
for i, v in enumerate(diabetes_by_bmi.values):
    ax2.text(i, v + 0.5, f'{v:.1f}%', ha='center', fontweight='bold')

# Sleep vs Depression
ax3 = axes[2]
sleep_bins = [0, 5, 7, 9, 24]
sleep_labels = ['<5 hrs', '5-6 hrs', '7-8 hrs', '>9 hrs']
heart['Sleep_Category'] = pd.cut(heart['SleepHours'], bins=sleep_bins, labels=sleep_labels)
depression_by_sleep = heart.groupby('Sleep_Category', observed=True)['HadDepressiveDisorder'].mean() * 100
colors_sleep = ['#e74c3c', '#f39c12', '#2ecc71', '#3498db']
ax3.bar(sleep_labels, depression_by_sleep.values, color=colors_sleep, 
        edgecolor='black', linewidth=1.5)
ax3.set_ylabel('Depression Prevalence (%)', fontweight='bold')
ax3.set_title('Sleep → Mental Health', fontweight='bold', fontsize=12)
ax3.set_ylim(0, max(depression_by_sleep.values) * 1.3)
for i, v in enumerate(depression_by_sleep.values):
    ax3.text(i, v + 0.5, f'{v:.1f}%', ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig(PRESENTATION_DIR / '3_lifestyle_impact.png', bbox_inches='tight')
plt.close()
print("   ✓ Saved: 3_lifestyle_impact.png")

# ============================================================================
# 4. STATE COMPARISON - Geographic story
# ============================================================================
print("4. Geographic patterns...")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Top 15 states by diabetes
ax1 = axes[0]
diabetes_by_state = heart.groupby('State', observed=False).apply(
    lambda x: (x['HadDiabetes'] == 3).mean() * 100, include_groups=False
).sort_values(ascending=False).head(15)
ax1.barh(range(15), diabetes_by_state.values, color='#e74c3c', edgecolor='black')
ax1.set_yticks(range(15))
ax1.set_yticklabels([s.title() for s in diabetes_by_state.index])
ax1.set_xlabel('Diabetes Prevalence (%)', fontweight='bold')
ax1.set_title('Top 15 States: Highest Diabetes Rates', fontweight='bold', fontsize=12)
ax1.invert_yaxis()
for i, v in enumerate(diabetes_by_state.values):
    ax1.text(v + 0.3, i, f'{v:.1f}%', va='center', fontweight='bold')

# Bottom 15 states by diabetes
ax2 = axes[1]
diabetes_by_state_low = heart.groupby('State', observed=False).apply(
    lambda x: (x['HadDiabetes'] == 3).mean() * 100, include_groups=False
).sort_values(ascending=True).head(15)
ax2.barh(range(15), diabetes_by_state_low.values, color='#2ecc71', edgecolor='black')
ax2.set_yticks(range(15))
ax2.set_yticklabels([s.title() for s in diabetes_by_state_low.index])
ax2.set_xlabel('Diabetes Prevalence (%)', fontweight='bold')
ax2.set_title('Bottom 15 States: Lowest Diabetes Rates', fontweight='bold', fontsize=12)
ax2.invert_yaxis()
for i, v in enumerate(diabetes_by_state_low.values):
    ax2.text(v + 0.3, i, f'{v:.1f}%', va='center', fontweight='bold')

plt.tight_layout()
plt.savefig(PRESENTATION_DIR / '4_state_comparison.png', bbox_inches='tight')
plt.close()
print("   ✓ Saved: 4_state_comparison.png")

# ============================================================================
# 5. THE COMBO EFFECT - Multiple risk factors
# ============================================================================
print("5. Combined risk factors...")

fig, ax = plt.subplots(figsize=(12, 7))

# Define risk groups
conditions = [
    ('No Risk Factors', 
     (heart['BMI'] < 25) & (heart['SmokerStatus'] == 0) & (heart['PhysicalActivities'] == 1)),
    
    ('1 Risk Factor',
     ((heart['BMI'] >= 25) & (heart['SmokerStatus'] == 0) & (heart['PhysicalActivities'] == 1)) |
     ((heart['BMI'] < 25) & (heart['SmokerStatus'] > 0) & (heart['PhysicalActivities'] == 1)) |
     ((heart['BMI'] < 25) & (heart['SmokerStatus'] == 0) & (heart['PhysicalActivities'] == 0))),
    
    ('2 Risk Factors',
     ((heart['BMI'] >= 25) & (heart['SmokerStatus'] > 0) & (heart['PhysicalActivities'] == 1)) |
     ((heart['BMI'] >= 25) & (heart['SmokerStatus'] == 0) & (heart['PhysicalActivities'] == 0)) |
     ((heart['BMI'] < 25) & (heart['SmokerStatus'] > 0) & (heart['PhysicalActivities'] == 0))),
    
    ('All 3 Risk Factors',
     (heart['BMI'] >= 25) & (heart['SmokerStatus'] > 0) & (heart['PhysicalActivities'] == 0))
]

diabetes_rates = []
group_sizes = []

for name, condition in conditions:
    group = heart[condition]
    rate = (group['HadDiabetes'] == 3).mean() * 100
    diabetes_rates.append(rate)
    group_sizes.append(len(group))

x = np.arange(len(conditions))
colors = ['#2ecc71', '#f39c12', '#e67e22', '#e74c3c']

bars = ax.bar(x, diabetes_rates, color=colors, edgecolor='black', linewidth=2, width=0.6)

# Add labels
for i, (bar, rate, size) in enumerate(zip(bars, diabetes_rates, group_sizes)):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
            f'{rate:.1f}%\n({size:,} people)',
            ha='center', va='bottom', fontweight='bold', fontsize=10)

ax.set_ylabel('Diabetes Prevalence (%)', fontsize=12, fontweight='bold')
ax.set_title('Cumulative Effect of Risk Factors on Diabetes\n(Overweight + Smoking + No Exercise)',
             fontsize=13, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels([c[0] for c in conditions], fontsize=11)
ax.set_ylim(0, max(diabetes_rates) * 1.3)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(PRESENTATION_DIR / '5_combined_risks.png', bbox_inches='tight')
plt.close()
print("   ✓ Saved: 5_combined_risks.png")

# ============================================================================
# 6. PREVENTIVE CARE GAPS
# ============================================================================
print("6. Preventive care analysis...")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Checkup rates by age
ax1 = axes[0]
recent_checkup = heart.groupby('AgeCategory', observed=False).apply(
    lambda x: (x['LastCheckupTime'] == 0).mean() * 100, include_groups=False
)
ax1.plot(range(13), recent_checkup.values, 'o-', linewidth=3, 
         markersize=10, color='#3498db')
ax1.fill_between(range(13), recent_checkup.values, alpha=0.3, color='#3498db')
ax1.axhline(y=80, color='green', linestyle='--', linewidth=2, label='Target: 80%')
ax1.set_xticks(range(13))
ax1.set_xticklabels(age_labels, rotation=45, ha='right')
ax1.set_ylabel('% with Checkup in Past Year', fontweight='bold')
ax1.set_title('Preventive Care: Checkup Rates by Age', fontweight='bold', fontsize=12)
ax1.legend()
ax1.grid(alpha=0.3)
ax1.set_ylim(60, 100)

# Vaccination rates
ax2 = axes[1]
vaccines = {
    'Flu Vaccine\n(Past Year)': heart['FluVaxLast12'].mean() * 100,
    'Pneumonia Vaccine\n(Ever)': heart['PneumoVaxEver'].mean() * 100,
    'HIV Testing\n(Ever)': heart['HIVTesting'].mean() * 100
}
colors_vax = ['#3498db', '#2ecc71', '#9b59b6']
bars = ax2.bar(range(len(vaccines)), list(vaccines.values()), 
               color=colors_vax, edgecolor='black', linewidth=1.5)
ax2.set_xticks(range(len(vaccines)))
ax2.set_xticklabels(list(vaccines.keys()), fontsize=10)
ax2.set_ylabel('Vaccination/Testing Rate (%)', fontweight='bold')
ax2.set_title('Preventive Care: Vaccination Rates', fontweight='bold', fontsize=12)
ax2.set_ylim(0, 100)
ax2.axhline(y=70, color='green', linestyle='--', alpha=0.5, label='Target: 70%')
ax2.legend()

for i, v in enumerate(vaccines.values()):
    ax2.text(i, v + 2, f'{v:.1f}%', ha='center', fontweight='bold', fontsize=11)

plt.tight_layout()
plt.savefig(PRESENTATION_DIR / '6_preventive_care.png', bbox_inches='tight')
plt.close()
print("   ✓ Saved: 6_preventive_care.png")

# ============================================================================
# SUMMARY STATS for presentation
# ============================================================================

print("\n" + "="*80)
print("📊 KEY INSIGHTS FOR YOUR PRESENTATION")
print("="*80)

print("\n1. DISEASE PREVALENCE:")
print(f"   • Depression is #1: {diseases['Depression']:.1f}% (1 in {int(100/diseases['Depression'])})")
print(f"   • Diabetes affects: {diseases['Diabetes']:.1f}%")
print(f"   • Heart attacks: {diseases['Heart Attack']:.1f}%")

print("\n2. AGE PATTERNS:")
young_diabetes = heart[heart['AgeCategory'] < 5]['HadDiabetes']
old_diabetes = heart[heart['AgeCategory'] >= 9]['HadDiabetes']
print(f"   • Diabetes risk increases {(old_diabetes == 3).mean() / (young_diabetes == 3).mean():.1f}x from young to old")

print("\n3. LIFESTYLE IMPACT:")
never_smokers = heart[heart['SmokerStatus'] == 0]
daily_smokers = heart[heart['SmokerStatus'] == 3]
print(f"   • Daily smokers: {daily_smokers['HadHeartAttack'].mean() / never_smokers['HadHeartAttack'].mean():.1f}x more heart attacks")

normal_bmi = heart[heart['BMI'] < 25]
obese = heart[heart['BMI'] >= 30]
print(f"   • Obesity: {((obese['HadDiabetes'] == 3).mean() / (normal_bmi['HadDiabetes'] == 3).mean()):.1f}x more diabetes")

print("\n4. GEOGRAPHY:")
top_state = diabetes_by_state.index[0]
top_rate = diabetes_by_state.values[0]
bottom_state = diabetes_by_state_low.index[0]
bottom_rate = diabetes_by_state_low.values[0]
print(f"   • Highest: {top_state.title()} ({top_rate:.1f}%)")
print(f"   • Lowest: {bottom_state.title()} ({bottom_rate:.1f}%)")
print(f"   • Gap: {top_rate - bottom_rate:.1f} percentage points")

print("\n5. PREVENTIVE CARE GAPS:")
print(f"   • Only {vaccines['Flu Vaccine\n(Past Year)']:.1f}% got flu vaccine")
print(f"   • {100 - recent_checkup.mean():.1f}% haven't had checkup recently")

print("\n" + "="*80)
print(f"✅ ALL VISUALS SAVED TO: presentation_visuals/")
print("="*80)
print("\n🎯 You now have 6 powerful slides ready to go!")
print("\nNext steps:")
print("  1. Open presentation_visuals/ folder")
print("  2. Review each image")
print("  3. Copy key insights above into your slides")
print("  4. Build your story around these visuals")