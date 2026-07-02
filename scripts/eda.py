"""
EDA for Corn Belt Drought Stress Classification
====================================================
Generates visualizations to understand feature-label relationships
before modeling. Mirrors the FHB EDA structure.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('whitegrid')

# ---- Load final merged dataset ----
df = pd.read_csv('processed/corn_belt_final_dataset.csv')
df['date'] = pd.to_datetime(df['date'])
df['year'] = df['date'].dt.year

OUT_DIR = 'processed/eda/'

# ---- 1. Label distribution ----
plt.figure(figsize=(6, 5))
df['stressed'].value_counts().plot(kind='bar', color=['#4C72B0', '#C44E52'])
plt.title('Drought Stress Label Distribution')
plt.xlabel('Stressed (1 = severe drought, D2+)')
plt.ylabel('Count')
plt.xticks([0, 1], ['Not Stressed', 'Stressed'], rotation=0)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}01_label_distribution.png', dpi=150)
plt.close()

# ---- 2. NDVI by stress label (boxplot) ----
plt.figure(figsize=(6, 5))
sns.boxplot(data=df, x='stressed', y='ndvi_mean', palette=['#4C72B0', '#C44E52'])
plt.title('NDVI by Drought Stress Status')
plt.xlabel('Stressed')
plt.ylabel('NDVI (mean)')
plt.xticks([0, 1], ['Not Stressed', 'Stressed'])
plt.tight_layout()
plt.savefig(f'{OUT_DIR}02_ndvi_by_stress.png', dpi=150)
plt.close()

# ---- 3. EVI by stress label (boxplot) ----
plt.figure(figsize=(6, 5))
sns.boxplot(data=df, x='stressed', y='evi_mean', palette=['#4C72B0', '#C44E52'])
plt.title('EVI by Drought Stress Status')
plt.xlabel('Stressed')
plt.ylabel('EVI (mean)')
plt.xticks([0, 1], ['Not Stressed', 'Stressed'])
plt.tight_layout()
plt.savefig(f'{OUT_DIR}03_evi_by_stress.png', dpi=150)
plt.close()

# ---- 4. Precipitation by stress label (boxplot) ----
plt.figure(figsize=(6, 5))
sns.boxplot(data=df, x='stressed', y='precip_mm', palette=['#4C72B0', '#C44E52'])
plt.title('Weekly Precipitation by Drought Stress Status')
plt.xlabel('Stressed')
plt.ylabel('Precipitation (mm)')
plt.xticks([0, 1], ['Not Stressed', 'Stressed'])
plt.tight_layout()
plt.savefig(f'{OUT_DIR}04_precip_by_stress.png', dpi=150)
plt.close()

# ---- 5. Max temperature by stress label (boxplot) ----
plt.figure(figsize=(6, 5))
sns.boxplot(data=df, x='stressed', y='tmax_c', palette=['#4C72B0', '#C44E52'])
plt.title('Max Temperature by Drought Stress Status')
plt.xlabel('Stressed')
plt.ylabel('Max Temp (°C)')
plt.xticks([0, 1], ['Not Stressed', 'Stressed'])
plt.tight_layout()
plt.savefig(f'{OUT_DIR}05_tmax_by_stress.png', dpi=150)
plt.close()

# ---- 6. Correlation heatmap ----
numeric_cols = ['ndvi_mean', 'evi_mean', 'precip_mm', 'tmax_c', 'tmin_c', 'pct_severe_drought', 'stressed']
corr = df[numeric_cols].corr()

plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0, vmin=-1, vmax=1)
plt.title('Feature Correlation Heatmap')
plt.tight_layout()
plt.savefig(f'{OUT_DIR}06_correlation_heatmap.png', dpi=150)
plt.close()

# ---- 7. Stress rate over time (by year) ----
yearly_stress = df.groupby('year')['stressed'].mean() * 100

plt.figure(figsize=(10, 5))
yearly_stress.plot(kind='line', marker='o', color='#C44E52')
plt.title('Percent of County-Weeks Labeled Stressed by Year')
plt.xlabel('Year')
plt.ylabel('% Stressed')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}07_stress_over_time.png', dpi=150)
plt.close()

print('EDA complete. 7 plots saved to processed/eda/')
print()
print('Correlation with stressed label:')
print(corr['stressed'].sort_values(ascending=False))