"""
Corn Belt Drought Stress Classification — Random Forest
=============================================================
Year-based train/test split to avoid spatial leakage between
counties within the same year/week.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# ---- 1. Load data ----
df = pd.read_csv('processed/corn_belt_features.csv')
df['date'] = pd.to_datetime(df['date'])
df['year'] = df['date'].dt.year

# ---- 2. Define features (excluding leakage and identifier columns) ----
feature_cols = [
    'ndvi_mean', 'evi_mean', 'ndvi_lag1', 'ndvi_change', 'ndvi_roll3',
    'precip_deficit_4wk', 'tmax_roll4wk', 'tmax_anomaly_roll4wk', 'precip_roll4wk',
]
# Explicitly NOT included: pct_severe_drought (this IS the label source — leakage)
# Also excluded: county_fips, county_name, state, date, pixel_count (identifiers, not features)

X = df[feature_cols]
y = df['stressed']

# ---- 3. Year-based train/test split ----
train_mask = df['year'] <= 2020
test_mask = df['year'] >= 2021

X_train, X_test = X[train_mask], X[test_mask]
y_train, y_test = y[train_mask], y[test_mask]

print(f'Train: {len(X_train)} rows ({train_mask.sum() / len(df) * 100:.1f}%), years 2010-2020')
print(f'Test:  {len(X_test)} rows ({test_mask.sum() / len(df) * 100:.1f}%), years 2021-2023')
print(f'\nTrain label balance:\n{y_train.value_counts(normalize=True)}')
print(f'\nTest label balance:\n{y_test.value_counts(normalize=True)}')

# ---- 4. Train Random Forest with class balancing ----
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1,
)
model.fit(X_train, y_train)

# ---- 5. Evaluate ----
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

print('\n=== Classification Report ===')
print(classification_report(y_test, y_pred, target_names=['Not Stressed', 'Stressed']))

auc = roc_auc_score(y_test, y_proba)
print(f'ROC-AUC: {auc:.3f}')

# ---- 6. Confusion matrix plot ----
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Not Stressed', 'Stressed'],
            yticklabels=['Not Stressed', 'Stressed'])
plt.title('Confusion Matrix — Test Set (2021-2023)')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.savefig('processed/models/confusion_matrix.png', dpi=150)
plt.close()

# ---- 7. Feature importance plot ----
importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values()
plt.figure(figsize=(8, 5))
importances.plot(kind='barh', color='#4C72B0')
plt.title('Feature Importance — Random Forest')
plt.xlabel('Importance')
plt.tight_layout()
plt.savefig('processed/models/feature_importance.png', dpi=150)
plt.close()

# ---- 8. ROC curve plot ----
fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, color='#C44E52', label=f'ROC-AUC = {auc:.3f}')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve — Test Set (2021-2023)')
plt.legend()
plt.tight_layout()
plt.savefig('processed/models/roc_curve.png', dpi=150)
plt.close()

# ---- 9. Save model ----
joblib.dump(model, 'processed/models/drought_risk_rf_model.pkl')

print('\nDone. Plots saved to processed/models/, model saved as drought_risk_rf_model.pkl')
print('\nFeature importance ranking:')
print(importances.sort_values(ascending=False))