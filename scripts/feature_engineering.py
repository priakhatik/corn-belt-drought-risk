"""
Feature Engineering: Rolling/Lagged NDVI + Precipitation + Temperature Anomaly
====================================================================================
12-week rolling windows (extended from 8-week) to test whether drought
signal continues to strengthen with longer accumulation periods.
"""

import pandas as pd

# ==================================================================
# PART 1: Weekly gridMET rolling features
# ==================================================================
gridmet = pd.read_csv('raw/weather/corn_belt_gridmet_weekly_2010_2023.csv')
gridmet['week_start'] = pd.to_datetime(gridmet['week_start'])
gridmet['county_fips'] = gridmet['county_fips'].astype(str).str.zfill(5)
gridmet['tmax_c'] = gridmet['tmax_k'] - 273.15
gridmet['tmin_c'] = gridmet['tmin_k'] - 273.15
gridmet['year'] = gridmet['week_start'].dt.year

# Season week index: weeks since April 1 of that year (0, 1, 2, ...)
season_start = pd.to_datetime(gridmet['year'].astype(str) + '-04-01')
gridmet['season_week'] = ((gridmet['week_start'] - season_start).dt.days // 7)

gridmet = gridmet.sort_values(['county_fips', 'week_start'])

# ---- Precipitation: rolling sum (12-week) + climatology deficit ----
gridmet['precip_roll4wk'] = (
    gridmet.groupby('county_fips')['precip_mm']
    .transform(lambda x: x.rolling(window=12, min_periods=1).sum())
)

precip_climatology = (
    gridmet.groupby(['county_fips', 'season_week'])['precip_mm']
    .transform('mean')
)
gridmet['precip_climatology'] = precip_climatology

gridmet['precip_climatology_roll4wk'] = (
    gridmet.groupby('county_fips')['precip_climatology']
    .transform(lambda x: x.rolling(window=12, min_periods=1).sum())
)

gridmet['precip_deficit_4wk'] = (
    gridmet['precip_roll4wk'] - gridmet['precip_climatology_roll4wk']
)

# ---- Temperature: rolling mean (12-week) + climatology anomaly ----
gridmet['tmax_roll4wk'] = (
    gridmet.groupby('county_fips')['tmax_c']
    .transform(lambda x: x.rolling(window=12, min_periods=1).mean())
)

tmax_climatology = (
    gridmet.groupby(['county_fips', 'season_week'])['tmax_c']
    .transform('mean')
)
gridmet['tmax_anomaly'] = gridmet['tmax_c'] - tmax_climatology

gridmet['tmax_anomaly_roll4wk'] = (
    gridmet.groupby('county_fips')['tmax_anomaly']
    .transform(lambda x: x.rolling(window=12, min_periods=1).mean())
)

# ==================================================================
# PART 2: NDVI lag/trend features
# ==================================================================
ndvi = pd.read_csv('raw/ndvi/corn_belt_ndvi_evi_2010_2023.csv')
ndvi['date'] = pd.to_datetime(ndvi['date'])
ndvi['county_fips'] = ndvi['county_fips'].astype(str).str.zfill(5)
ndvi['ndvi_mean'] = ndvi['ndvi_mean'] * 0.0001
ndvi['evi_mean'] = ndvi['evi_mean'] * 0.0001

ndvi = ndvi.sort_values(['county_fips', 'date'])

ndvi['ndvi_lag1'] = ndvi.groupby('county_fips')['ndvi_mean'].shift(1)
ndvi['ndvi_change'] = ndvi['ndvi_mean'] - ndvi['ndvi_lag1']

ndvi['ndvi_roll3'] = (
    ndvi.groupby('county_fips')['ndvi_mean']
    .transform(lambda x: x.rolling(window=4, min_periods=1).mean())
)

# ==================================================================
# PART 3: USDM label
# ==================================================================
usdm = pd.read_csv('raw/usdm/usdm_corn_belt_2010_2023.csv')
usdm['ValidStart'] = pd.to_datetime(usdm['ValidStart'])
usdm['FIPS'] = usdm['FIPS'].astype(str).str.zfill(5)

drought_cols = ['None', 'D0', 'D1', 'D2', 'D3', 'D4']
for col in drought_cols:
    usdm[col] = pd.to_numeric(usdm[col], errors='coerce')

usdm['total_area'] = usdm['None'] + usdm['D0'] + usdm['D1'] + usdm['D2'] + usdm['D3'] + usdm['D4']
usdm['pct_severe_drought'] = (usdm['D1'] + usdm['D2'] + usdm['D3'] + usdm['D4']) / usdm['total_area'] * 100
usdm['stressed'] = (usdm['pct_severe_drought'] >= 20).astype(int)

# ==================================================================
# PART 4: Merge everything on nearest date, per county
# ==================================================================
ndvi_sorted = ndvi.sort_values('date')
gridmet_sorted = gridmet.sort_values('week_start')

merged = pd.merge_asof(
    ndvi_sorted,
    gridmet_sorted[['county_fips', 'week_start', 'precip_deficit_4wk',
                     'tmax_roll4wk', 'tmax_anomaly_roll4wk', 'precip_roll4wk']],
    left_on='date',
    right_on='week_start',
    left_by='county_fips',
    right_by='county_fips',
    direction='nearest',
)

usdm_sorted = usdm.sort_values('ValidStart')
merged_sorted = merged.sort_values('date')

final = pd.merge_asof(
    merged_sorted,
    usdm_sorted[['FIPS', 'ValidStart', 'pct_severe_drought', 'stressed']],
    left_on='date',
    right_on='ValidStart',
    left_by='county_fips',
    right_by='FIPS',
    direction='nearest',
)

final = final.drop(columns=['FIPS', 'ValidStart', 'week_start'])
final = final.dropna(subset=['pct_severe_drought', 'stressed'])
final = final.dropna(subset=['ndvi_lag1', 'ndvi_change'])

final.to_csv('processed/corn_belt_features.csv', index=False)

print(f'Done. {len(final)} rows saved to processed/corn_belt_features.csv')
print(f'\nLabel distribution:')
print(final['stressed'].value_counts(normalize=True))
print(f'\nMissing values in new features:')
print(final[['ndvi_lag1', 'ndvi_change', 'ndvi_roll3', 'precip_deficit_4wk',
              'tmax_roll4wk', 'tmax_anomaly_roll4wk']].isnull().sum())
print(f'\nCorrelation with stressed (excluding pct_severe_drought, which is the label source):')
feature_cols = ['ndvi_mean', 'evi_mean', 'ndvi_change', 'ndvi_roll3',
                 'precip_deficit_4wk', 'tmax_roll4wk', 'tmax_anomaly_roll4wk',
                 'precip_roll4wk', 'stressed']
print(final[feature_cols].corr()['stressed'].sort_values(ascending=False))