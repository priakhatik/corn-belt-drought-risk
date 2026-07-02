"""
Merge NDVI, USDM, and gridMET data into final modeling dataset.
====================================================================
Aligns biweekly NDVI composites to the nearest weekly USDM/gridMET
record per county, builds the drought stress label.
"""

import pandas as pd

# ---- 1. Load NDVI data ----
ndvi = pd.read_csv('raw/ndvi/corn_belt_ndvi_evi_2010_2023.csv')
ndvi['date'] = pd.to_datetime(ndvi['date'])
ndvi['county_fips'] = ndvi['county_fips'].astype(str).str.zfill(5)

# Apply MODIS scale factor (raw values are x10000)
ndvi['ndvi_mean'] = ndvi['ndvi_mean'] * 0.0001
ndvi['evi_mean'] = ndvi['evi_mean'] * 0.0001

# ---- 2. Load USDM data, build the stress label ----
usdm = pd.read_csv('raw/usdm/usdm_corn_belt_2010_2023.csv')
usdm['ValidStart'] = pd.to_datetime(usdm['ValidStart'])
usdm['FIPS'] = usdm['FIPS'].astype(str).str.zfill(5)

drought_cols = ['None', 'D0', 'D1', 'D2', 'D3', 'D4']
for col in drought_cols:
    usdm[col] = pd.to_numeric(usdm[col], errors='coerce')

usdm['total_area'] = usdm['None'] + usdm['D0'] + usdm['D1'] + usdm['D2'] + usdm['D3'] + usdm['D4']
usdm['pct_severe_drought'] = (usdm['D2'] + usdm['D3'] + usdm['D4']) / usdm['total_area'] * 100
usdm['stressed'] = (usdm['pct_severe_drought'] >= 20).astype(int)

# ---- 3. Load gridMET data ----
gridmet = pd.read_csv('raw/weather/corn_belt_gridmet_weekly_2010_2023.csv')
gridmet['week_start'] = pd.to_datetime(gridmet['week_start'])
gridmet['county_fips'] = gridmet['county_fips'].astype(str).str.zfill(5)

# Convert Kelvin to Celsius
gridmet['tmax_c'] = gridmet['tmax_k'] - 273.15
gridmet['tmin_c'] = gridmet['tmin_k'] - 273.15

# ---- 4. Merge NDVI -> nearest USDM week (per county) ----
# merge_asof requires sorting by the 'on' column globally, not by-group
ndvi_sorted = ndvi.sort_values('date')
usdm_sorted = usdm.sort_values('ValidStart')

merged = pd.merge_asof(
    ndvi_sorted,
    usdm_sorted[['FIPS', 'ValidStart', 'pct_severe_drought', 'stressed']],
    left_on='date',
    right_on='ValidStart',
    left_by='county_fips',
    right_by='FIPS',
    direction='nearest',
)

# ---- 5. Merge in nearest gridMET week (per county) ----
merged_sorted = merged.sort_values('date')
gridmet_sorted = gridmet.sort_values('week_start')

final = pd.merge_asof(
    merged_sorted,
    gridmet_sorted[['county_fips', 'week_start', 'precip_mm', 'tmax_c', 'tmin_c']],
    left_on='date',
    right_on='week_start',
    left_by='county_fips',
    right_by='county_fips',
    direction='nearest',
)

# ---- 6. Clean up and save ----
final = final.drop(columns=['FIPS', 'ValidStart', 'week_start'])

final.to_csv('processed/corn_belt_final_dataset.csv', index=False)
print(f'Done. {len(final)} rows saved to processed/corn_belt_final_dataset.csv')
print(f'\nLabel distribution:')
print(final['stressed'].value_counts(normalize=True))
print(f'\nMissing values per column:')
print(final.isnull().sum())