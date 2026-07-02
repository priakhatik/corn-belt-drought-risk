"""
Corn Belt gridMET Extraction — Weekly Aggregation via Earth Engine
======================================================================
Pulls weekly total precipitation and mean min/max temperature for
Iowa, Illinois, Indiana counties, 2010-2023 (April-September).
Weekly cadence matches USDM data for easier merging later.
"""

import ee
import pandas as pd
import datetime

# ---- 1. Authenticate ----
ee.Authenticate()
ee.Initialize(project='corn-belt-ndvi')

# ---- 2. Define region: Corn Belt core states ----
STATE_FIPS = {
    'Iowa': '19',
    'Illinois': '17',
    'Indiana': '18',
}

counties = ee.FeatureCollection('TIGER/2018/Counties')

corn_belt_counties = counties.filter(
    ee.Filter.inList('STATEFP', list(STATE_FIPS.values()))
)

# ---- 3. Load gridMET ----
gridmet = ee.ImageCollection('IDAHO_EPSCOR/GRIDMET') \
    .select(['pr', 'tmmx', 'tmmn'])

# ---- 4. Build weekly composites: sum precip, mean temps ----
def weekly_composite(start_date):
    end_date = start_date.advance(7, 'day')
    week_imgs = gridmet.filterDate(start_date, end_date)

    precip_sum = week_imgs.select('pr').sum().rename('precip_mm')
    tmax_mean = week_imgs.select('tmmx').mean().rename('tmax_k')
    tmin_mean = week_imgs.select('tmmn').mean().rename('tmin_k')

    composite = precip_sum.addBands(tmax_mean).addBands(tmin_mean)
    return composite.set('week_start', start_date.format('YYYY-MM-dd'))

# ---- 5. Zonal stats per week ----
def extract_county_means(image):
    week_start = image.get('week_start')

    stats = image.reduceRegions(
        collection=corn_belt_counties,
        reducer=ee.Reducer.mean(),
        scale=4000,
    )

    def tag_date(feature):
        return feature.set('week_start', week_start)

    return stats.map(tag_date)

# ---- 6. Loop years, generate weekly start dates for Apr-Sep ----
all_results = []

for year in range(2010, 2024):
    start = datetime.date(year, 4, 1)
    end = datetime.date(year, 9, 30)

    week_starts = []
    current = start
    while current <= end:
        week_starts.append(current)
        current += datetime.timedelta(days=7)

    print(f'{year}: {len(week_starts)} weeks')

    for ws in week_starts:
        ee_date = ee.Date(ws.strftime('%Y-%m-%d'))
        composite = weekly_composite(ee_date)
        fc = extract_county_means(composite)

        features = fc.getInfo()['features']
        for f in features:
            props = f['properties']
            all_results.append({
                'county_fips': props.get('GEOID'),
                'county_name': props.get('NAME'),
                'state_fips': props.get('STATEFP'),
                'week_start': props.get('week_start'),
                'precip_mm': props.get('precip_mm'),
                'tmax_k': props.get('tmax_k'),
                'tmin_k': props.get('tmin_k'),
            })

    print(f'  -> {len(all_results)} rows so far')

# ---- 7. Export to CSV ----
df = pd.DataFrame(all_results)

fips_to_state = {v: k for k, v in STATE_FIPS.items()}
df['state'] = df['state_fips'].map(fips_to_state)

df.to_csv('corn_belt_gridmet_weekly_2010_2023.csv', index=False)
print(f'Done. {len(df)} rows exported to corn_belt_gridmet_weekly_2010_2023.csv')