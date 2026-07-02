"""
Corn Belt NDVI/EVI Extraction — MODIS MOD13Q1
================================================
Iowa, Illinois, Indiana. Growing season composites (Apr-Sep), 2010-2023.
"""

import ee
import pandas as pd

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

# ---- 3. Load MODIS MOD13Q1 (16-day NDVI/EVI, 250m) ----
modis = ee.ImageCollection('MODIS/061/MOD13Q1') \
    .select(['NDVI', 'EVI']) \
    .filterDate('2010-01-01', '2023-12-31')

def growing_season_filter(collection, year):
    """Filter to April-September for a given year."""
    start = f'{year}-04-01'
    end = f'{year}-09-30'
    return collection.filterDate(start, end)

# ---- 4. Zonal stats: mean NDVI/EVI per county per composite date ----
def extract_county_means(image):
    date = image.date().format('YYYY-MM-dd')

    stats = image.reduceRegions(
        collection=corn_belt_counties,
        reducer=ee.Reducer.mean().combine(ee.Reducer.count(), sharedInputs=True),
        scale=250,
    )

    def tag_date(feature):
        return feature.set('date', date)

    return stats.map(tag_date)

# ---- 5. Loop years: full 2010-2023 range ----
all_results = []

for year in range(2010, 2024):
    season = growing_season_filter(modis, year)
    season_list = season.toList(season.size())
    n_images = season.size().getInfo()
    print(f'{year}: {n_images} composites')

    for i in range(n_images):
        img = ee.Image(season_list.get(i))
        fc = extract_county_means(img)

        features = fc.getInfo()['features']
        for f in features:
            props = f['properties']
            all_results.append({
                'county_fips': props.get('GEOID'),
                'county_name': props.get('NAME'),
                'state_fips': props.get('STATEFP'),
                'date': props.get('date'),
                'ndvi_mean': props.get('NDVI_mean'),
                'evi_mean': props.get('EVI_mean'),
                'pixel_count': props.get('NDVI_count'),
            })

    print(f'  -> {len(all_results)} rows so far')

# ---- 6. Export to CSV ----
df = pd.DataFrame(all_results)

fips_to_state = {v: k for k, v in STATE_FIPS.items()}
df['state'] = df['state_fips'].map(fips_to_state)

df.to_csv('corn_belt_ndvi_evi_2010_2023.csv', index=False)
print(f'Done. {len(df)} rows exported to corn_belt_ndvi_evi_2010_2023.csv')