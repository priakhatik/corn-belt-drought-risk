"""
USDM (U.S. Drought Monitor) County-Level Data Fetch
======================================================
Pulls weekly D0-D4 drought category statistics for every county
in Iowa, Illinois, and Indiana, 2010-2023, via the USDM REST API.
No authentication needed.
"""

import requests
import pandas as pd
import time
import io

STATES = ['IA', 'IL', 'IN']
YEARS = range(2010, 2024)

BASE_URL = 'https://usdmdataservices.unl.edu/api/CountyStatistics/GetDroughtSeverityStatisticsByArea'

all_dfs = []

for state in STATES:
    for year in YEARS:
        params = {
            'aoi': state,
            'startdate': f'1/1/{year}',
            'enddate': f'12/31/{year}',
            'statisticsType': 1,
        }
        resp = requests.get(BASE_URL, params=params)
        resp.raise_for_status()

        # Response comes back as CSV text
        df = pd.read_csv(io.StringIO(resp.text))
        df['query_state'] = state
        df['query_year'] = year
        all_dfs.append(df)

        print(f'{state} {year}: {len(df)} rows')
        time.sleep(0.5)  # be polite to the API

usdm_df = pd.concat(all_dfs, ignore_index=True)
usdm_df.to_csv('usdm_corn_belt_2010_2023.csv', index=False)
print(f'Done. {len(usdm_df)} total rows exported to usdm_corn_belt_2010_2023.csv')