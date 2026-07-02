# Corn Belt Drought Risk Classification

A satellite- and climate-informed machine learning system for classifying county-level agricultural drought stress across the Corn Belt (Iowa, Illinois, Indiana), 2010–2023.

Built as a portfolio project targeting data engineering / ML roles in agricultural technology.

## Problem

Drought is one of the most consequential and costly risks in U.S. row-crop agriculture. This project asks: **can we predict whether a county is experiencing agriculturally significant drought stress using satellite vegetation data and weather anomalies, in a way that could plausibly feed an early-warning system?**

The target label is a binary classification: whether a county-week falls under moderate-to-exceptional drought (USDM categories D1–D4) covering at least 20% of the county's area.

## Data Sources

| Source | What | Coverage |
|---|---|---|
| **MODIS MOD13Q1** (via Google Earth Engine) | 16-day NDVI/EVI vegetation composites, 250m resolution | 293 counties, 2010–2023 growing seasons (Apr–Sep) |
| **U.S. Drought Monitor** (USDM REST API) | Weekly county-level drought severity (D0–D4 area) | Same counties, full date range |
| **gridMET** (via Google Earth Engine) | Daily precipitation, max/min temperature, aggregated to weekly | Same counties, full date range |

## Methodology

### Label construction
A county-week is labeled `stressed = 1` if ≥20% of its area is under USDM category D1 (moderate drought) or worse. This threshold was chosen for agricultural relevance — D0 ("abnormally dry") is often tolerated by crops, while D1+ marks the point where yield impact becomes plausible.

### Feature engineering: the key finding
Early modeling using single-timepoint (instantaneous) NDVI, precipitation, and temperature produced a weak model (**ROC-AUC 0.65**). Correlation analysis showed almost no relationship between raw NDVI and drought status — vegetation stress lags behind meteorological drought rather than tracking it in real time.

The fix was reframing every feature as an **accumulated anomaly relative to each county's own seasonal climatology**, rather than a raw snapshot value — conceptually similar to how the companion FHB disease project used a biology-informed 15-day infection window instead of single-day weather readings.

Rolling window size was tested empirically rather than assumed:

| Window | ROC-AUC |
|---|---|
| 4 weeks | 0.673 |
| 8 weeks | 0.753 |
| **12 weeks** | **0.837** |
| 16 weeks | 0.827 (plateau/slight decline) |

12 weeks (~3 months) was selected as the final window — consistent with drought being a slow-accumulating, seasonal-scale phenomenon rather than a short-term event.

### Final features
- `precip_deficit_4wk` — 12-week cumulative precipitation vs. that county's historical seasonal average
- `precip_roll4wk` — raw 12-week cumulative precipitation
- `tmax_anomaly_roll4wk` — 12-week mean max temperature anomaly vs. seasonal normal
- `tmax_roll4wk` — raw 12-week mean max temperature
- `ndvi_mean`, `evi_mean`, `ndvi_lag1`, `ndvi_change`, `ndvi_roll3` — vegetation index and short-term trend

(Column names retain a `_4wk` suffix from an earlier iteration of the pipeline; the underlying window is 12 weeks.)

### Model
Random Forest classifier (`class_weight='balanced'` to address the ~86/14 label imbalance), trained on 2010–2020 and tested on a held-out 2021–2023 window. A **year-based split** was used deliberately — a random row split would leak information between spatially correlated counties within the same drought event.

## Results

| Metric | Value |
|---|---|
| ROC-AUC | **0.837** |
| Precision (stressed class) | 0.73 |
| Recall (stressed class) | 0.40 |
| Overall accuracy | 0.81 |

**Feature importance** is dominated by precipitation deficit and temperature anomaly (combined ~83% of importance); NDVI-derived features contribute only marginally (~17% combined). This is an honest, notable result: despite satellite vegetation data being the differentiating, novel input for this project, weather-derived accumulation features carried nearly all of the predictive signal once properly engineered. NDVI's lag relative to meteorological drought (visible in EDA — stressed counties showed a long tail of very low NDVI outliers rather than a shifted median) limits its value as a same-week predictor, though it likely retains value as a *confirming* indicator after drought has already taken hold.

## Validation against a known event
The engineered `stressed` label independently recovers the 2012 Midwest drought — one of the most severe on record — as a clear spike (~40% of county-weeks labeled stressed that year, vs. low single digits in most other years), providing real-world validation that the label construction is sound.

## Repository Structure

- `raw/ndvi/` — MODIS NDVI/EVI extraction (via Earth Engine)
- `raw/usdm/` — U.S. Drought Monitor county data (via REST API)
- `raw/weather/` — gridMET weekly precipitation/temperature (via Earth Engine)
- `processed/eda/` — Exploratory data analysis plots
- `processed/models/` — Confusion matrix, feature importance, ROC curve, trained model
- `scripts/gee_ndvi_extraction.py` — MODIS NDVI/EVI extraction
- `scripts/fetch_usdm.py` — USDM drought data fetch
- `scripts/gee_gridmet_extraction.py` — gridMET weather extraction
- `scripts/feature_engineering.py` — Rolling/anomaly feature construction + label
- `scripts/eda.py` — Exploratory analysis and plots
- `scripts/model.py` — Random Forest training and evaluation

## How to Run
1. Set up a Google Earth Engine project and register for noncommercial access
2. Run `scripts/gee_ndvi_extraction.py` and `scripts/gee_gridmet_extraction.py` (requires Earth Engine auth)
3. Run `scripts/fetch_usdm.py` (no auth required — public REST API)
4. Run `scripts/feature_engineering.py` to build the final modeling dataset
5. Run `scripts/eda.py` for exploratory plots
6. Run `scripts/model.py` to train and evaluate the classifier

## Tech Stack
Python, pandas, scikit-learn, Google Earth Engine API, matplotlib/seaborn, USDM REST API

## Limitations & Future Work
- Recall on the stressed class (40%) leaves room for improvement — likely candidates include gradient boosting models (XGBoost/LightGBM), SPI/SPEI drought indices as direct inputs, and soil moisture data
- NDVI's weak same-week predictive value suggests it may be better suited as a lagged/confirming feature (e.g., predicting *next* week's drought risk) rather than a concurrent one
- The 20% area / D1+ severity threshold for the label was chosen for agricultural relevance but is one reasonable choice among several; sensitivity to this threshold was tested (D1+ vs. D2+) with minimal difference in outcome
