"""Leakage-safe, freshness-bounded joins for county-level time series."""
from __future__ import annotations

import pandas as pd

# Weekly weather and USDM observations must not be carried forward through
# missing weeks. A future observation must never label an earlier NDVI row.
MAX_OBSERVATION_AGE = pd.Timedelta(days=7)


def join_latest_observation(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    left_on: str,
    right_on: str,
    left_by: str,
    right_by: str,
) -> pd.DataFrame:
    """Join the latest same-county observation at most seven days old.

    Sorting by timestamp globally is required by pandas.merge_asof, even
    when a county grouping key is provided. Reject missing join timestamps
    rather than silently accepting an incomplete feature/label timeline.
    """
    if left[left_on].isna().any() or right[right_on].isna().any():
        raise ValueError("Temporal join timestamps must not be missing")
    return pd.merge_asof(
        left.sort_values(left_on),
        right.sort_values(right_on),
        left_on=left_on,
        right_on=right_on,
        left_by=left_by,
        right_by=right_by,
        direction="backward",
        tolerance=MAX_OBSERVATION_AGE,
    )
