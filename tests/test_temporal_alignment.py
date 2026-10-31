"""Regression checks against future-observation and stale-week leakage."""

import pandas as pd
import pytest

from scripts.temporal_alignment import join_latest_observation


def _frame(dates, counties, *, left):
    data = {
        ("date" if left else "week_start"): pd.to_datetime(dates),
        ("county_fips" if left else "FIPS"): counties,
    }
    if not left:
        data["signal"] = list(range(1, len(dates) + 1))
    return pd.DataFrame(data)


def _join(left, right):
    return join_latest_observation(
        left, right, left_on="date", right_on="week_start",
        left_by="county_fips", right_by="FIPS",
    )


def test_future_observations_are_never_used():
    left = _frame(["2021-07-01"], ["001"], left=True)
    right = _frame(["2021-07-02"], ["001"], left=False)
    result = _join(left, right)
    assert pd.isna(result.iloc[0]["signal"])


def test_stale_observations_are_rejected_but_seven_day_boundary_is_included():
    left = _frame(["2021-07-08", "2021-07-09"], ["001", "001"], left=True)
    right = _frame(["2021-07-01"], ["001"], left=False)
    result = _join(left, right).set_index("date")
    assert result.loc[pd.Timestamp("2021-07-08"), "signal"] == 1
    assert pd.isna(result.loc[pd.Timestamp("2021-07-09"), "signal"])


def test_counties_are_isolated_even_when_dates_overlap():
    left = _frame(["2021-07-05", "2021-07-05"], ["001", "002"], left=True)
    right = _frame(["2021-07-01", "2021-07-04"], ["002", "001"], left=False)
    result = _join(left, right).set_index("county_fips")
    assert result.loc["001", "signal"] == 2
    assert result.loc["002", "signal"] == 1


@pytest.mark.parametrize("missing_left", [True, False])
def test_missing_timestamps_fail_closed(missing_left):
    left = _frame([None if missing_left else "2021-07-05"], ["001"], left=True)
    right = _frame(["2021-07-01" if missing_left else None], ["001"], left=False)
    with pytest.raises(ValueError, match="timestamps must not be missing"):
        _join(left, right)
