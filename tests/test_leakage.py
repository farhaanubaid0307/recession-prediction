"""Guards against the two mistakes that would invalidate every result:
future data reaching the features, and test labels reaching the fit.

Run:  python -m pytest tests/ -q
"""
import pandas as pd
import pytest
from pandas.tseries.offsets import DateOffset

from src.config import FEATURES, HORIZONS, SPLIT_DATE
from src.evaluate import load_dataset
from src.features import forward_recession


@pytest.fixture(scope="module")
def df():
    return load_dataset()


@pytest.mark.parametrize("h", HORIZONS)
def test_training_stops_h_months_before_split(df, h):
    """A row's label depends on the h months after it, so training has to
    end h months early or the fit sees test-period outcomes."""
    target = f"TARGET_{h}M"
    data = df.dropna(subset=[target])
    train_end = pd.Timestamp(SPLIT_DATE) - DateOffset(months=h)
    train = data.loc[:train_end]
    assert train.index.max() <= train_end
    # The last training label looks at most up to the split, never past it.
    assert train.index.max() + DateOffset(months=h) <= pd.Timestamp(SPLIT_DATE)


@pytest.mark.parametrize("h", HORIZONS)
def test_train_and_test_do_not_overlap(df, h):
    target = f"TARGET_{h}M"
    data = df.dropna(subset=[target])
    train = data.loc[:pd.Timestamp(SPLIT_DATE) - DateOffset(months=h)]
    test = data.loc[SPLIT_DATE:]
    assert train.index.intersection(test.index).empty


def test_features_are_lagged_one_month(df):
    """SPREAD at month t must equal the raw 10y minus 3m spread at t-1."""
    gs10 = pd.read_csv("data/raw/GS10.csv", parse_dates=["date"]).set_index("date")["value"]
    tb3 = pd.read_csv("data/raw/TB3MS.csv", parse_dates=["date"]).set_index("date")["value"]
    raw = (gs10 - tb3).dropna()
    joined = df["SPREAD"].to_frame("lagged").join(raw.shift(1).rename("expected"))
    joined = joined.dropna()
    assert len(joined) > 400
    pd.testing.assert_series_equal(
        joined["lagged"], joined["expected"], check_names=False, atol=1e-9
    )


def test_target_never_peeks_past_end_of_data(df):
    usrec = df["USREC"].dropna()
    for h in HORIZONS:
        fwd = forward_recession(usrec, h)
        assert fwd.tail(h).isna().all(), f"TARGET_{h}M defined past the data end"


def test_no_feature_is_all_nan(df):
    assert df[FEATURES].notna().all().all()
