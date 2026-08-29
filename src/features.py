"""Build the model-ready monthly dataset from raw FRED series.

Design decisions that prevent the classic mistakes in this literature:
- All features are LAGGED ONE MONTH: month t's prediction uses data published
  by t (you don't know June's unemployment rate until early July).
- Targets are forward-looking: TARGET_hM = 1 if any of months t+1 .. t+h is
  an NBER recession month.
- YoY transforms double as stationarity handling.

Run:  python -m src.features
"""
import pandas as pd

from src.config import FEATURES, HORIZONS, PROCESSED_DIR, RAW_DIR, SAMPLE_START, SERIES


def load_raw(series_id: str) -> pd.Series:
    df = pd.read_csv(RAW_DIR / f"{series_id}.csv", parse_dates=["date"])
    return df.set_index("date")["value"].sort_index()


def to_monthly(s: pd.Series, freq: str) -> pd.Series:
    """Resample to month-start frequency. Daily/weekly series are averaged
    within the month except equity prices, which use the last observation.

    Interior gaps of up to 2 months are linearly interpolated: UNRATE and
    CPI have a genuine hole at 2025-10 (the government shutdown canceled
    that month's data collection), and a single missing month would
    otherwise poison rolling-window features for over a year. Ends are
    never extrapolated."""
    if freq == "M":
        out = s.asfreq("MS") if s.index.freq else s.resample("MS").first()
    else:
        out = s.resample("MS").mean()
    return out.interpolate(limit=2, limit_area="inside")


def sahm_gap(unrate: pd.Series) -> pd.Series:
    """3-month moving average of unemployment minus its minimum over the
    preceding 12 months (Sahm 2019). Gap >= 0.5pp signals recession onset."""
    ma3 = unrate.rolling(3).mean()
    prior_min = ma3.shift(1).rolling(12).min()
    return ma3 - prior_min


def forward_recession(usrec: pd.Series, h: int) -> pd.Series:
    """1 if any of months t+1 .. t+h is a recession month, NaN once the
    window extends past the end of the data."""
    fwd = pd.concat([usrec.shift(-k) for k in range(1, h + 1)], axis=1)
    out = fwd.max(axis=1)
    out[fwd.isna().any(axis=1)] = pd.NA
    return out


def build_dataset() -> pd.DataFrame:
    raw = {sid: to_monthly(load_raw(sid), meta["freq"]) for sid, meta in SERIES.items()}
    # Equity index: month-end level, not monthly average
    nasdaq = load_raw("NASDAQCOM").resample("MS").last()

    f = pd.DataFrame(index=raw["USREC"].index)
    f["SPREAD"] = raw["GS10"] - raw["TB3MS"]
    f["UNRATE_CHG12"] = raw["UNRATE"] - raw["UNRATE"].shift(12)
    f["SAHM_GAP"] = sahm_gap(raw["UNRATE"])
    f["INDPRO_YOY"] = raw["INDPRO"].pct_change(12, fill_method=None) * 100
    f["UMCSENT_LEVEL"] = raw["UMCSENT"]
    f["UMCSENT_CHG6"] = raw["UMCSENT"] - raw["UMCSENT"].shift(6)
    f["CPI_YOY"] = raw["CPIAUCSL"].pct_change(12, fill_method=None) * 100
    f["PERMIT_YOY"] = raw["PERMIT"].pct_change(12, fill_method=None) * 100
    f["CLAIMS_YOY"] = raw["ICSA"].pct_change(12, fill_method=None) * 100
    f["CREDIT_SPREAD"] = raw["BAA10YM"]
    f["EQUITY_RET6"] = nasdaq.pct_change(6, fill_method=None) * 100
    f["FEDFUNDS_CHG12"] = raw["FEDFUNDS"] - raw["FEDFUNDS"].shift(12)

    # Publication lag: at month t we only act on data released by t.
    f = f[FEATURES].shift(1)

    df = f.copy()
    df["USREC"] = raw["USREC"]
    for h in HORIZONS:
        df[f"TARGET_{h}M"] = forward_recession(raw["USREC"], h)

    df = df.loc[SAMPLE_START:]
    # Keep rows with complete features; targets may be NaN at the sample end
    # (those rows are still needed for the dashboard's live nowcast).
    df = df.dropna(subset=FEATURES)
    df.index.name = "date"
    return df


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df = build_dataset()
    out = PROCESSED_DIR / "dataset.csv"
    df.to_csv(out)
    n_rec = int(df["USREC"].sum())
    print(f"Dataset: {len(df)} months, {df.index.min().date()} -> {df.index.max().date()}")
    print(f"Recession months: {n_rec} ({n_rec / len(df):.1%}) — class imbalance is real.")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
