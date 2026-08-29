"""Download all FRED series to data/raw/ as CSV.

Uses FRED's public fredgraph.csv endpoint, which requires no API key.
Run:  python -m src.download          (skips already-downloaded series)
      python -m src.download --force  (re-downloads everything)
"""
import io
import sys
import time
import urllib.request

import pandas as pd

from src.config import FRED_CSV_URL, RAW_DIR, SERIES


def fetch_series(series_id: str, freq: str = "M", retries: int = 3) -> pd.DataFrame:
    url = FRED_CSV_URL.format(series_id=series_id)
    # Daily/weekly series are large enough that fredgraph can 504 on the
    # full history; the sample starts in 1985, so 1980 is ample lookback.
    # Daily series are additionally aggregated server-side to monthly
    # end-of-period values (FRED's own aggregation, fq/fam params).
    if freq in ("W", "D"):
        url += "&cosd=1980-01-01"
    if freq == "D":
        url += "&fq=Monthly&fam=eop"
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=120) as resp:
                raw = resp.read().decode("utf-8")
            break
        except (TimeoutError, OSError):
            if attempt == retries - 1:
                raise
            time.sleep(5)
    df = pd.read_csv(io.StringIO(raw), na_values=".")
    df.columns = ["date", "value"]
    df["date"] = pd.to_datetime(df["date"])
    return df.dropna(subset=["value"])


def main(force: bool = False) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for series_id, meta in SERIES.items():
        out = RAW_DIR / f"{series_id}.csv"
        if out.exists() and not force:
            print(f"  {series_id:<10} cached ({meta['name']})")
            continue
        df = fetch_series(series_id, freq=meta["freq"])
        df.to_csv(out, index=False)
        print(f"  {series_id:<10} {len(df):>6} obs  {df['date'].min().date()} -> "
              f"{df['date'].max().date()}  ({meta['name']})")
    print("Done.")


if __name__ == "__main__":
    main(force="--force" in sys.argv)
