"""Central configuration: FRED series, paths, horizons."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"

# All series are downloadable without an API key via fredgraph.csv.
# freq: native frequency (M=monthly, W=weekly, D=daily). Everything is
# resampled to monthly in features.py.
SERIES = {
    "USREC":    {"name": "NBER recession indicator (target)", "freq": "M"},
    "GS10":     {"name": "10-Year Treasury constant maturity rate", "freq": "M"},
    "TB3MS":    {"name": "3-Month Treasury bill secondary market rate", "freq": "M"},
    "UNRATE":   {"name": "Civilian unemployment rate", "freq": "M"},
    "INDPRO":   {"name": "Industrial production index", "freq": "M"},
    "UMCSENT":  {"name": "U. Michigan consumer sentiment", "freq": "M"},
    "CPIAUCSL": {"name": "CPI, all urban consumers", "freq": "M"},
    "PERMIT":   {"name": "New private housing units authorized (permits)", "freq": "M"},
    "ICSA":     {"name": "Initial jobless claims", "freq": "W"},
    "BAA10YM":  {"name": "Moody's Baa corporate yield minus 10Y Treasury", "freq": "M"},
    "NASDAQCOM": {"name": "NASDAQ composite index (equity market proxy)", "freq": "D"},
    "FEDFUNDS": {"name": "Effective federal funds rate", "freq": "M"},
}

# Analysis sample. Data is downloaded from full history so that YoY/12-month
# transforms are valid at the sample start.
SAMPLE_START = "1985-01-01"

# Forecast horizons in months: P(recession within h months).
HORIZONS = [3, 6, 12]

# Single chronological train/test split. Training covers the 1990-91 and
# 2001 recessions; the test period covers 2008 and 2020.
SPLIT_DATE = "2006-01-01"

FEATURES = [
    "SPREAD",          # GS10 - TB3MS yield spread (level)
    "UNRATE_CHG12",    # 12-month change in unemployment rate
    "SAHM_GAP",        # 3-mo avg unemployment minus its min over prior 12 months
    "INDPRO_YOY",      # industrial production, YoY % change
    "UMCSENT_LEVEL",   # consumer sentiment level
    "UMCSENT_CHG6",    # 6-month change in sentiment
    "CPI_YOY",         # CPI inflation, YoY %
    "PERMIT_YOY",      # building permits, YoY %
    "CLAIMS_YOY",      # initial claims (monthly avg), YoY %
    "CREDIT_SPREAD",   # Baa minus 10Y Treasury (level)
    "EQUITY_RET6",     # equity market 6-month return
    "FEDFUNDS_CHG12",  # 12-month change in fed funds rate
]
