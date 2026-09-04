"""Regenerate the three numeric tables in the report.

Table 1  descriptive statistics for the headline indicators
Table 2  out-of-sample model comparison at the 6-month horizon
Table 3  random forest variable importance

The four documentation tables in results/tables/ (raw series, feature
transforms, data dictionary, targets) are written by hand and are not
regenerated here.

Run:  python -m src.tables      (after src.download and src.features)
"""
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from pandas.tseries.offsets import DateOffset

from src.config import FEATURES, PROCESSED_DIR, RESULTS_DIR, SPLIT_DATE
from src.evaluate import load_dataset, score_predictions, train_test_eval

HORIZON = 6
TABLES_DIR = RESULTS_DIR / "tables"

# Indicators shown in Table 1, with the labels used in the report.
TABLE1_ROWS = [
    ("SPREAD", "Yield spread, 10y minus 3m (pp)"),
    ("UNRATE_CHG12", "Unemployment, 12-month change (pp)"),
    ("INDPRO_YOY", "Industrial production, YoY (%)"),
    ("UMCSENT_LEVEL", "Consumer sentiment (index)"),
    ("PERMIT_YOY", "Building permits, YoY (%)"),
    ("CLAIMS_YOY", "Jobless claims, YoY (%)"),
    ("CREDIT_SPREAD", "Credit spread, Baa minus 10y (pp)"),
]

DISPLAY_NAMES = {"Sahm rule": "Sahm Rule"}


def table1_descriptive_stats(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col, label in TABLE1_ROWS:
        s = df[col]
        rows.append({
            "Indicator": label,
            "Mean": round(s.mean(), 2),
            "Std dev": round(s.std(), 2),
            "Min": round(s.min(), 2),
            "Max": round(s.max(), 2),
        })
    return pd.DataFrame(rows)


def table2_model_comparison(df: pd.DataFrame) -> pd.DataFrame:
    preds = train_test_eval(df, HORIZON)
    m = score_predictions(preds, HORIZON)
    out = pd.DataFrame({
        "Model": m["model"].map(lambda n: DISPLAY_NAMES.get(n, n)),
        "ROC-AUC": m["roc_auc"].round(3),
        "PR-AUC": m["pr_auc"].round(3),
        "Brier": m["brier"].round(3),
        "Precision": m["precision"].round(3),
        "Recall": m["recall"].round(3),
        "F1": m["f1"].round(3),
    })
    # The Sahm Rule emits a gap, not a probability, so it has no
    # threshold-based or calibration metrics.
    return out.fillna("n/a")


def table3_feature_importance(df: pd.DataFrame) -> pd.DataFrame:
    target = f"TARGET_{HORIZON}M"
    # Same window the models are fitted on: training only, ending HORIZON
    # months before the split. Importances read off the full sample would
    # be describing data the model was never allowed to see.
    train_end = pd.Timestamp(SPLIT_DATE) - DateOffset(months=HORIZON)
    train = df.dropna(subset=[target]).loc[:train_end]
    rf = RandomForestClassifier(
        n_estimators=500, max_depth=6, min_samples_leaf=5,
        class_weight="balanced", random_state=42, n_jobs=-1,
    ).fit(train[FEATURES], train[target].astype(int))
    imp = (pd.Series(rf.feature_importances_, index=FEATURES) * 100).round(1)
    imp = imp.sort_values(ascending=False)
    return pd.DataFrame({"Variable": imp.index, "Importance (%)": imp.to_numpy()})


def main() -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    df = load_dataset()
    for name, frame in [
        ("table1_descriptive_stats", table1_descriptive_stats(df)),
        ("table2_model_comparison", table2_model_comparison(df)),
        ("table3_feature_importance", table3_feature_importance(df)),
    ]:
        out = TABLES_DIR / f"{name}.csv"
        frame.to_csv(out, index=False)
        print(f"  wrote {out.relative_to(PROCESSED_DIR.parent.parent)}")


if __name__ == "__main__":
    main()
