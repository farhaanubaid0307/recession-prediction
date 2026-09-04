"""Out-of-sample evaluation on a single chronological split.

Protocol:
- One chronological split: train on everything before SPLIT_DATE, test on
  everything from SPLIT_DATE onward. Models are fitted once.
- Real-time label availability: training stops h months before SPLIT_DATE,
  because rows after that have labels that depend on months in the test
  period and would not have been known when the model was fitted.
- Metrics: ROC-AUC (primary), PR-AUC,
  Brier score, and precision/recall/F1 at the 0.5 threshold.
  The Sahm rule emits a score, not a probability, so it gets
  threshold-free metrics only.

The 2007-09 zoom in the report (Figure 5) is a window onto these same
predictions, not a separately trained model: the split already stops at
2005-07, so the crisis months are genuine out-of-sample forecasts.

Run:  python -m src.evaluate
"""
import numpy as np
import pandas as pd
from pandas.tseries.offsets import DateOffset
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_recall_fscore_support,
    roc_auc_score,
)

from src.config import (
    FEATURES,
    HORIZONS,
    PROCESSED_DIR,
    RESULTS_DIR,
    SPLIT_DATE,
)
from src.models import make_models

SEED = 42


def load_dataset() -> pd.DataFrame:
    return pd.read_csv(PROCESSED_DIR / "dataset.csv", parse_dates=["date"], index_col="date")


def train_test_eval(df: pd.DataFrame, horizon: int,
                    split_date: str = SPLIT_DATE) -> pd.DataFrame:
    """Fit once on the pre-split years, then score every test month.

    Training stops `horizon` months before the split so that no training
    label depends on a month inside the test period.
    """
    target = f"TARGET_{horizon}M"
    data = df.dropna(subset=[target])
    train_end = pd.Timestamp(split_date) - DateOffset(months=horizon)
    train = data.loc[:train_end]
    test = data.loc[split_date:]

    X_tr, y_tr = train[FEATURES], train[target].astype(int)
    out = pd.DataFrame(index=test.index)
    out["target"] = test[target].astype(int)
    for name, model in make_models(y_tr).items():
        model.fit(X_tr, y_tr)
        out[name] = model.predict_proba(test[FEATURES])[:, 1]
    return out


def block_bootstrap_auc(y: np.ndarray, p: np.ndarray, block: int = 12,
                        n_boot: int = 1000, seed: int = SEED) -> tuple[float, float]:
    """Moving-block bootstrap CI for ROC-AUC (time series have serial
    dependence, so i.i.d. resampling would understate uncertainty).

    The generator is seeded per call so that scoring the same predictions
    twice in one process gives the same interval."""
    rng = np.random.default_rng(seed)
    n = len(y)
    if len(np.unique(y)) < 2:
        return float("nan"), float("nan")
    aucs = []
    for _ in range(n_boot):
        starts = rng.integers(0, n - block + 1, size=int(np.ceil(n / block)))
        idx = np.concatenate([np.arange(s, s + block) for s in starts])[:n]
        if len(np.unique(y[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y[idx], p[idx]))
    if not aucs:
        return float("nan"), float("nan")
    return float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


def score_predictions(preds: pd.DataFrame, horizon: int) -> pd.DataFrame:
    y = preds["target"].to_numpy()
    out = []
    for name in [c for c in preds.columns if c != "target"]:
        p = preds[name].to_numpy()
        lo, hi = block_bootstrap_auc(y, p)
        m = {
            "horizon": horizon,
            "model": name,
            "roc_auc": roc_auc_score(y, p),
            "roc_auc_lo": lo,
            "roc_auc_hi": hi,
            "pr_auc": average_precision_score(y, p),
        }
        if name != "Sahm rule":  # score-based benchmark: no prob metrics
            prec, rec, f1, _ = precision_recall_fscore_support(
                y, (p >= 0.5).astype(int), average="binary", zero_division=0
            )
            m.update(brier=brier_score_loss(y, p), precision=prec, recall=rec, f1=f1)
        out.append(m)
    return pd.DataFrame(out)


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df = load_dataset()
    all_metrics = []
    for h in HORIZONS:
        print(f"Train/test split at {SPLIT_DATE}, horizon={h}m ...")
        preds = train_test_eval(df, h)
        preds.to_csv(RESULTS_DIR / f"predictions_h{h}.csv")
        metrics = score_predictions(preds, h)
        all_metrics.append(metrics)
        print(metrics[["model", "roc_auc", "pr_auc"]].to_string(index=False))
    pd.concat(all_metrics).to_csv(RESULTS_DIR / "metrics.csv", index=False)
    print(f"Saved results to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
