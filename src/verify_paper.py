"""Recompute every number quoted in the FSJ paper and check it.

Run:  python -m src.verify_paper      (after src.features and src.evaluate)

Each line prints PASS or FAIL, the paper's value, the value recomputed
from the data, and the file it was computed from. Nothing is written.
"""
import numpy as np
import pandas as pd
from pandas.tseries.offsets import DateOffset
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

from src.config import FEATURES, RESULTS_DIR, SPLIT_DATE
from src.evaluate import load_dataset, train_test_eval
from src.features import load_raw, to_monthly
from src.models import make_models

H = 6
TARGET = f"TARGET_{H}M"
LR, RF, XGB, PROBIT = "Logistic Regression", "Random Forest", "XGBoost", "Probit (yield spread)"

failures = 0


def check(label: str, paper, actual, tol: float = 0.0, src: str = "") -> None:
    global failures
    if isinstance(paper, (int, float)) and not isinstance(paper, bool):
        ok = abs(float(actual) - paper) <= tol + 1e-9
        shown = f"{actual:.3f}" if isinstance(actual, float) else str(actual)
    else:
        ok = str(actual) == str(paper)
        shown = str(actual)
    failures += not ok
    print(f"  {'PASS' if ok else 'FAIL'}  {label:<52} paper={paper!s:<18} data={shown:<18} [{src}]")


def ym(ts) -> str:
    return pd.Timestamp(ts).strftime("%b %Y")


def runs(index: pd.DatetimeIndex) -> list[str]:
    """Group months into consecutive runs, e.g. 'Apr 2020-Aug 2020 (5)'."""
    out, start, prev = [], None, None
    for d in index:
        if prev is not None and d == prev + DateOffset(months=1):
            prev = d
            continue
        if start is not None:
            out.append(f"{ym(start)}-{ym(prev)} ({len(pd.date_range(start, prev, freq='MS'))})")
        start = prev = d
    if start is not None:
        out.append(f"{ym(start)}-{ym(prev)} ({len(pd.date_range(start, prev, freq='MS'))})")
    return out


def main() -> None:
    df = load_dataset()
    preds = pd.read_csv(RESULTS_DIR / f"predictions_h{H}.csv", parse_dates=["date"], index_col="date")
    metrics = pd.read_csv(RESULTS_DIR / "metrics.csv").query("horizon == @H").set_index("model")

    print("\nSample (data/processed/dataset.csv)")
    check("Monthly observations", 499, len(df), src="dataset.csv")
    check("First month", "Jan 1985", ym(df.index.min()), src="dataset.csv")
    check("Last month", "Jul 2026", ym(df.index.max()), src="dataset.csv")
    check("Usable rows with a 6-month label", 493, int(df[TARGET].notna().sum()), src="dataset.csv")
    check("Positive (pre-recession) rows", 56, int(df[TARGET].sum()), src="dataset.csv")

    train_end = pd.Timestamp(SPLIT_DATE) - DateOffset(months=H)
    lab = df.dropna(subset=[TARGET])
    train, test = lab.loc[:train_end], lab.loc[SPLIT_DATE:]
    check("Training ends", "Jul 2005", ym(train.index.max()), src="evaluate.py")
    check("Training months", 247, len(train), src="evaluate.py")
    check("Training positives", 26, int(train[TARGET].sum()), src="evaluate.py")
    check("Test window", "Jan 2006-Jan 2026", f"{ym(test.index.min())}-{ym(test.index.max())}", src="evaluate.py")
    check("Test months", 241, len(test), src="evaluate.py")
    check("Test positives", 30, int(test[TARGET].sum()), src="evaluate.py")

    def episodes(rec: pd.Series) -> list[str]:
        starts = rec.index[(rec == 1) & (rec.shift(1, fill_value=0) == 0)]
        return [str(d.year) for d in starts]
    check("NBER recessions starting in training years", "1990, 2001",
          ", ".join(episodes(df.loc[:train_end, "USREC"])), src="dataset.csv")
    # USREC flags the month AFTER the NBER peak, so the recession the NBER
    # says began in December 2007 shows up here as starting January 2008.
    check("NBER recessions starting in test years", "2008, 2020",
          ", ".join(episodes(df.loc[SPLIT_DATE:, "USREC"])), src="dataset.csv")

    print("\nRaw-data facts (data/raw, before the one-month lag)")
    spread = (to_monthly(load_raw("GS10"), "M") - to_monthly(load_raw("TB3MS"), "M")).loc["1985":]
    check("Deepest yield-curve inversion (pp)", -1.57, round(spread.min(), 2), src="GS10-TB3MS")
    check("...in month", "May 2023", ym(spread.idxmin()), src="GS10-TB3MS")
    claims = (to_monthly(load_raw("ICSA"), "W").pct_change(12, fill_method=None) * 100).loc["1985":]
    check("Peak jobless-claims YoY growth (%)", 2123, round(claims.max()), src="ICSA")
    check("...in month", "Apr 2020", ym(claims.idxmax()), src="ICSA")
    check("First negative spread month in 2022", "Nov 2022",
          ym(spread.loc["2022"][spread.loc["2022"] < 0].index.min()), src="GS10-TB3MS")
    full = to_monthly(load_raw("GS10"), "M") - to_monthly(load_raw("TB3MS"), "M")
    deeper = full.loc[:"2021-12-01"][full.loc[:"2021-12-01"] < -1.57]
    check("'Deepest since 1981': last deeper month before 2022", "1981",
          str(deeper.index.max().year) if len(deeper) else "none", src="GS10-TB3MS")

    print("\nTable 1: out-of-sample performance (results/metrics.csv)")
    paper_t1 = {
        PROBIT: (0.457, 0.114, 0.155, 0.000, 0.000, 0.000),
        "Sahm rule": (0.774, 0.249, None, None, None, None),
        LR: (0.885, 0.446, 0.096, 0.552, 0.533, 0.542),
        RF: (0.825, 0.523, 0.084, 0.636, 0.233, 0.341),
        XGB: (0.780, 0.434, 0.100, 0.700, 0.233, 0.350),
    }
    cols = ["roc_auc", "pr_auc", "brier", "precision", "recall", "f1"]
    for model, vals in paper_t1.items():
        for col, v in zip(cols, vals):
            if v is not None:
                check(f"{model}: {col}", v, round(metrics.loc[model, col], 3), src="metrics.csv")
    check("Logistic ROC-AUC bootstrap low", 0.746, round(metrics.loc[LR, "roc_auc_lo"], 3), src="metrics.csv")
    check("Logistic ROC-AUC bootstrap high", 0.969, round(metrics.loc[LR, "roc_auc_hi"], 3), src="metrics.csv")
    check("Probit ROC-AUC bootstrap low", 0.203, round(metrics.loc[PROBIT, "roc_auc_lo"], 3), src="metrics.csv")
    check("Probit ROC-AUC bootstrap high", 0.833, round(metrics.loc[PROBIT, "roc_auc_hi"], 3), src="metrics.csv")

    print("\nAlarms at the 0.5 cutoff, Figure 1 (results/predictions_h6.csv)")
    lr_alarm = preds[LR] >= 0.5
    tp = int((lr_alarm & (preds["target"] == 1)).sum())
    fp_idx = preds.index[lr_alarm & (preds["target"] == 0)]
    check("Logistic regression alarm months", 29, int(lr_alarm.sum()), src="predictions_h6")
    check("...right (target == 1)", 16, tp, src="predictions_h6")
    tp_idx = preds.index[lr_alarm & (preds["target"] == 1)]
    # 'though 15 of those came after the 2007-09 recession had begun'
    check("...of which already inside a recession (USREC=1)", 15, int(df.loc[tp_idx, "USREC"].sum()), src="dataset.csv")
    check("...false alarms", 13, len(fp_idx), src="predictions_h6")
    pos_idx = preds.index[preds["target"] == 1]
    # the label counts early recession months too, so most positives are in-recession
    check("Positive test months already inside a recession", 18, int(df.loc[pos_idx, "USREC"].sum()), src="dataset.csv")
    # 'all 16 correct alarms came from December 2007 on'
    check("Correct logistic alarms span", "Dec 2007-Mar 2009", f"{ym(tp_idx.min())}-{ym(tp_idx.max())}",
          src="predictions_h6")
    # 'every recession month but the last also counts'
    rec = df["USREC"] == 1
    last_of_run = rec & ~rec.shift(-1, fill_value=False)
    labelled = df[TARGET].notna()
    check("Every recession month except the last has label 1", True,
          bool((df.loc[rec & ~last_of_run & labelled, TARGET] == 1).all()
               and (df.loc[last_of_run & labelled, TARGET] == 0).all()), src="dataset.csv")
    out = preds[df.loc[preds.index, "USREC"] == 0]
    check("Scored only outside recessions: logistic ROC-AUC", 0.836,
          round(roc_auc_score(out["target"], out[LR]), 3), src="predictions_h6")
    check("Scored only outside recessions: probit ROC-AUC", 0.735,
          round(roc_auc_score(out["target"], out[PROBIT]), 3), src="predictions_h6")
    print("        false-alarm runs:", ", ".join(runs(fp_idx)))
    fp22 = fp_idx[(fp_idx >= "2022-01-01") & (fp_idx <= "2022-12-01")]
    check("False alarms in 2022", 7, len(fp22), src="predictions_h6")
    check("...running", "May 2022-Nov 2022", f"{ym(fp22.min())}-{ym(fp22.max())}", src="predictions_h6")
    p22 = preds.loc["2022", LR]
    check("2022 logistic peak probability", 0.95, round(p22.max(), 2), src="predictions_h6")
    check("...in month", "Jul 2022", ym(p22.idxmax()), src="predictions_h6")
    check("Dec 2022 logistic probability", 0.46, round(preds.loc["2022-12-01", LR], 2), src="predictions_h6")
    # dataset rows are already lagged one month: row May 2022 holds April 2022 data
    check("Spread in May 2022 forecast (Apr data, pp)", 1.99, round(df.loc["2022-05-01", "SPREAD"], 2), src="dataset.csv")
    check("Spread in Jul 2022 forecast (Jun data, pp)", 1.65, round(df.loc["2022-07-01", "SPREAD"], 2), src="dataset.csv")
    check("6-month equity return, Mar 2022 data (%)", -1.6, round(df.loc["2022-04-01", "EQUITY_RET6"], 1), src="dataset.csv")
    check("6-month equity return, Apr 2022 data (%)", -20.4, round(df.loc["2022-05-01", "EQUITY_RET6"], 1), src="dataset.csv")
    # 'Inflation far above anything in training': every one of the seven
    # 2022 alarm forecasts sees CPI above the training-period maximum.
    check("CPI behind all seven 2022 alarms > training max", True,
          bool((df.loc[fp22, "CPI_YOY"] > train["CPI_YOY"].max()).all()), src="dataset.csv")

    pr_alarm = preds.index[preds[PROBIT] >= 0.5]
    check("Probit max probability in any positive month", 0.445, round(preds.loc[pos_idx, PROBIT].max(), 3),
          src="predictions_h6")
    # Ends Aug 2019: Sep-Dec 2019 labels are already driven by the COVID recession.
    pre19 = preds.loc[:"2019-08-01"]
    check("Probit ROC-AUC, 2006 to Aug 2019 (no COVID months)", 0.466,
          round(roc_auc_score(pre19["target"], pre19[PROBIT]), 3), src="predictions_h6")
    check("Probit alarm months", 21, len(pr_alarm), src="predictions_h6")
    # Not one unbroken run: Nov 2023 sits at 0.499, just under the cutoff.
    check("...all between", "Jan 2023-Oct 2024", f"{ym(pr_alarm.min())}-{ym(pr_alarm.max())}", src="predictions_h6")
    print("        probit alarm runs:", ", ".join(runs(pr_alarm)))
    check("...and every one a false alarm", 0, int(preds.loc[pr_alarm, "target"].sum()), src="predictions_h6")
    check("Probit peak probability", 0.734, round(preds[PROBIT].max(), 3), src="predictions_h6")

    print("\n2007-09 case study (results/predictions_h6.csv)")
    pre = preds.loc[:"2007-11-01", PROBIT]
    check("Probit peak before the recession", 0.46, round(pre.max(), 2), src="predictions_h6")
    check("...in month", "Apr 2007", ym(pre.idxmax()), src="predictions_h6")
    check("Probit reached 0.5 at any point 2006-2009", False,
          bool((preds.loc[:"2009-12-01", PROBIT] >= 0.5).any()), src="predictions_h6")
    check("Probit in Dec 2007", 0.19, round(preds.loc["2007-12-01", PROBIT], 2), src="predictions_h6")
    for model, month in [(LR, "Dec 2007"), (RF, "Jan 2008"), (XGB, "Jan 2008")]:
        w = preds.loc["2007-01-01":"2009-12-01", model]
        check(f"{model} first crosses 0.5 (2007 on)", month, ym(w.index[w >= 0.5][0]), src="predictions_h6")
    check("Logistic probability in Dec 2007", 0.58, round(preds.loc["2007-12-01", LR], 2), src="predictions_h6")
    sahm = df.loc["2007-01-01":"2009-12-01", "SAHM_GAP"]
    check("Sahm gap first reaches 0.5 (forecast month)", "Mar 2008", ym(sahm.index[sahm >= 0.5][0]), src="dataset.csv")

    print("\nHow settled the ranking is (paper: Results and Discussion)")
    y = preds["target"].to_numpy()
    a, b = preds[LR].to_numpy(), preds[RF].to_numpy()
    rng, n, diffs = np.random.default_rng(42), len(y), []
    for _ in range(1000):  # paired moving-block bootstrap, same scheme as evaluate.py
        starts = rng.integers(0, n - 12 + 1, size=int(np.ceil(n / 12)))
        idx = np.concatenate([np.arange(s, s + 12) for s in starts])[:n]
        if len(np.unique(y[idx])) == 2:
            diffs.append(roc_auc_score(y[idx], a[idx]) - roc_auc_score(y[idx], b[idx]))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    check("Logistic minus forest ROC-AUC, 95% low", -0.055, round(lo, 3), src="bootstrap")
    check("Logistic minus forest ROC-AUC, 95% high", 0.284, round(hi, 3), src="bootstrap")
    rf_all = RandomForestClassifier(n_estimators=300, max_depth=6, min_samples_leaf=5, max_features=None,
                                    class_weight="balanced", random_state=42, n_jobs=-1)
    rf_all.fit(train[FEATURES], train[TARGET].astype(int))
    check("Same forest, every indicator at each split: ROC-AUC", 0.897,
          round(roc_auc_score(test[TARGET], rf_all.predict_proba(test[FEATURES])[:, 1]), 3), src="refit")
    p10 = train_test_eval(df, H, "2010-01-01")
    check("2010 split: probit ROC-AUC", 0.826, round(roc_auc_score(p10["target"], p10[PROBIT]), 3), src="refit")
    check("2010 split: logistic ROC-AUC", 0.697, round(roc_auc_score(p10["target"], p10[LR]), 3), src="refit")
    # 'Splits from 2004 to 2008 keep the order': logistic first, probit last
    for sp in ["2004-01-01", "2005-01-01", "2007-01-01", "2008-01-01"]:
        ps = train_test_eval(df, H, sp)
        aucs = pd.Series({m: roc_auc_score(ps["target"], ps[m]) for m in ps.columns.drop("target")})
        check(f"Split {sp[:4]}: logistic first, probit last", True,
              bool(aucs.idxmax() == LR and aucs.idxmin() == PROBIT), src="refit")
    # Paired interval for logistic minus probit ('borderline')
    c = preds[PROBIT].to_numpy()
    rng2, d2 = np.random.default_rng(42), []
    for _ in range(1000):
        starts = rng2.integers(0, n - 12 + 1, size=int(np.ceil(n / 12)))
        idx = np.concatenate([np.arange(s, s + 12) for s in starts])[:n]
        if len(np.unique(y[idx])) == 2:
            d2.append(roc_auc_score(y[idx], a[idx]) - roc_auc_score(y[idx], c[idx]))
    lo2, hi2 = np.percentile(d2, [2.5, 97.5])
    check("Logistic minus probit ROC-AUC, 95% low", -0.018, round(lo2, 3), src="bootstrap")
    check("Logistic minus probit ROC-AUC, 95% high", 0.707, round(hi2, 3), src="bootstrap")

    print("\nRandom forest importance (NOT in the 4-page paper; README, table3 csv, fig4 png)")
    rf = RandomForestClassifier(n_estimators=500, max_depth=6, min_samples_leaf=5,
                                class_weight="balanced", random_state=42, n_jobs=-1)
    rf.fit(train[FEATURES], train[TARGET].astype(int))
    imp = pd.Series(rf.feature_importances_ * 100, index=FEATURES).round(1)
    for var, v in [("EQUITY_RET6", 25.3), ("CLAIMS_YOY", 24.7), ("SPREAD", 13.9),
                   ("INDPRO_YOY", 8.6), ("UNRATE_CHG12", 7.8), ("CPI_YOY", 6.4)]:
        check(f"Importance {var} (%)", v, imp[var], src="refit")
    check("Top three, in order", "EQUITY_RET6>CLAIMS_YOY>SPREAD",
          ">".join(imp.sort_values(ascending=False).index[:3]), src="refit")
    saved = pd.read_csv(RESULTS_DIR / "tables" / "table3_feature_importance.csv").set_index("Variable")
    check("Saved table3 csv agrees with this refit", True,
          bool(np.allclose(saved.loc[FEATURES, "Importance (%)"].to_numpy(), imp[FEATURES].to_numpy())),
          src="table3 csv")
    # Not circular: the forest actually scored in Table 1 (300 trees, models.py)
    rf_eval = make_models(train[TARGET].astype(int))[RF].fit(train[FEATURES], train[TARGET].astype(int))
    imp_eval = pd.Series(rf_eval.feature_importances_, index=FEATURES).sort_values(ascending=False)
    check("Evaluated 300-tree forest has the same top three", "EQUITY_RET6>CLAIMS_YOY>SPREAD",
          ">".join(imp_eval.index[:3]), src="models.py fit")

    print("\nLogistic regression coefficients (NOT in the 4-page paper; README / original report)")
    y_tr = train[TARGET].astype(int)
    lr = make_models(y_tr)[LR].fit(train[FEATURES], y_tr)
    coef = pd.Series(lr[-1].coef_[0], index=FEATURES).round(2)
    for var, v in [("EQUITY_RET6", -2.29), ("SPREAD", -1.70), ("FEDFUNDS_CHG12", -1.64), ("CPI_YOY", 1.47)]:
        check(f"Coefficient {var}", v, coef[var], src="refit")

    # What pushed the 2022 false alarm up: each feature's contribution to the
    # log-odds (standardised value x coefficient), averaged May-Jul 2022,
    # the months the alarm started and peaked.
    win = df.loc["2022-05-01":"2022-07-01", FEATURES]
    contrib = pd.DataFrame(lr[0].transform(win) * lr[-1].coef_[0], columns=FEATURES).mean()
    contrib = contrib.sort_values(ascending=False)
    print("        mean log-odds contribution, May-Jul 2022:",
          ", ".join(f"{k} {v:+.2f}" for k, v in contrib.head(4).items()))
    check("2022 alarm: two largest drivers", "CPI_YOY>EQUITY_RET6", ">".join(contrib.index[:2]), src="refit")
    check("2022 alarm: spread contribution small May-Jul (<0.5)", True, bool(abs(contrib["SPREAD"]) < 0.5), src="refit")
    # 'the flattening curve joined from August'
    late = df.loc["2022-08-01":"2022-11-01", FEATURES]
    late_sp = pd.DataFrame(lr[0].transform(late) * lr[-1].coef_[0], columns=FEATURES)["SPREAD"]
    check("2022 alarm: spread contribution Aug-Nov all > +1", True, bool((late_sp > 1).all()), src="refit")
    check("Highest CPI inflation in training years (%)", 6.38, round(train["CPI_YOY"].max(), 2), src="dataset.csv")
    check("CPI behind May-Jul 2022 forecasts all above 8%", True, bool((win["CPI_YOY"] > 8).all()), src="dataset.csv")
    # Signs the paper calls 'contrary to theory': a wider Sahm gap should
    # raise recession risk (expect +), higher sentiment should lower it (expect -).
    check("Sahm gap coefficient is negative (wrong sign)", True, bool(coef["SAHM_GAP"] < 0), src="refit")
    check("Sentiment level coefficient is positive (wrong sign)", True, bool(coef["UMCSENT_LEVEL"] > 0), src="refit")
    check("Credit spread coefficient near zero (|c|<0.1)", True, bool(abs(coef["CREDIT_SPREAD"]) < 0.1), src="refit")

    print("\nBaseline")
    check("Accuracy of always saying 'no recession' (%)", 88, round(100 * (1 - test[TARGET].mean())), src="dataset.csv")

    print("\nDeterminism (same code re-run; shows the saved file is current, not that the code is right)")
    fresh = train_test_eval(df, H)
    model_cols = list(fresh.columns.drop("target"))
    check("predictions_h6.csv matches a fresh fit", True,
          bool(np.allclose(fresh[model_cols].to_numpy(), preds[model_cols].to_numpy(), atol=1e-6)),
          src="evaluate.py")

    print(f"\n{'All checks passed.' if failures == 0 else f'{failures} check(s) FAILED.'}")
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
