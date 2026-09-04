"""Regenerate every figure used in the report, at the 6-month horizon.

Run:  python -m src.plots      (after src.download and src.features)

Writes PNGs to results/figures/ with an _h6 suffix so the older
twelve-month figures are left untouched.
"""
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from pandas.tseries.offsets import DateOffset
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve

from src.config import FEATURES, FIGURES_DIR, SPLIT_DATE
from src.evaluate import load_dataset, train_test_eval

HORIZON = 6
COLORS = {
    "Probit (yield spread)": "#8c8c8c",
    "Sahm rule": "#e08214",
    "Logistic Regression": "#1f4e9c",
    "Random Forest": "#2e8b57",
    "XGBoost": "#a02c66",
}
STYLES = {
    "Probit (yield spread)": "--",
    "Sahm rule": ":",
    "Logistic Regression": "-",
    "Random Forest": "-.",
    "XGBoost": (0, (3, 1, 1, 1)),
}


def shade_recessions(ax, usrec: pd.Series) -> None:
    """Grey bars behind the data for NBER recession months."""
    inrec = usrec.fillna(0).astype(int)
    start = None
    for date, flag in inrec.items():
        if flag and start is None:
            start = date
        elif not flag and start is not None:
            ax.axvspan(start, date, color="0.85", zorder=0)
            start = None
    if start is not None:
        ax.axvspan(start, inrec.index[-1], color="0.85", zorder=0)


def fig_series_panel(df: pd.DataFrame) -> None:
    panel = [
        ("SPREAD", "Yield spread\n10y minus 3m (pp)"),
        ("UNRATE_CHG12", "Unemployment\n12-mo change (pp)"),
        ("INDPRO_YOY", "Industrial production\nYoY (%)"),
        ("UMCSENT_LEVEL", "Consumer sentiment\n(index)"),
        ("PERMIT_YOY", "Building permits\nYoY (%)"),
        ("CLAIMS_YOY", "Jobless claims\nYoY (%)"),
    ]
    fig, axes = plt.subplots(len(panel), 1, figsize=(9, 11), sharex=True)
    for ax, (col, label) in zip(axes, panel):
        shade_recessions(ax, df["USREC"])
        ax.plot(df.index, df[col], color="#1f4e9c", lw=1.0)
        ax.axhline(0, color="0.4", lw=0.6)
        ax.set_ylabel(label, fontsize=8)
        ax.tick_params(labelsize=8)
    axes[-1].set_xlabel("Year")
    fig.suptitle("Leading indicators, 1985-2026 (shaded = NBER recession)", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig1_series_panel_h6.png", dpi=150)
    plt.close(fig)


def fig_probabilities(preds: pd.DataFrame, usrec: pd.Series) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    shade_recessions(ax, usrec.loc[preds.index.min():preds.index.max()])
    for name in preds.columns.drop("target"):
        if name == "Sahm rule":       # a gap in pp, not a probability
            continue
        ax.plot(preds.index, preds[name], label=name, lw=1.2,
                color=COLORS[name], ls=STYLES[name])
    ax.axhline(0.5, color="0.3", lw=0.8, ls=":")
    ax.set_ylim(0, 1)
    ax.set_ylabel("P(recession within 6 months)")
    ax.set_xlabel("Year")
    ax.set_title("Out-of-sample recession probabilities, 6-month horizon")
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig2_oos_probabilities_h6.png", dpi=150)
    plt.close(fig)


def fig_roc(preds: pd.DataFrame) -> None:
    y = preds["target"].to_numpy()
    fig, ax = plt.subplots(figsize=(6, 6))
    for name in preds.columns.drop("target"):
        p = preds[name].to_numpy()
        fpr, tpr, _ = roc_curve(y, p)
        ax.plot(fpr, tpr, label=f"{name} ({roc_auc_score(y, p):.3f})",
                color=COLORS[name], ls=STYLES[name], lw=1.4)
    ax.plot([0, 1], [0, 1], color="0.5", lw=0.8, ls=":", label="Random (0.500)")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curves, out-of-sample, 6-month horizon")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig3_roc_curves_h6.png", dpi=150)
    plt.close(fig)


def fig_importance(df: pd.DataFrame) -> None:
    """Importances from the same fit that produces Table 3: training data
    only, ending HORIZON months before the split, max_depth=6. Fitting on
    the full sample instead would rank the features off data the model was
    never allowed to see."""
    target = f"TARGET_{HORIZON}M"
    train_end = pd.Timestamp(SPLIT_DATE) - DateOffset(months=HORIZON)
    data = df.dropna(subset=[target]).loc[:train_end]
    X, y = data[FEATURES], data[target].astype(int)
    rf = RandomForestClassifier(n_estimators=500, max_depth=6, min_samples_leaf=5,
                                class_weight="balanced", random_state=42).fit(X, y)
    imp = pd.Series(rf.feature_importances_ * 100, index=FEATURES).sort_values()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(imp.index, imp.to_numpy(), color="#1f4e9c")
    ax.set_xlabel("Share of total importance (%)")
    ax.set_title("Random Forest variable importance, 6-month horizon")
    for i, v in enumerate(imp.to_numpy()):
        ax.text(v + 0.3, i, f"{v:.1f}", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig4_feature_importance_h6.png", dpi=150)
    plt.close(fig)


def fig_gfc(preds: pd.DataFrame) -> None:
    """Zoom into 2007-2009 of the main test run.

    The split already trains only on pre-2006 data, so these months are
    genuine out-of-sample forecasts of the financial crisis.
    """
    win = preds.loc["2007-01-01":"2009-12-01"]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.axvspan(pd.Timestamp("2007-12-01"), pd.Timestamp("2009-06-01"),
               color="0.85", zorder=0, label="NBER recession")
    for name in win.columns.drop("target"):
        if name == "Sahm rule":
            continue
        ax.plot(win.index, win[name], label=name, lw=1.4,
                color=COLORS[name], ls=STYLES[name])
    ax.axhline(0.5, color="0.3", lw=0.8, ls=":")
    ax.set_ylim(0, 1)
    ax.set_ylabel("P(recession within 6 months)")
    ax.set_xlabel("Month")
    ax.set_title("Zoom on 2007-2009: forecasting the financial crisis")
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig5_gfc_case_study_h6.png", dpi=150)
    plt.close(fig)


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    df = load_dataset()
    preds = train_test_eval(df, HORIZON, SPLIT_DATE)

    fig_series_panel(df)
    fig_probabilities(preds, df["USREC"])
    fig_roc(preds)
    fig_importance(df)
    fig_gfc(preds)
    print(f"Wrote 5 figures to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
