# prediction of US  recessions with ml

ml project meant to beat the yield curve at forecasting recessions, six months in advance.

Twelve monthly FRED indicators from 1985-2026. 
Five models: the Estrella-Mishkin yield-curve probit, the Sahm Rule, logistic regression, a random forest, and XGBoost. 
Trained once on 1985 to mid-2005, tested on 2006 onward, so the 2008 crisis and COVID are genuinely out of sample.


## Results

| Model | ROC-AUC | Precision | Recall |
|---|---|---|---|
| Probit (yield spread) | 0.457 | 0.000 | 0.000 |
| Sahm Rule | 0.774 | — | — |
| **Logistic Regression** | **0.885** | 0.552 | 0.533 |
| Random Forest | 0.825 | 0.636 | 0.233 |
| XGBoost | 0.780 | 0.700 | 0.233 |

*241 test months, 30 of them recession-preceding.*

![Out-of-sample recession probabilities](results/figures/fig2_oos_probabilities_h6.png)

**Three findings:**

**Logistic regression wins.** and both tree models lose to it , this matters as the models are first one variable then twelve then non linear . Logistic being linear and winning means that the gain is coming from *more indicators* not from finding complex patterns. Two recessions for training data out of 4 total isn't enough to learn 	it .
**The complex models are cautious.** XGBoost has the highest precision at 0.7 but the worst recall at 0.233 which means that if it predicts the positive class its usually right but it rarely states it .
**No model predicted the 2008 recession.** The probit peaked at 0.46 in April 2007  and never crossed 0.50. Logistic regression crossed in December 2007, the  month the recession began. XGBoost went 0.015 in December 2007 to 0.976 in January 2008, a month late which is awfully terrible for a warning.

## Quick start

```bash
pip install -r requirements.txt
python -m src.download    # 12 FRED series, no API key needed
python -m src.features    # builds the monthly dataset
python -m src.evaluate    # fits and scores the five models
python -m src.plots       # regenerates the figures
```

Deterministic: same data in, same numbers out. Runs in a few minutes.

## Repo

```
src/config.py      series list, features, split date
src/download.py    keyless FRED CSV ingestion
src/features.py    transformations, 1-month publication lag, target
src/models.py      the five models
src/evaluate.py    train/test split + metrics (seed 42)
src/plots.py       the five report figures
data/              raw FRED series + processed panel
results/           figures, tables, predictions
```

## Design notes

- **One-month publication lag.** Economic data is usually given with lag so to account for 
  this constraint, every feature is lagged. Without it the prediction is fiction 
  as that data would not exist the month it makes that prediction(look-ahead bias)
- **Split at 2006, not 2010.** Puts two recessions on each side. A 2010 split
  leaves only COVID to test against, as the model would already know the 2008 recession.
- **Training stops in July 2005, not December.** It tries to predict if a recession starts 
  in the next six months and hence training data stops mid 2005 as it gets tested from 2006 onwards.
- **Accuracy is not used.** Always saying "no recession" scores about 88%.

## Limitations

Only four recessions exist in the sample and two are in training, which is the
binding constraint on everything here.FRED serves revised data, not real-time
vintages (ALFRED would fix this).Several indicators overlap, which left a
couple of logistic coefficients with signs that don't match theory.Project is US only.

## Data

All series are public, from [FRED](https://fred.stlouisfed.org/). Accessed
25 August 2026. Full variable list in the report's Appendix A.
