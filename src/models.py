"""Model zoo: two traditional benchmarks and three ML models.

The deliberate ladder (see PLAN.md §1.2):
  probit (1 variable)  ->  logit (all variables, linear)  ->  RF / XGBoost
                                                               (nonlinear)
If logit beats probit, the gain came from richer information; if the tree
models beat logit, the gain came from nonlinearity and interactions.
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


class ProbitSpread:
    """Estrella & Mishkin (1998): probit on the yield spread alone.

    The canonical 'traditional indicator' benchmark — the same functional
    form behind the New York Fed's published recession probabilities.
    """

    def fit(self, X: pd.DataFrame, y: pd.Series):
        exog = sm.add_constant(X[["SPREAD"]].to_numpy())
        self.result_ = sm.Probit(np.asarray(y, dtype=float), exog).fit(disp=0)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        exog = sm.add_constant(X[["SPREAD"]].to_numpy(), has_constant="add")
        p = np.asarray(self.result_.predict(exog))
        return np.column_stack([1 - p, p])


class SahmRule:
    """Sahm (2019) unemployment trigger, used as a scoring benchmark.

    The gap itself is the score (higher = more recession-like). It is an
    indicator, not a calibrated probability, so evaluate it with
    threshold-free metrics (ROC-AUC, PR-AUC) only.
    """

    is_probabilistic = False

    def fit(self, X: pd.DataFrame, y: pd.Series):
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        score = X["SAHM_GAP"].to_numpy(dtype=float)
        return np.column_stack([-score, score])


def make_models(y_train: pd.Series) -> dict:
    """Fresh model instances per walk-forward step.

    XGBoost's scale_pos_weight depends on the class balance of the current
    training window, so models are rebuilt rather than reused.
    """
    pos = max(int(np.sum(y_train)), 1)
    neg = len(y_train) - pos
    return {
        "Probit (yield spread)": ProbitSpread(),
        "Sahm rule": SahmRule(),
        "Logistic Regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000),
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=6,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=3,
            subsample=0.9,
            colsample_bytree=0.9,
            scale_pos_weight=neg / pos,
            random_state=42,
            n_jobs=-1,
            eval_metric="logloss",
        ),
    }
