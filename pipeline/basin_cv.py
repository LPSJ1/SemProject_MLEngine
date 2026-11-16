import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# same 13 features as train_logreg.py (that file runs on import, so we can't import from it)
BASE = ["rfh", "rfh_avg", "rfq", "r1h", "r3h", "r1q", "r3q",
        "rfh_lag1", "rfh_lag2", "rfq_lag1",
        "season_sin", "season_cos", "nat_frac_anom"]
SEED, C = 42, 0.001

df = pd.read_csv("data/processed/features.csv", parse_dates=["date"])
train = df[df.date < "2020-01-01"]  #test years are never loaded below this line
dates = np.sort(train.date.unique())

def make_model():
    return make_pipeline(StandardScaler(),
                         LogisticRegression(C=C, class_weight="balanced",
                                             max_iter=1000, random_state=SEED))

def cv_scores(features):
    scores = []
    for tr_idx, va_idx in TimeSeriesSplit(n_splits=5).split(dates):
        tr = train[train.date.isin(dates[tr_idx])]
        va = train[train.date.isin(dates[va_idx])]
        m  = make_model().fit(tr[features], tr.flooded)
        scores.append(average_precision_score(va.flooded, m.predict_proba(va[features])[:, 1]))
    return scores 

variants = {
    "baseline (13)":            BASE,
    "+ basin_rfq (K=0)":        BASE + ["basin_rfq"],
    "+ basin lag1 (K=1)":       BASE + ["basin_rfq_lag1"],
    "+ basin lag2 (K=2)":       BASE + ["basin_rfq_lag2"],
    "+all three":               BASE + ["basin_rfq", "basin_rfq_lag1", "basin_rfq_lag2"],
}
base_scores = cv_scores(BASE)
for name, feats in variants.items():
    s=cv_scores(feats)
    wins = sum(a > b for a, b in zip(s, base_scores))
    print(f"{name:<20} mean PR-AUC {np.mean(s):.3f} | folds: "
          f"{','.join(f'{x:.2f}' for x in s)} | better than baseline in {wins}/5 folds")