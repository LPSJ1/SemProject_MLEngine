import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score
from sklearn.model_selection import TimeSeriesSplit

BASE = ["rfh", "rfh_avg", "rfq", "r1h", "r3h", "r1q", "r3q",
        "rfh_lag1", "rfh_lag2", "rfq_lag1",
        "season_sin", "season_cos", "nat_frac_anom"]

SEED = 42

df = pd.read_csv("data/processed/features.csv", parse_dates=["date"])
train = df[df.date < "2020-01-01"].copy()  #test years never loaded below here
dates = np.sort(train.date.unique())

#control: shuffle the upstream values among the rows that have them
# the "has a value" pattern stays (it identifies the place), the rainfall link is destroyed.
rng = np.random.default_rng(SEED)
has = train.upstream_rfq.notna()
train["upstream_shuffled"] = train.upstream_rfq
train.loc[has, "upstream_shuffled"] = rng.permutation(train.loc[has, "upstream_rfq"].values)
train["upstream_shuffled_lag1"] = train.upstream_rfq_lag1
train.loc[has, "upstream_shuffled_lag1"] = rng.permutation(train.loc[has, "upstream_rfq_lag1"].values)

def make_model():
    #heavily restrained: our labels are today are noisy, and the forest overfit them
    return HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200,
                                          min_samples_leaf=200, l2_regularization=1.0,
                                          class_weight="balanced", random_state=SEED)


def cv_scores(features):
    all_rows, sixteen = [], []
    for tr_idx, va_idx in TimeSeriesSplit(n_splits=5).split(dates):
        tr = train[train.date.isin(dates[tr_idx])]
        va = train[train.date.isin(dates[va_idx])]
        m = make_model().fit(tr[features], tr.flooded)
        p = m.predict_proba(va[features])[:, 1]
        all_rows.append(average_precision_score(va.flooded, p))
        k = va.upstream_rfq.notna().values
        sixteen.append(average_precision_score(va.flooded[k], p[k]))
    return all_rows, sixteen

variants = {
    "boosting baseline (13)":  BASE,
    "+ upstream_rfq (K=0)":    BASE + ["upstream_rfq"],
    "+ upstream lag1 (K=1)":   BASE + ["upstream_rfq_lag1"],
    "+ upstream lag2 (K=2)":   BASE + ["upstream_rfq_lag2"],
    "CONTROL: shuffled K=0":   BASE + ["upstream_shuffled"],
    "CONTROL: shuffled K=1":   BASE + ["upstream_shuffled_lag1"],
}    
for name, feats in variants.items():
    a, s = cv_scores(feats)
    print(f"{name:<24} all 73: {np.mean(a):.3f} | only the 16: {np.mean(s):.3f} | folds(all): {','.join(f'{x:.2f}' for x in a)}")