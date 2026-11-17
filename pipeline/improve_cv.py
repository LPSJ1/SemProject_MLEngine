import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# repo root, so it runs from any folder
ROOT = Path(__file__).resolve().parent.parent

BASE = ["rfh", "rfh_avg", "rfq", "r1h", "r3h", "r1q", "r3q",
        "rfh_lag1", "rfh_lag2", "rfq_lag1",
        "season_sin", "season_cos", "nat_frac_anom"]
SEED, C = 42, 0.001

# the keep rule from docs/IMPROVEMENT_PROTOCOL.md section 3
BASELINE_MEAN = 0.177 #baseline CV PR-AUC, as written in protocol
MIN_GAIN = 0.17     # mean PR-AUC must be 17% above baseline
MIN_FOLDS = 4       # and better in at least 4 of 5 folds

def make_model():
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(C=C, class_weight="balanced",
                           max_iter=1000, random_state=SEED),
    )

def load_train():
    df = pd.read_csv(ROOT / "data/processed/features.csv", parse_dates=["date"])
    return df[df.date < "2020-01-01"]       # test years never loaded

def cv_run(train, features, label_col="flooded", weight_col=None):
    dates = np.sort(train.date.unique())
    fold_scores, oof = [], []
    for tr_idx, va_idx in TimeSeriesSplit(n_splits=5).split(dates):
        tr = train[train.date.isin(dates[tr_idx])]
        va = train[train.date.isin(dates[va_idx])]

        fit_args = {}
        if weight_col:
            fit_args["logisticregression__sample_weight"] = tr[weight_col]
        # train on label_col (may be audit-extended), score on the original
        m = make_model().fit(tr[features], tr[label_col], **fit_args)

        prob = m.predict_proba(va[features])[:, 1]
        fold_scores.append(average_precision_score(va.flooded, prob))
        oof.append(va.assign(prob=prob)[["flooded", "rfh", "rfq", "prob"]])
    return fold_scores, pd.concat(oof)    


def recall_at_rule_precision(oof):
    # section 4: highest recall whose precision >= the combined rule's precision
    alerts = (oof.rfh >= 50) & (oof.rfq >= 200)
    target = oof.flooded[alerts].mean()
    prec, rec, thr = precision_recall_curve(oof.flooded, oof.prob)
    ok = prec[:-1] >= target
    if not ok.any():
        return target, None, 0.0
    i = np.where(ok, rec[:-1], -1).argmax()
    return target, thr[i], rec[i]



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extra", nargs="*", default=[], help="features to add to the 13")
    ap.add_argument("--label-col", default="flooded", help="training labels only")
    ap.add_argument("--weight-col", default=None, help="sample weights column")
    a = ap.parse_args()

    train = load_train()
    base, base_oof = cv_run(train, BASE)
    exp, exp_oof = cv_run(train, BASE + a.extra, a.label_col, a.weight_col)

    for name, s, oof in [("baseline", base, base_oof), ("experiment", exp, exp_oof)]:
        target, thr, rec = recall_at_rule_precision(oof)
        print(f"{name:<11} mean PR-AUC {np.mean(s):.3f} | "
              f"folds {', '.join(f'{x:.2f}' for x in s)} | std {np.std(s):.3f}")
        # lift = how many times better than guessing (the base rate)
        lift = np.mean(s) / oof.flooded.mean()
        print(f"{'':<11} lift {lift:.1f}x over the base rate {oof.flooded.mean():.3f}")
        print(f"{'':<11} recall {rec:.3f} at precision >= {target:.3f}"
              f" (threshold {thr})")

    wins = sum(e > b for e, b in zip(exp, base))
    needed = BASELINE_MEAN * (1 + MIN_GAIN)
    keep = np.mean(exp) >= needed and wins >= MIN_FOLDS
    print(f"\nneeded mean >= {needed:.3f}; got {np.mean(exp):.3f}; "
          f"better in {wins}/5 folds -> {'KEEP' if keep else 'REJECT'}")

if __name__ == "__main__":
    main()    