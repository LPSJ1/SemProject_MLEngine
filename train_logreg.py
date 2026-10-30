import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, roc_auc_score,
                             precision_score, recall_score, f1_score, confusion_matrix)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------
# PURPOSE
# Train the BASE MODEL - L2-regularised logistic regression - and
# compare it against the 50mm/200% threshold rule on unseen years.
# Output: data/predictions.csv in the agreed team format.
# ---------------------------------------------------------------

FEATURES = ["rfh", "rfh_avg", "rfq", "r1h", "r3h", "r1q", "r3q",
            "rfh_lag1", "rfh_lag2", "rfq_lag1",
            "season_sin", "season_cos", "nat_frac_anom"]
SEED = 42   # fixed so every result in the notebook is reproducible

df = pd.read_csv("data/features.csv", parse_dates=["date"]).sort_values(["date", "PCODE"])

# ---------- Time-based split ----------
# Train on 2002-2019, test on 2020-2024. The test years are never
# touched until the very end - not for tuning, not for anything.
df["split"] = np.where(df.date >= "2020-01-01", "test", "train")
train, test = df[df.split == "train"], df[df.split == "test"]
X_train, y_train = train[FEATURES], train.flooded
X_test, y_test = test[FEATURES], test.flooded
print(f"train: {len(train)} rows, {y_train.sum()} floods | "
      f"test: {len(test)} rows, {y_test.sum()} floods")


def make_model(C):
    # StandardScaler first: features are on wildly different scales
    # (mm of rain vs a 0-1 fraction), and regularisation penalises
    # coefficients by size, so unscaled features would be punished unfairly.
    # class_weight="balanced": floods are ~4% of rows; without this the
    # model can score 96% by always saying "no flood".
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(C=C, class_weight="balanced", max_iter=1000, random_state=SEED),
    )


# ---------- Cross-validation to choose regularisation strength ----------
# C controls L2 regularisation: SMALL C = strong penalty = simpler model.
# TimeSeriesSplit always validates on a period AFTER its training period.
# We split on whole dekads (not rows) so all 73 sub-counties of one dekad
# land on the same side - otherwise the same storm sits in both halves.
dates = np.sort(train.date.unique())
tscv = TimeSeriesSplit(n_splits=5)

# Score = average precision (area under the precision-recall curve).
# Not accuracy: with 4% floods, accuracy rewards ignoring floods.
print("\nC        mean PR-AUC across 5 time folds")
cv_scores = {}
for C in [0.001, 0.01, 0.1, 1, 10]:
    fold_scores = []
    for tr_idx, va_idx in tscv.split(dates):
        tr = train[train.date.isin(dates[tr_idx])]
        va = train[train.date.isin(dates[va_idx])]
        m = make_model(C).fit(tr[FEATURES], tr.flooded)
        fold_scores.append(average_precision_score(va.flooded, m.predict_proba(va[FEATURES])[:, 1]))
    cv_scores[C] = np.mean(fold_scores)
    print(f"{C:<8} {cv_scores[C]:.3f}   (folds: {', '.join(f'{s:.2f}' for s in fold_scores)})")

best_C = max(cv_scores, key=cv_scores.get)
print("chosen C:", best_C)

# ---------- Final model: all training years, chosen C ----------
model = make_model(best_C).fit(X_train, y_train)
test_prob = model.predict_proba(X_test)[:, 1]
test_pred = (test_prob >= 0.5).astype(int)

# ---------- The baseline rule on the same test rows ----------
rule_pred = ((test.rfh >= 50) & (test.rfq >= 200)).astype(int)


def report(name, y, pred, prob=None):
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    line = (f"{name:<14} precision {precision_score(y, pred):.3f}  "
            f"recall {recall_score(y, pred):.3f}  F1 {f1_score(y, pred):.3f}")
    if prob is not None:
        line += f"  ROC-AUC {roc_auc_score(y, prob):.3f}  PR-AUC {average_precision_score(y, prob):.3f}"
    print(line)
    print(f"{'':<14} caught {tp} floods, missed {fn}, false alarms {fp}")


print("\n=== TEST SET 2020-2024 (never seen during training or tuning) ===")
print(f"base rate: {y_test.mean():.3f}  <- a random guesser's PR-AUC")
report("50mm rule", y_test, rule_pred)
report("logreg @0.5", y_test, test_pred, test_prob)

# ---------- What did the model learn? ----------
# Features were standardised, so coefficient sizes are comparable:
# bigger = more influence. Positive = pushes towards "flood".
coefs = pd.Series(model[-1].coef_[0], index=FEATURES).sort_values(key=abs, ascending=False)
print("\ncoefficients (standardised):")
print(coefs.round(3).to_string())

# ---------- Save predictions in the agreed team format ----------
out = df[["PCODE", "date", "split", "flooded"]].rename(columns={"flooded": "y_true"})
out["rule_pred"] = ((df.rfh >= 50) & (df.rfq >= 200)).astype(int)
out["logreg_prob"] = model.predict_proba(df[FEATURES])[:, 1]
out.to_csv("data/predictions.csv", index=False)
print("\nsaved data/predictions.csv")