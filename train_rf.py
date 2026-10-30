import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_curve
from sklearn.model_selection import TimeSeriesSplit

# ---------------------------------------------------------------
# PURPOSE
# Train the COMPARISON MODEL - a random forest - with exactly the same
# data, split and cross-validation as the logistic regression, so any
# difference in results comes from the model, not the setup.
# Adds an rf_prob column to data/predictions.csv.
# ---------------------------------------------------------------

FEATURES = ["rfh", "rfh_avg", "rfq", "r1h", "r3h", "r1q", "r3q",
            "rfh_lag1", "rfh_lag2", "rfq_lag1",
            "season_sin", "season_cos", "nat_frac_anom"]
SEED = 42

df = pd.read_csv("data/features.csv", parse_dates=["date"]).sort_values(["date", "PCODE"])
df["split"] = np.where(df.date >= "2020-01-01", "test", "train")
train, test = df[df.split == "train"], df[df.split == "test"]


def make_model(min_leaf):
    # A random forest = many decision trees, each trained on a random
    # sample of rows and features, voting together. Unlike logistic
    # regression it can learn interactions ("heavy rain matters MORE
    # when the ground is already wet") without us coding them.
    # min_samples_leaf is its main brake on overfitting: each final
    # decision must be based on at least this many rows.
    # No scaling needed - trees split on thresholds, so units don't matter.
    return RandomForestClassifier(
        n_estimators=300, min_samples_leaf=min_leaf, max_features="sqrt",
        class_weight="balanced", n_jobs=-1, random_state=SEED,
    )


# ---------- Same time-series CV as logreg, tuning the overfitting brake ----------
dates = np.sort(train.date.unique())
tscv = TimeSeriesSplit(n_splits=5)

print("min_leaf  mean PR-AUC across 5 time folds")
cv_scores = {}
for leaf in [5, 20, 50, 100, 200]:
    fold_scores = []
    for tr_idx, va_idx in tscv.split(dates):
        tr = train[train.date.isin(dates[tr_idx])]
        va = train[train.date.isin(dates[va_idx])]
        m = make_model(leaf).fit(tr[FEATURES], tr.flooded)
        fold_scores.append(average_precision_score(va.flooded, m.predict_proba(va[FEATURES])[:, 1]))
    cv_scores[leaf] = np.mean(fold_scores)
    print(f"{leaf:<9} {cv_scores[leaf]:.3f}   (folds: {', '.join(f'{s:.2f}' for s in fold_scores)})")

best_leaf = max(cv_scores, key=cv_scores.get)
print("chosen min_samples_leaf:", best_leaf)

model = make_model(best_leaf).fit(train[FEATURES], train.flooded)
test_prob = model.predict_proba(test[FEATURES])[:, 1]

# ---------- Test results, on the same terms as logreg ----------
print("\n=== TEST SET 2020-2024 ===")
print(f"ROC-AUC {roc_auc_score(test.flooded, test_prob):.3f}  "
      f"PR-AUC {average_precision_score(test.flooded, test_prob):.3f}  "
      f"(base rate {test.flooded.mean():.3f})")

# Fair comparison with the rule: what precision at the rule's recall,
# and what recall at the rule's precision?
p, r, _ = precision_recall_curve(test.flooded, test_prob)
rule = (test.rfh >= 50) & (test.rfq >= 200)
rule_p = test.flooded[rule].mean()
rule_r = rule[test.flooded == 1].mean()
print(f"rule:            precision {rule_p:.3f} at recall {rule_r:.3f}")
# precision_recall_curve lists recall from high to low; the LAST point
# still at or above the rule's recall is the RF's best precision there
same_recall_precision = p[np.where(r >= rule_r)[0][-1]]
same_precision_recall = r[p >= rule_p].max()
print(f"RF at the rule's recall:    precision {same_recall_precision:.3f}")
print(f"RF at the rule's precision: recall {same_precision_recall:.3f}")

# ---------- What does the forest rely on? ----------
# Importance = how much each feature reduces uncertainty across all splits.
# Always positive - it says HOW MUCH a feature matters, not which direction.
imp = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=False)
print("\nfeature importance:")
print(imp.round(3).to_string())

# ---------- Add to the shared predictions file ----------
pred = pd.read_csv("data/predictions.csv", parse_dates=["date"])
pred = pred.drop(columns=["rf_prob"], errors="ignore")
rf_all = df[["PCODE", "date"]].assign(rf_prob=model.predict_proba(df[FEATURES])[:, 1])
pred = pred.merge(rf_all, on=["PCODE", "date"], how="left")
pred.to_csv("data/predictions.csv", index=False)
print("\nadded rf_prob to data/predictions.csv")