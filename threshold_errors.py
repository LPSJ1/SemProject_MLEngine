import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, precision_score, recall_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------
# PURPOSE
# 1. CHOOSE the alert threshold for the base model - using training
#    years only, so the test years stay untouched.
# 2. ERROR ANALYSIS - on the test years, look at which floods the
#    model missed and where its false alarms come from.
# ---------------------------------------------------------------

FEATURES = ["rfh", "rfh_avg", "rfq", "r1h", "r3h", "r1q", "r3q",
            "rfh_lag1", "rfh_lag2", "rfq_lag1",
            "season_sin", "season_cos", "nat_frac_anom"]
SEED, C = 42, 0.001          # same settings train_logreg.py chose

df = pd.read_csv("data/features.csv", parse_dates=["date"]).sort_values(["date", "PCODE"])
pred = pd.read_csv("data/predictions.csv", parse_dates=["date"])
df = df.merge(pred[["PCODE", "date", "split", "logreg_prob", "rule_pred"]], on=["PCODE", "date"])
train, test = df[df.split == "train"], df[df.split == "test"]


def make_model():
    return make_pipeline(StandardScaler(),
                         LogisticRegression(C=C, class_weight="balanced",
                                            max_iter=1000, random_state=SEED))


# ================= PART 1: choose the threshold =================
# The threshold turns a probability into an alert. Picking it by looking
# at TEST results would be cheating - we'd be tuning on the exam paper.
# So we get "out-of-fold" probabilities on the TRAINING years: each
# validation period is scored by a model that never saw it.
dates = np.sort(train.date.unique())
oof = []
for tr_idx, va_idx in TimeSeriesSplit(n_splits=5).split(dates):
    tr = train[train.date.isin(dates[tr_idx])]
    va = train[train.date.isin(dates[va_idx])]
    m = make_model().fit(tr[FEATURES], tr.flooded)
    oof.append(pd.DataFrame({"y": va.flooded, "p": m.predict_proba(va[FEATURES])[:, 1],
                             "rule": ((va.rfh >= 50) & (va.rfq >= 200)).astype(int)}))
oof = pd.concat(oof)

# RULE FOR THE THRESHOLD: catch as many floods as possible (recall first,
# because a missed flood strands a responder) WITHOUT being less precise
# than the rule already in use. "At least as trustworthy as today's
# alerts, and catches more."
rule_precision = precision_score(oof.y, oof.rule)
p, r, th = precision_recall_curve(oof.y, oof.p)
ok = np.where(p[:-1] >= rule_precision)[0]          # thresholds meeting the bar
best = ok[np.argmax(r[ok])]                         # ...with the highest recall
threshold = th[best]
print(f"rule's precision on training years: {rule_precision:.3f}")
print(f"chosen threshold: {threshold:.3f}  "
      f"(training-years recall {r[best]:.3f}, precision {p[best]:.3f})")

# ================= PART 2: does it hold on the unseen years? =================
test = test.copy()
test["alert"] = (test.logreg_prob >= threshold).astype(int)
print("\n=== TEST 2020-2024 at the chosen threshold ===")
for name, col in [("50mm rule", "rule_pred"), ("logreg", "alert")]:
    print(f"{name:<10} precision {precision_score(test.flooded, test[col]):.3f}  "
          f"recall {recall_score(test.flooded, test[col]):.3f}  "
          f"alerts {test[col].sum()}  floods caught {(test[col] & test.flooded).sum()}")

# ================= PART 3: error analysis =================
floods = test[test.flooded == 1]
caught, missed = floods[floods.alert == 1], floods[floods.alert == 0]

print(f"\n--- MISSED floods ({len(missed)}) vs CAUGHT floods ({len(caught)}) ---")
print("median values:")
cols = ["rfh", "rfq", "r1q", "r3q", "nat_frac_anom"]
print(pd.DataFrame({"caught": caught[cols].median(), "missed": missed[cols].median()}).round(2).to_string())

print("\nshare of floods missed, by label source:")
print(floods.groupby("source").alert.apply(lambda a: f"{1 - a.mean():.0%} missed of {len(a)}").to_string())

print("\nshare of floods missed, by month:")
print(floods.groupby(floods.date.dt.month).alert
      .apply(lambda a: f"{1 - a.mean():.0%} of {len(a)}").to_string())

# False alarms: alerts where no flood was recorded. Are they random,
# or clustered right next to real floods (same sub-county, adjacent dekad)?
test = test.sort_values(["PCODE", "date"])
g = test.groupby("PCODE").flooded
test["near_flood"] = ((g.shift(1) == 1) | (g.shift(-1) == 1)).astype(int)
fa = test[(test.alert == 1) & (test.flooded == 0)]
print(f"\n--- FALSE ALARMS ({len(fa)}) ---")
print(f"right next to a recorded flood (dekad before/after): {fa.near_flood.mean():.0%}")
print(f"during a country-wide wet spell (>=10% of sub-counties anomalous): "
      f"{(fa.nat_frac_anom >= 0.10).mean():.0%}")
print("false alarms by year:")
print(fa.date.dt.year.value_counts().sort_index().to_string())

# ---------- Where are floods missed most? ----------
# Missed floods had near-normal LOCAL rain. Check whether that's
# concentrated in counties that flood from rain falling elsewhere:
# Tana River and Garissa sit downstream on the Tana River, which is fed
# by rain in the central highlands (Mt Kenya / Aberdares), not locally.
names = pd.read_csv("data/kenya_counties.csv").set_index("admin1Pcode").admin1Name_en
floods = floods.assign(county=floods.PCODE.str[:5].map(names))
print("\nmiss rate by county (counties with the most test floods):")
by_county = floods.groupby("county").alert.agg(floods="size", missed=lambda a: 1 - a.mean())
print(by_county.sort_values("floods", ascending=False).head(10).round(2).to_string())

downstream = floods.county.isin(["Tana River", "Garissa"])
print(f"\nTana River + Garissa: {1 - floods[downstream].alert.mean():.0%} missed "
      f"| all other counties: {1 - floods[~downstream].alert.mean():.0%} missed")