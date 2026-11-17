import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import average_precision_score, precision_recall_curve

from pipeline.improve_cv import COUNTY_BASE, make_model

# repo root, so it runs from any folder
ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------
# PURPOSE
# DESCRIPTIVE chart, made AFTER the single final run. It is not used to
# change the threshold or anything else. It answers: at the 2020-2024
# results, does the model's precision-recall curve sit above the two
# rules? Each rule is a single point (one precision, one recall).
# If a rule's point lies BELOW the curve, the model gets a higher
# precision at that rule's recall, whatever threshold was chosen.
# The model is fitted the same way as in final_run.py (same data, same
# settings, same seed), so the probabilities are the same.
# Output: docs/final_pr_curve_2020_2024.png
# ---------------------------------------------------------------

df = pd.read_csv(ROOT / "data/processed/county_features.csv", parse_dates=["date"])
train, test = df[df.date < "2020-01-01"], df[df.date >= "2020-01-01"].copy()
model = make_model().fit(train[COUNTY_BASE], train.flooded)
test["prob"] = model.predict_proba(test[COUNTY_BASE])[:, 1]

prec, rec, _ = precision_recall_curve(test.flooded, test.prob)
base = test.flooded.mean()

res = pd.read_csv(ROOT / "docs/final_results_2020_2024.csv").set_index("method")
print(f"model PR-AUC {average_precision_score(test.flooded, test.prob):.3f}, base rate {base:.3f}")
for name in ["plain", "combined"]:
    r0, p0 = res.loc[name, "recall"], res.loc[name, "precision"]
    p_model = prec[rec >= r0].max()      # best precision the model reaches at that recall or higher
    print(f"{name:<9} rule: precision {p0:.3f} at recall {r0:.3f} | "
          f"model at that recall: precision {p_model:.3f} "
          f"({'above' if p_model > p0 else 'NOT above'} the rule)")

fig, ax = plt.subplots(figsize=(6.5, 4.5))
ax.plot(rec, prec, color="steelblue", label="model (all thresholds)")
ax.axhline(base, color="grey", ls=":", label=f"random guess ({base:.3f})")
ax.scatter(res.loc["plain", "recall"], res.loc["plain", "precision"],
           color="darkorange", s=60, zorder=3, label="plain rule (rain >= 50 mm)")
ax.scatter(res.loc["combined", "recall"], res.loc["combined", "precision"],
           color="firebrick", s=60, zorder=3, label="combined rule (50 mm and 200%)")
ax.scatter(res.loc["model", "recall"], res.loc["model", "precision"],
           color="black", marker="x", s=70, zorder=3, label="model at chosen threshold")
ax.set_xlabel("recall (share of floods caught)")
ax.set_ylabel("precision (share of alerts that are real)")
ax.set_title("County level, 2020-2024")
ax.set_ylim(0, max(0.35, prec.max() * 1.05))
ax.legend(fontsize=8)
fig.text(0.01, 0.005, "Descriptive, made after the single final run. Not used to change the threshold.",
         fontsize=7, color="grey")
plt.tight_layout(rect=(0, 0.02, 1, 1))
plt.savefig(ROOT / "docs/final_pr_curve_2020_2024.png", dpi=150)
print("saved docs/final_pr_curve_2020_2024.png")
