import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# repo root, so it runs from any folder
ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------
# PURPOSE
# Look before modelling: is Kenya's rain (rfq = rain compared with normal)
# different in different MJO phases? Only an exploration chart. It is not
# part of the keep/reject decision.
# The phase number is worked out from the angle of (ROMI1, ROMI2), using
# the usual convention (phase 1 starts at 180 degrees, 8 sectors of 45
# degrees). Days when the MJO is weak (amplitude < 1) are left out.
# Output: docs/romi_phase_rainfall.png
# ---------------------------------------------------------------

romi = pd.read_csv(ROOT / "data/processed/romi_dekad.csv", parse_dates=["date"])
cty = pd.read_csv(ROOT / "data/processed/county_features.csv", parse_dates=["date"])

# one national number per dekad: mean rfq over all counties
rain = cty.groupby("date").rfq.mean().rename("rfq").reset_index()
d = rain.merge(romi, on="date")
d = d[d.date < "2020-01-01"]      # training years only, the test years are never looked at
d = d[d.amp >= 1].copy()

angle = np.degrees(np.arctan2(d.romi2, d.romi1)) % 360
d["phase"] = ((angle - 180) % 360 // 45 + 1).astype(int)

seasons = {"March-May (long rains)": [3, 4, 5], "Oct-Dec (short rains)": [10, 11, 12]}
fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
for ax, (name, months) in zip(axes, seasons.items()):
    s = d[d.date.dt.month.isin(months)]
    g = s.groupby("phase").rfq.agg(["mean", "count"]).reindex(range(1, 9))
    ax.bar(g.index, g["mean"], color="steelblue")
    ax.axhline(s.rfq.mean(), color="black", ls="--", lw=1, label="season average")
    for p, row in g.iterrows():
        ax.text(p, 2, f"n={int(row['count'])}", ha="center", fontsize=7, color="white")
    ax.set_title(name)
    ax.set_xlabel("MJO phase")
ax.legend()
axes[0].set_ylabel("mean rfq (rain as % of normal)")
plt.tight_layout()
plt.savefig(ROOT / "docs/romi_phase_rainfall.png", dpi=150)
print(d.groupby([d.date.dt.month.isin([3, 4, 5]).map({True: "MAM", False: "other"}), "phase"]).size().unstack().fillna(0).astype(int))
print("saved docs/romi_phase_rainfall.png")
