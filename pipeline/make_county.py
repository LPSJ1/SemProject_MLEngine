import pandas as pd
from pathlib import Path

# repo root, so it runs from any folder
ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------
# PURPOSE
# The flood records are county-level (one report is copied onto every
# sampled sub-county of that county), so also build ONE ROW PER COUNTY
# PER DEKAD. See section 9.2 of docs/IMPROVEMENT_PROTOCOL.md.
# Output: data/processed/county_features.csv
# ---------------------------------------------------------------

MEAN_COLS = ["rfh", "rfh_avg", "rfq", "r1h", "r3h", "r1q", "r3q",
             "rfh_lag1", "rfh_lag2", "rfq_lag1"]
MAX_COLS = ["rfh", "rfq", "r1q", "r3q"]          # the wettest sub-county
SHARED = ["season_sin", "season_cos", "nat_frac_anom"]   # same for the whole county

df = pd.read_csv(ROOT / "data/processed/features.csv", parse_dates=["date"])
df["county"] = df.PCODE.str[:5]      # first 5 characters of a sub-county code = its county

g = df.groupby(["county", "date"])
out = g[MEAN_COLS + SHARED].mean()
out = out.join(g[MAX_COLS].max().add_suffix("_max"))
out["flooded"] = g.flooded.max()             # flooded if ANY sampled sub-county is
out["record_days"] = g.record_days.min()     # shortest record touching the county dekad
out["n_sub"] = g.size()                      # how many sub-counties were sampled
out = out.reset_index()

# same weight rule as the sub-county table (protocol 9.1)
out["weight"] = 1.0
out.loc[out.record_days > 30, "weight"] = 0.1

# candidate MJO features (kept OUT of the baseline; used only through --extra)
romi = ROOT / "data/processed/romi_dekad.csv"
if romi.exists():
    out = out.merge(pd.read_csv(romi, parse_dates=["date"])
                    [["date", "romi1", "romi2", "romi1_lag1", "romi2_lag1"]],
                    on="date", how="left")

out.to_csv(ROOT / "data/processed/county_features.csv", index=False)

print("counties:", out.county.nunique(), "| rows:", len(out),
      "| flooded:", out.flooded.sum(), f"({100 * out.flooded.mean():.2f}%)")
print("sampled sub-counties per county: min", out.n_sub.min(), "max", out.n_sub.max())

# the label is copied onto every sub-county of a county, so they should agree
disagree = (g.flooded.nunique() > 1).sum()
print("county dekads where sampled sub-counties disagree on the label:", disagree)
