import numpy as np
import pandas as pd

# ---------------------------------------------------------------
# The purpose of this is to:
# Build the model's inputs (features) for every (sub-county, dekad)
# row, then join them to the flood labels.
# it'll Output: data/features.csv - the final modelling table.
#
# The task is a NOWCAST: "given the rain observed up to and including
# this dekad, is this sub-county flooding?" Every feature only uses
# information from this dekad or earlier - never the future.
# ---------------------------------------------------------------

rain = pd.read_csv("data/ken-rainfall-subnat-full.csv", parse_dates=["date"])
rain = rain[rain.adm_level == 2].copy()          # sub-counties only
rain = rain.sort_values(["PCODE", "date"])       # required for lags to be correct

# ---------- 1. Features the dataset already gives us ----------
# rfh      rainfall this dekad (mm)
# rfh_avg  long-term average for this dekad here, how wet that place normally is
# rfq      this dekad as % of normal
# r1h/r3h  rainfall over the last 1 and 3 months - ANTECEDENT rain.
#          Saturated ground floods on less rain, so this captures soil
#          wetness, which we can't observe directly.
# r1q/r3q  those as % of normal
BASE = ["rfh", "rfh_avg", "rfq", "r1h", "r3h", "r1q", "r3q"]

# ---------- 2. Lags: what happened in the previous dekads ----------
# Computed per sub-county (groupby) so one place's rain never leaks
# into another's row. Computed on the FULL 1981+ series before we cut
# to 2002, so the very first 2002 rows still have a previous dekad.
g = rain.groupby("PCODE")
rain["rfh_lag1"] = g.rfh.shift(1)
rain["rfh_lag2"] = g.rfh.shift(2)
rain["rfq_lag1"] = g.rfq.shift(1)

# ---------- 3. Season, as a cycle ----------
# Dekad of year runs 1..36. Plain numbers would tell the model dekad 36
# (late Dec) is far from dekad 1 (early Jan) - they're neighbours.
# Sine/cosine wrap the year into a circle so Dec sits next to Jan.
dekad_of_year = (rain.date.dt.month - 1) * 3 + (rain.date.dt.day // 10).clip(upper=2) + 1
rain["season_sin"] = np.sin(2 * np.pi * dekad_of_year / 36)
rain["season_cos"] = np.cos(2 * np.pi * dekad_of_year / 36)

# ---------- 4. How widespread is the wet spell? ----------
# Share of all 73 sub-counties above 200% of normal this dekad.
# A storm hitting one place vs a country-wide El Nino season look
# identical to a single sub-county's rain gauge - this tells them apart.
rain["nat_frac_anom"] = rain.groupby("date").rfq.transform(lambda s: (s >= 200).mean())

FEATURES = BASE + ["rfh_lag1", "rfh_lag2", "rfq_lag1",
                   "season_sin", "season_cos", "nat_frac_anom"]

# DELIBERATELY EXCLUDED:
#  - year:  flood REPORTING collapsed after 2016, so year would teach
#           the model reporting history, not flooding.
#  - PCODE: would let the model memorise which counties report floods
#           most often, rather than learn from rainfall.

# ---------- Join to labels ----------
labels = pd.read_csv("data/labels.csv", parse_dates=["date"])
df = labels.merge(rain[["PCODE", "date"] + FEATURES], on=["PCODE", "date"], how="left")

print("rows:", len(df), "| positives:", df.flooded.sum(),
      f"({100 * df.flooded.mean():.2f}%)")
print("missing values per feature (should all be 0):")
print(df[FEATURES].isna().sum().to_string())

# Quick sanity check - flooded dekads should look wetter on every rain feature
print("\nmedian feature value, flooded vs not:")
print(df.groupby("flooded")[FEATURES].median().T.round(2).to_string())

df[["PCODE", "date", "flooded", "source"] + FEATURES].to_csv("data/features.csv", index=False)
print("\nsaved data/features.csv")