import pandas as pd
from pathlib import Path

# repo root, so it runs from any folder
ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------
# PURPOSE
# Turn the DAILY ROMI index (NOAA CPC, an OLR-based MJO index) into
# DEKAD features that line up with the rainfall rows.
# ROMI1 and ROMI2 describe where the MJO is; we use the two numbers
# themselves (not the phase number, which is circular: phase 8 sits
# next to phase 1).
# Output: data/processed/romi_dekad.csv
# ---------------------------------------------------------------

r = pd.read_csv(ROOT / "data/raw/romi.cpcolr.1x.txt", sep=r"\s+", header=None,
                names=["year", "month", "day", "hour", "romi1", "romi2", "amp"])
r["date"] = pd.to_datetime(r[["year", "month", "day"]])

# snap each day to the start of its dekad: 1st, 11th or 21st
day = r.date.dt.day
r["dekad"] = r.date - pd.to_timedelta(day - day.map(lambda d: 1 if d <= 10 else 11 if d <= 20 else 21), unit="D")

out = r.groupby("dekad")[["romi1", "romi2", "amp"]].mean().reset_index().rename(columns={"dekad": "date"})
out = out.sort_values("date")
# the MJO of the PREVIOUS dekad (it moves through Kenya over weeks)
out["romi1_lag1"] = out.romi1.shift(1)
out["romi2_lag1"] = out.romi2.shift(1)
# cut to our years AFTER the lag, so 2002's first dekad still has a lag
# (the file starts in 1991)
out = out[out.date.between("2002-01-01", "2024-12-31")]

out.to_csv(ROOT / "data/processed/romi_dekad.csv", index=False)
print("dekads:", len(out), "| first:", out.date.min().date(), "| last:", out.date.max().date())
print(out[["romi1", "romi2", "amp"]].describe().round(2).T[["mean", "std", "min", "max"]])
