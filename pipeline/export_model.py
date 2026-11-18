"""
export_model.py - Task 4: turn the trained county model into a small JSON
file (model.json) that the Android app can use WITHOUT Python or sklearn.

Why this works: our model is logistic regression, which is just arithmetic:
    z      = (x - mean) / scale            (StandardScaler, per feature)
    logit  = intercept + sum(coef * z)     (the linear part)
    prob   = 1 / (1 + e^-logit)            (turns the logit into 0..1)
So if we save mean, scale, coef and intercept, the app can repeat the
same arithmetic and get the same answer as Python.

What gets saved in model.json:
  - the 17 feature names, in order
  - scaler mean + scale, coefficients, intercept
  - the alert threshold (0.646, picked on the training years in the protocol)
  - feature GROUPS, so the app can explain a result in plain words
  - per-county "normals": the usual rainfall for each of the 36 dekads
    (10-day periods) of the year, so the app can turn rain in mm into
    "% of normal"

The model is fitted on 2002-2019 ONLY. We never touch 2020-2024 here
(that test was one-shot, see docs/IMPROVEMENT_PROTOCOL.md section 7).

Run from the repo root:   python -m pipeline.export_model
Output:                   export/model.json
"""

import argparse
import bisect
import datetime
import json
import math
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from pipeline.improve_cv import COUNTY_BASE, cv_run, make_model

ROOT = Path(__file__).resolve().parent.parent  # repo root
THRESHOLD = 0.646  # alert cut-off, picked on the training years (protocol)
MIN_STEP_ROWS = 200  # every calibration step must rest on at least this many rows
DISPLAY_STEP = 0.05  # the chance shown to users is rounded to the nearest 5%

# Plain-language groups, used to explain WHY a county is at risk.
# Every one of the 17 features must be in exactly one group.
GROUPS = {
    "this_10_days": ["rfh", "rfq", "rfh_max", "rfq_max"],
    "past_month": ["r1h", "r1q", "r1q_max", "rfh_lag1", "rfh_lag2", "rfq_lag1"],
    "past_3_months": ["r3h", "r3q", "r3q_max"],
    "time_of_year": ["season_sin", "season_cos", "rfh_avg"],
    "rest_of_kenya": ["nat_frac_anom"],
}

NORMAL_COLS = ["rfh_avg", "r1h_avg", "r3h_avg"]  # 10-day, 1-month, 3-month normals

# How the Android app builds the 17 features from what the user types.
# Inputs: county, dekad of year (1..36), three 10-day rain values in mm
# (rfh = newest, rfh_lag1, rfh_lag2), the 3-month total r3h in mm, and a
# "how widespread" slider (0..1). Written into model.json so the app
# developer implements it exactly this way.
APP_NOTES = {
    "inputs": "county, dekad (1..36), rfh, rfh_lag1, rfh_lag2 (mm per 10 days, "
              "newest first), r3h (mm over 3 months), widespread slider 0..1",
    "exact": {
        "r1h": "rfh + rfh_lag1 + rfh_lag2",
        "rfh_avg": "normals[county].rfh_avg[dekad-1]",
        "season_sin": "sin(2*pi*dekad/36)",
        "season_cos": "cos(2*pi*dekad/36)",
    },
    "ratio_of_means": {
        "rfq": "100 * rfh / normals[county].rfh_avg[dekad-1]",
        "rfq_lag1": "100 * rfh_lag1 / normals[county].rfh_avg[previous dekad-1] "
                    "(previous of dekad 1 is dekad 36)",
        "r1q": "100 * r1h / normals[county].r1h_avg[dekad-1]",
        "r3q": "100 * r3h / normals[county].r3h_avg[dekad-1]",
        "caveat": "The model was trained on the average of each sub-county's "
                  "ratio; the app divides the county's mean rain by the county's "
                  "mean normal. Mean difference vs training: rfq 12, r1q 4, r3q 1 "
                  "points.",
    },
    "approximated": {
        "rfh_max": "set equal to rfh",
        "rfq_max": "set equal to rfq",
        "r1q_max": "set equal to r1q",
        "r3q_max": "set equal to r3q",
        "nat_frac_anom": "taken from the slider (0 = nowhere in Kenya well above "
                         "normal, 1 = everywhere)",
        "caveat": "The _max features are the wettest sub-county; the app has "
                  "only one county value. Tested on 30,456 training-year county "
                  "dekads: mean probability difference 0.008, and the alert "
                  "decision at the threshold differs on 0.52% of rows.",
    },
}


def git_source():
    """'SemProject_MLEngine@<short commit hash>' so we know which code made this.
    Adds '-dirty' if tracked files have uncommitted changes (the hash would lie)."""
    def git(*args):
        return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True,
                              text=True, check=True).stdout.strip()
    h = git("rev-parse", "--short", "HEAD")
    if git("status", "--porcelain", "--untracked-files=no"):
        h += "-dirty"
    return f"SemProject_MLEngine@{h}"


def dekad_of_year(dates):
    """1..36: three dekads per month (days 1-10, 11-20, 21-end).
    Same formula as pipeline/make_features.py, so the numbers line up."""
    return (dates.dt.month - 1) * 3 + (dates.dt.day // 10).clip(upper=2) + 1


# ---------------------------------------------------------------------------
# 1. Per-county normals from the HDX rainfall file
# ---------------------------------------------------------------------------
def build_normals():
    h = pd.read_csv(ROOT / "data/raw/ken-rainfall-subnat-full.csv", parse_dates=["date"])

    # The model's county numbers are averages over the SAMPLED sub-counties
    # (the 73 in features.csv), so the normals must use the same ones.
    sampled = pd.read_csv(ROOT / "data/processed/features.csv", usecols=["PCODE"]).PCODE.unique()
    h = h[(h.adm_level == 2) & h.PCODE.isin(sampled)].copy()  # level 2 = sub-counties
    h["dekad"] = dekad_of_year(h.date)

    # The "normal" is the same every year, so average over years to merge the
    # copies (and skip the few blanks in the first dekads of 1981).
    per_sub = h.groupby(["PCODE", "dekad"])[NORMAL_COLS].mean().reset_index()
    per_sub["county"] = per_sub.PCODE.str[:5]  # first 5 chars = county code

    # Then average the sampled sub-counties of each county.
    county = per_sub.groupby(["county", "dekad"])[NORMAL_COLS].mean()

    normals = {}
    for code, g in county.groupby(level="county"):
        g = g.droplevel("county").sort_index()  # ordered dekad 1..36
        assert list(g.index) == list(range(1, 37)), f"{code}: missing dekads"
        assert not g.isna().any().any(), f"{code}: blank normals"
        # index 0 of each list = dekad 1, index 35 = dekad 36
        normals[code] = {c: [round(float(v), 4) for v in g[c]] for c in NORMAL_COLS}
    return normals


# ---------------------------------------------------------------------------
# 2. score(): the SAME arithmetic the Android app will do, using only model.json
# ---------------------------------------------------------------------------
def _sigmoid(x):
    # 1 / (1 + e^-x), written so it never overflows for very big/small x
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def _contributions(features, model):
    """coef * z for each feature (how much each one pushes the logit)."""
    out = {}
    for name, mean, scale, coef in zip(
        model["features"], model["scaler_mean"], model["scaler_scale"], model["coefficients"]
    ):
        z = (features[name] - mean) / scale  # standardise
        out[name] = coef * z
    return out


def score(features, model):
    """features: dict {feature name: value}. Returns the flood probability."""
    logit = model["intercept"] + sum(_contributions(features, model).values())
    return _sigmoid(logit)


def calibrate(prob, model):
    """Turn the model's raw number into a calibrated chance (0..1) with a
    lookup table: find where `prob` sits between two table points and
    interpolate in a straight line. Outside the table we clip to the ends.
    The app does the same with model["calibration"]["x"] / ["y"]."""
    xs, ys = model["calibration"]["x"], model["calibration"]["y"]
    if prob <= xs[0]:
        return ys[0]
    if prob >= xs[-1]:
        return ys[-1]
    i = bisect.bisect_right(xs, prob)  # first table point above prob
    x0, x1, y0, y1 = xs[i - 1], xs[i], ys[i - 1], ys[i]
    return y0 + (y1 - y0) * (prob - x0) / (x1 - x0)


def display_chance(prob, model):
    """The calibrated chance rounded to the nearest 5% (0.05 steps).
    This is the ONLY chance number users may see. Anything under 2.5%
    rounds to 0, so the app should show that as "under 5%", not "0%"."""
    step = model["calibration"]["display_rounding"]
    c = calibrate(prob, model)
    return round(min(1.0, math.floor(c / step + 0.5) * step), 2)


def group_contributions(features, model):
    """Sum of coef*z per group. Positive = pushes risk UP, negative = DOWN.
    This is what the explainable-AI part of the app shows."""
    c = _contributions(features, model)
    return {g: sum(c[f] for f in fs) for g, fs in model["groups"].items()}


# ---------------------------------------------------------------------------
# 2b. Calibration table: isotonic regression, with small steps merged
# ---------------------------------------------------------------------------
def _merge(a, b):
    """Join two neighbouring steps. A step = [rows, floods, lowest raw prob, highest raw prob]."""
    return [a[0] + b[0], a[1] + b[1], min(a[2], b[2]), max(a[3], b[3])]


def build_calibration(oof, min_rows=MIN_STEP_ROWS):
    """Raw probability -> real flood rate, as a lookup table.

    1. Isotonic regression finds a staircase: each step is a range of raw
       probabilities that share one flood rate, and the rate never goes down
       as the raw probability goes up.
    2. Plain isotonic can create steps resting on a handful of rows (the top
       one had 6), whose rate is mostly luck. So any step with fewer than
       `min_rows` rows is merged into the neighbour with the closest rate,
       and its rate becomes the pooled rate of the joined rows.
    3. Merging can break the "never goes down" rule, so neighbours that are
       out of order are pooled too (the pool-adjacent-violators idea).
    4. Repeat until every step has at least `min_rows` rows.
    """
    o = oof.sort_values("prob")
    prob, flood = o.prob.values, o.flooded.values
    fit = IsotonicRegression(out_of_bounds="clip").fit(prob, flood).predict(prob)

    # starting steps = runs of equal isotonic value
    steps, start = [], 0
    for i in range(1, len(prob) + 1):
        if i == len(prob) or fit[i] != fit[start]:
            steps.append([i - start, int(flood[start:i].sum()), prob[start], prob[i - 1]])
            start = i

    def rate(s):
        return s[1] / s[0]

    while len(steps) > 1:
        small = [i for i, s in enumerate(steps) if s[0] < min_rows]
        if not small:
            break
        i = min(small, key=lambda k: steps[k][0])  # the smallest step first
        near = [j for j in (i - 1, i + 1) if 0 <= j < len(steps)]
        j = min(near, key=lambda k: abs(rate(steps[k]) - rate(steps[i])))  # closest rate
        a, b = sorted((i, j))
        steps[a:b + 1] = [_merge(steps[a], steps[b])]
        k = 0  # fix any neighbours now out of order
        while k < len(steps) - 1:
            if rate(steps[k]) > rate(steps[k + 1]):
                steps[k:k + 2] = [_merge(steps[k], steps[k + 1])]
                k = max(k - 1, 0)
            else:
                k += 1

    # The lookup table: each step is flat from its lowest to its highest raw
    # probability; between two steps the value moves in a straight line.
    xs, ys = [], []
    for s in steps:
        for x in sorted({float(s[2]), float(s[3])}):
            xs.append(x)
            ys.append(rate(s))
    return {
        "method": f"isotonic regression, steps with fewer than {min_rows} rows merged "
                  "into their neighbours (pool-adjacent-violators)",
        "fitted_on": f"out-of-fold predictions of the 2002-2019 time-series CV "
                     f"(5 folds, {len(oof)} county dekads)",
        "min_rows_per_step": min_rows,
        "use": "chance = linear interpolation of y over x; below x[0] -> y[0], "
               "above x[-1] -> y[-1]. Then round to the nearest display_rounding.",
        "display_rounding": DISPLAY_STEP,
        "steps": [{"rows": s[0], "floods": s[1], "raw_from": round(float(s[2]), 4),
                   "raw_to": round(float(s[3]), 4), "chance": round(rate(s), 4)} for s in steps],
        "x": xs,  # raw probability
        "y": ys,  # calibrated chance
    }


# ---------------------------------------------------------------------------
# 3. Fit the model and write model.json
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "export" / "model.json"))
    a = ap.parse_args()

    # Safety: groups must cover each of the 17 features exactly once.
    grouped = [f for fs in GROUPS.values() for f in fs]
    assert sorted(grouped) == sorted(COUNTY_BASE), "GROUPS do not match COUNTY_BASE"

    df = pd.read_csv(ROOT / "data/processed/county_features.csv", parse_dates=["date"])
    train = df[df.date < "2020-01-01"]  # 2002-2019 only; test years never used

    # Same model as the final run: scaler + logistic regression, no weights.
    pipe = make_model().fit(train[COUNTY_BASE], train.flooded)
    scaler = pipe.named_steps["standardscaler"]
    lr = pipe.named_steps["logisticregression"]

    # ---- calibration ----
    # The raw probability is inflated by class_weight="balanced" (flood
    # dekads are only ~4% of rows), so 0.65 does NOT mean "65% chance".
    # Isotonic regression learns the real flood rate for each raw value.
    # It is fitted ONLY on out-of-fold predictions: for each CV fold the
    # model predicts dekads it was not trained on. 2002-2019 only.
    # Steps resting on fewer than MIN_STEP_ROWS rows are merged (see above).
    _, oof = cv_run(train, COUNTY_BASE, rule_cols=("rfh_max", "rfq_max"))
    calibration = build_calibration(oof)

    model = {
        "version": 1,
        "source": git_source(),  # which commit of this repo produced the file
        "exported": datetime.date.today().isoformat(),
        "unit": "county",
        "trained_on": "2002-2019",
        "threshold": THRESHOLD,
        "features": COUNTY_BASE,
        "scaler_mean": [float(x) for x in scaler.mean_],
        "scaler_scale": [float(x) for x in scaler.scale_],
        "coefficients": [float(x) for x in lr.coef_[0]],
        "intercept": float(lr.intercept_[0]),
        "groups": GROUPS,
        "dekad_of_year": "1..36 = (month-1)*3 + (day//10, max 2) + 1; list index = dekad-1",
        "app_notes": APP_NOTES,
        "calibration": calibration,
        "normals": build_normals(),
    }

    # ---- check 1: our normals must match the rfh_avg the model was trained on ----
    tmp = train.copy()
    tmp["dekad"] = dekad_of_year(tmp.date)
    mine = [model["normals"][c]["rfh_avg"][d - 1] for c, d in zip(tmp.county, tmp.dekad)]
    normals_err = float(np.max(np.abs(np.array(mine) - tmp.rfh_avg.values)))
    print(f"normals vs training rfh_avg: max difference {normals_err:.4f} mm")

    # ---- check 2: score() must match sklearn's predict_proba ----
    sample = train.sample(5, random_state=0)
    sk = pipe.predict_proba(sample[COUNTY_BASE])[:, 1]
    print("\nrow  county  date        sklearn   score()   diff")
    for (i, row), p in zip(sample.iterrows(), sk):
        mine_p = score({f: row[f] for f in COUNTY_BASE}, model)
        print(f"{i:<5}{row.county}   {row.date.date()}  {p:.6f}  {mine_p:.6f}  {abs(p - mine_p):.1e}")

    big = train.sample(3000, random_state=1)  # wider check on 3000 rows
    sk_big = pipe.predict_proba(big[COUNTY_BASE])[:, 1]
    my_big = [score(r, model) for r in big[COUNTY_BASE].to_dict("records")]
    worst = float(np.max(np.abs(sk_big - np.array(my_big))))
    print(f"\nlargest difference over 3000 rows: {worst:.2e}")
    assert worst < 5e-4, "score() does not match sklearn to 3 decimal places"

    # ---- check 3: the calibration table is sound ----
    steps = calibration["steps"]
    assert all(s["rows"] >= MIN_STEP_ROWS for s in steps), "a step has too few rows"
    chances = [s["chance"] for s in steps]
    assert chances == sorted(chances), "chance goes down somewhere"
    assert all(b > a for a, b in zip(calibration["x"], calibration["x"][1:])), "x not increasing"
    grid = np.concatenate([sk_big, np.linspace(0, 1, 501)])  # real + edge values
    interp_err = max(abs(np.interp(p, calibration["x"], calibration["y"]) - calibrate(p, model))
                     for p in grid[:800])
    print(f"\ncalibration: {len(steps)} steps, every one with >= {MIN_STEP_ROWS} rows;"
          f" calibrate() vs numpy interp: {interp_err:.1e}")
    assert interp_err < 1e-12
    print("  rows  floods  raw range        chance")
    for s in steps:
        print(f"  {s['rows']:5d} {s['floods']:6d}  {s['raw_from']:.3f}-{s['raw_to']:.3f}    {s['chance']:.3f}")
    for p in (0.3, 0.5, model["threshold"], 0.8, 0.95, 0.999):
        print(f"  raw {p:.3f} -> chance {calibrate(p, model):.3f} -> shown as {display_chance(p, model):.2f}")
    print(f"  base rate of flooded county dekads in training: {train.flooded.mean():.3f}")

    out = Path(a.out)
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(model, indent=1))
    print(f"\nwrote {out}  ({out.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
