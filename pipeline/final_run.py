import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from pipeline.improve_cv import COUNTY_BASE, make_model, cv_run, recall_at_rule_precision

# repo root, so it runs from any folder
ROOT = Path(__file__).resolve().parent.parent
LOCK = ROOT / "data/processed/final_run_done.txt"

# ---------------------------------------------------------------
# PURPOSE
# The final run (protocol section 7): train on 2002-2019, pick the
# threshold on the training years, then score 2020-2024 ONCE.
# Final model = the county baseline with no weights (every experiment
# after it was rejected). Run:  python -m pipeline.final_run
# Try the code first with --dry-run, which uses only training years
# (train 2002-2014, "test" 2015-2019) and never touches 2020-2024.
# ---------------------------------------------------------------


def runs(x):
    # (start, end) positions of every run of consecutive 1s
    out, start = [], None
    for i, v in enumerate(x):
        if v and start is None:
            start = i
        if not v and start is not None:
            out.append((start, i - 1))
            start = None
    if start is not None:
        out.append((start, len(x) - 1))
    return out


def count_near(run_list, other, window=1):
    # how many runs have a 1 in `other` within `window` dekads of the run
    return sum(other[max(0, s - window): e + window + 1].any() for s, e in run_list)


def event_metrics(df, alert_col):
    # section 6, with the county as the unit
    events = caught = episodes = credible = 0
    for _, g in df.sort_values("date").groupby("county"):
        flood = g.flooded.values.astype(bool)
        alert = g[alert_col].values.astype(bool)
        ev, ep = runs(flood), runs(alert)
        events += len(ev)
        episodes += len(ep)
        caught += count_near(ev, alert)       # events with an alert within +-1 dekad
        credible += count_near(ep, flood)     # alert episodes within +-1 dekad of an event
    return events, caught, episodes, credible


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="test the code on training years only")
    a = ap.parse_args()

    df = pd.read_csv(ROOT / "data/processed/county_features.csv", parse_dates=["date"])
    if a.dry_run:
        train = df[df.date < "2015-01-01"]
        test = df[(df.date >= "2015-01-01") & (df.date < "2020-01-01")]
        print("DRY RUN: train 2002-2014, 'test' 2015-2019 (2020-2024 not used)")
    else:
        if LOCK.exists():
            raise SystemExit("The final run was already done (see " + str(LOCK) +
                             "). It is one-shot, protocol section 7.")
        train = df[df.date < "2020-01-01"]
        test = df[df.date >= "2020-01-01"]
        print("FINAL RUN: train 2002-2019, test 2020-2024")

    # ---- threshold from the TRAINING years only (section 4) ----
    _, oof = cv_run(train, COUNTY_BASE, rule_cols=("rfh_max", "rfq_max"))
    target, thr, oof_recall = recall_at_rule_precision(oof)
    if thr is None:
        raise SystemExit("no threshold reaches the combined rule's precision")
    print(f"threshold {thr:.3f} (combined rule's precision on training years {target:.3f}; "
          f"out-of-fold recall {oof_recall:.3f})")

    # ---- final model on all training rows, scored on the unseen years ----
    model = make_model().fit(train[COUNTY_BASE], train.flooded)
    test = test.copy()
    test["prob"] = model.predict_proba(test[COUNTY_BASE])[:, 1]
    test["model"] = (test.prob >= thr).astype(int)
    test["plain"] = (test.rfh_max >= 50).astype(int)
    test["combined"] = ((test.rfh_max >= 50) & (test.rfq_max >= 200)).astype(int)

    base_rate = test.flooded.mean()
    pr_auc = average_precision_score(test.flooded, test.prob)
    print(f"\ntest rows {len(test)}, floods {int(test.flooded.sum())}, base rate {base_rate:.3f}")
    print(f"model PR-AUC {pr_auc:.3f}  (lift {pr_auc / base_rate:.1f}x over the base rate)\n")

    rows = []
    for name in ["plain", "combined", "model"]:
        alerts = int(test[name].sum())
        tp = int(((test[name] == 1) & (test.flooded == 1)).sum())
        ev, caught, ep, cred = event_metrics(test, name)
        rows.append({"method": name, "alerts": alerts,
                     "precision": tp / alerts if alerts else np.nan,
                     "recall": tp / test.flooded.sum(),
                     "events": ev, "events_caught": caught,
                     "events_caught_pct": caught / ev,
                     "alert_episodes": ep, "credible": cred,
                     "credible_pct": cred / ep if ep else np.nan})
    res = pd.DataFrame(rows)
    print(res.round(3).to_string(index=False))

    out = "final_results_dry_run.csv" if a.dry_run else "final_results.csv"
    res.to_csv(ROOT / "data/processed" / out, index=False)
    print("\nsaved data/processed/" + out)
    if not a.dry_run:
        LOCK.write_text("final run completed; do not repeat (protocol section 7)\n")


if __name__ == "__main__":
    main()
