# Improvement protocol

Written on 17/11/2026; before running any improvement experiment. This file is committed first so the rules below can't be changed after seeing results.

## 1. Baseline to beat

- Data: 60,444 rows (73 sub-counties x 828 dekads), 2,557 floods (4.23%).
- Train 2002-2019, test 2020-2024.
- Model: logistic regression, 13 features, C = 0.001, balanced class weights, seed 42.

- Baseline cross-validation PR-AUC: 0.177 (folds 0.09, 0.19, 0.23, 0.08, 0.31).
- Rules we compare against: plain (rfh >= 50) and combined (rfh >= 50 and rfq >= 200).

## 2. How we test a change

- Use only 2002-2019, with 5 time-ordered folds split on whole dekads.
- Score with PR-AUC, because floods are only 4% of rows.
- Same folds, same seed, same settings every time.
- Changing one thing at a time.

## 3. When we keep a change

A change is kept only if BOTH are true:

1. its mean PR-AUC is at least 17% higher than the current baseline (0.177 at sub-county level, so at least 0.207), and
2. it is better in at least 4 of the 5 folds. Each fold is compared with the baseline on the same fold, because both models see the same years.

I chose 17% because my fold scores vary a lot (0.08 to 0.31), so a small gain could be luck. 17% (0.03) is clearly larger than the noise but small enough to be a real improvement.

Anything else is rejected. We still write down the result.
This applies to feature experiments: a new feature must also beat a shuffled copy of itself (its values permuted across the training rows), so we know it is not just memorising places.
If it doesn't, it's rejected even if the PR-AUC gain passes.

## 4. How we choose the alert threshold

Using out-of-fold predictions on the training years, we pick the threshold with the highest recall whose precision is at least the combined rule's precision. On the training years that target is 0.170, and it gives a threshold of 0.669. The test years are never used to pick it.

## 5. Labels

Every experiment is scored against the original labels. Floods found in the false-alarm audit may be added to training labels only. They never decide whether a change is kept, because the audit picks cases the model is already confident about.

## 6. Event metric

- An event is a run of consecutive flooded dekads in one sub-county.
- An alert episode is a run of consecutive alert dekads in one sub-county.
- Events caught = events with an alert within 1 dekad of them, divided by all events.
- Credible alerts = alert episodes within 1 dekad of an event, divided by all alert episodes.
  We report both, because alerting everywhere would catch every event.
- At county level (section 9.2) the same definitions apply with the county as the unit. A county alerts in a dekad if its model score is above the threshold.

I allow 1 dekad (10 days) either side because the flood records have approximate dates, so a correct alert could land in the next or previous dekad. More than that would let unrelated alerts count as hits.

## 7. Test years

The final model is run on 2020-2024 once, at the end. The test years already gave us ideas: the baseline error analysis found that Tana River and Garissa floods were missed, which led to the upstream experiment.
I state this openly.

## 8. Experiments

| #   | Experiment                        | Result                                              |
| --- | --------------------------------- | --------------------------------------------------- |
| 1   | Basin wetness feature             | Rejected: 0.178 vs 0.177 (run before this protocol was written) |
| 2   | Strict upstream rain              | Rejected: no gain over baseline or shuffled control (run before this protocol was written) |
| 3   | Long-record weights               | Mean 0.195 vs 0.177 (+10%), folds 0.12, 0.20, 0.23, 0.10, 0.33, better in 5/5. Kept under 9.1, which was written after a simulation had shown this result; fails the 17% rule in section 3, so weaker evidence. Recall 0.551 vs 0.514 at precision 0.170. |
| 4   | County-level target (new unit)    | Adopted as the unit (9.2), not for its score. County baseline: mean 0.196, folds 0.12, 0.20, 0.24, 0.07, 0.35, lift 4.4x, recall 0.549 at precision 0.166 (47 counties, 38,916 rows, 3.97% flooded). Long-record weights at county level: mean 0.212 (+8%), folds 0.15, 0.21, 0.23, 0.10, 0.37, better in 4/5 (lost fold 3, 0.23 vs 0.24), recall 0.587. Rejected under 9.1, which needs 5/5. |
| 5   | MJO features                      | Pending                                             |
| 6   | Daily rain intensity              | Not run: source mismatch. CHIRPS v3 pentads (the only version found) match our HDX rainfall in total (median ratio 0.99) but not dekad by dekad (correlation 0.919 in 2002, below the 0.98 gate; worst sub-county 0.68). My polygon averaging reproduces v3's own dekad files exactly (correlation 1.0), so the gap is the source, not the code. Mixing the two would teach the model the gap between products. The gate was not relaxed. Future work: rebuild the whole pipeline on one CHIRPS version. |
| 7   | False-alarm audit                 | Not run, listed as future work                      |

Many experiments means one of them can look good by luck. The guard against this is that the final model is run on 2020-2024 only once.

## 9. Amendments

Any rule change made after seeing results is listed here with the date and the reason.

### 9.1 Keep rule for weighting experiments (17/11/2026, written AFTER a simulation of this experiment)

A simulation of this experiment, run before this amendment was written, found a mean PR-AUC of about 0.195 against 0.177, better in 5 of 5 folds. That fails the 17% rule in section 3. This amendment was written after seeing that result, and the 5% bar sits just below it. It is not a rule set in advance, and the result should be read as weaker evidence than one that passed the original rule. (The first version of this amendment, committed earlier the same day, wrongly said it was written before the experiment was run. This is the correction.)

Reasons for changing the rule anyway: weights change how much each training row counts and add no new features, so they cannot memorise places, and the shuffled copy test does not apply to them. A consistent gain in all 5 folds is unlikely by luck. The 17% bar was set for feature experiments, where the risk of memorising places exists. For weighting experiments only, the rule is:

1. its mean PR-AUC is at least 5% higher than the baseline (at least 0.186), and
2. it is better in all 5 of the 5 folds. If the weights did nothing, the chance of winning every fold by luck is 1 in 32.

Only one weight value is tested for the keep decision: 0.1 for positives that come only from records longer than 30 days, and 1 for everything else. Other values may be reported as extra information, but they are not used to choose the weight. (The simulation also tried other values. 0.1 was picked from those results.)

From this point the weighting is part of the current baseline. Later experiments are compared with a baseline that already uses the weights.

### 9.2 Change of unit: county level (17/11/2026, written AFTER a simulation of this experiment)

A simulation run before this amendment was written found a county PR-AUC of about 0.196 without weights and about 0.212 with them. So this amendment is not pre-registered either. The county unit is adopted because it matches the labels and the app, not because of its score.

The flood records are county-level (one county report is copied onto every sampled sub-county in that county), and the app will report per county. So the model is also tried with one row per county per dekad. This is a change of unit, not a feature experiment, and it is adopted because it matches the labels and the app, not because of its score.

- Rows: one per county per dekad. Features: the mean of the 10 rain features across the county's sampled sub-counties, the max of rfh, rfq, r1q and r3q, plus season_sin, season_cos and nat_frac_anom. Label: 1 if any sampled sub-county of the county is flooded.
- The old baseline (0.177) cannot be used here, because the rows and the base rate are different. The county baseline is the same model with no weights, and its score is written into section 8 the first time it is run.
- The plain and combined rules are recomputed at county level using the max sub-county rain, and the threshold is picked again with the section 4 rule.
- Weights at county level use the shortest record touching the county dekad, as in 9.1. They are judged against the county baseline with the 9.1 rule.
- Results are compared with lift (PR-AUC divided by the base rate), because the base rates differ.
