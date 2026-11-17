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

I allow 1 dekad (10 days) either side because the flood records have approximate dates, so a correct alert could land in the next or previous dekad. More than that would let unrelated alerts count as hits.

## 7. Test years

The final model is run on 2020-2024 once, at the end. The test years already gave us ideas: the baseline error analysis found that Tana River and Garissa floods were missed, which led to the upstream experiment.
I state this openly.

## 8. Experiments

| #   | Experiment                        | Result                                              |
| --- | --------------------------------- | --------------------------------------------------- |
| 1   | Basin wetness feature             | Rejected: 0.178 vs 0.177 (run before this protocol was written) |
| 2   | Strict upstream rain              | Rejected: no gain over baseline or shuffled control (run before this protocol was written) |
| 3   | Long-record weights               | Mean 0.195 vs 0.177 (+10%), folds 0.12, 0.20, 0.23, 0.10, 0.33, better in 5/5. Kept under 9.1; fails the 17% rule in section 3. Recall 0.551 vs 0.514 at precision 0.170. |
| 4   | County-level target (new unit)    | Pending                                             |
| 5   | MJO features                      | Pending                                             |
| 6   | Daily rain intensity              | Pending                                             |
| 7   | False-alarm audit                 | Not run, listed as future work                      |

Many experiments means one of them can look good by luck. The guard against this is that the final model is run on 2020-2024 only once.

## 9. Amendments

Any rule change made after seeing results is listed here with the date and the reason.

### 9.1 Keep rule for weighting experiments (17/11/2026, written before they are run)

Weights change how much each training row counts. They add no new features, so they cannot memorise places, and the shuffled copy test does not apply to them. A 17% gain is too strict for that kind of change, so for weighting experiments only the rule is:

1. its mean PR-AUC is at least 5% higher than the baseline (at least 0.186), and
2. it is better in all 5 of the 5 folds. If the weights did nothing, the chance of winning every fold by luck is 1 in 32.

Only one weight value is tested for the keep decision: 0.1 for positives that come only from records longer than 30 days, and 1 for everything else. Other values may be reported as extra information, but they are not used to choose the weight.

The gain from this experiment was estimated before it was run, not measured. I am saying so here so the 5% rule is not mistaken for one fitted to a result.
