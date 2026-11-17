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

1. its mean PR-AUC is at least 17% higher than 0.1777, meaning at least 0.207 and
2. it is better in at least 4 of the 5 folds.

I chose 17% because my fold scores vary a lot( 0.08 to 0.31), so a small gain could be luck. 17% (0.03) is clearly larger than the noise but small enough to be a real improvement.

Anything else is rejected. We still write down the result.
Features must also beat a shuffled copy of themselves, so we know they are not just memorising places.
If it doesn't, it's rejected even if the PR-AUC gain passes.

## 4. How we choose the alert threshold

Using out-of-fold predictions on the training years, we pick the threshold with the highest recall whose precision is at least the combined rule's precision. The test years are never used to pick it.

## 5. Labels

Every experiment is scored against the original labels. Floods found in the false-alarm audit may be added to training labels only. They never decide whether a change is kept, because the audit picks cases the model is already confident about.

## 6. Event metric

- An event is a run of consecutive flooded dekads in one sub-county.
- An event is detected if the model alerts within 1 dekad of it.
- A false alarm episode is an alert that is not within 1 dekad of any event.
  We report both, because alerting everywhere would detect every event.

I allow 1 dekad (10 days) either side because the flood records have approximate dates, so a correct alert could land in the next or previous dekad. More than that would let unrelated alerts count as hits.

## 7. Test years

The final model is run on 2020-2024 once, at the end. The test years already gave us ideas: the baseline error analysis found that Tana River and Garissa floods were missed, which led to the upstream experiment.
I state this openly.

## 8. Experiments

| #   | Experiment                        | Result                                              |
| --- | --------------------------------- | --------------------------------------------------- |
| 1   | Basin wetness feature             | Rejected: 0.178 vs 0.177                            |
| 2   | Strict upstream rain              | Rejected: no gain over baseline or shuffled control |
| 3   | False-alarm audit                 |                                                     |
| 4   | Weights for weak reporting years  |                                                     |
| 5   | Date-precision weights            |                                                     |
| 6   | Persistence, neighbour rain       |                                                     |
| 7   | Interactions, boosting, averaging |                                                     |
