# Model comparison approach

The evaluation compares logistic regression, a random forest, and the plain
50 mm rainfall rule on the same held-out test periods. The two models' predicted
probabilities are converted to flood/no-flood predictions at a stated
probability threshold (0.5 by default). The rule predicts a flood when `rfh` is
at least 50 mm. Precision, recall, F1, ROC-AUC, and each confusion matrix are
reported for every method. The comparison excludes accuracy because floods are
rare and a model that never predicts a flood could still appear accurate.

Recall is the primary decision metric: a missed flood can strand a responder,
while a false alarm mainly causes a detour. Precision and F1 remain useful
supporting measures, and ROC-AUC assesses how well the probability models rank
flood periods across thresholds. For the fixed 50 mm rule, ROC-AUC uses the
underlying rainfall value (`rfh`) as its score. Comparisons use the same test
rows and labels for all three methods; the training split is not used to claim
generalisation performance.

The script includes seeded fake data to exercise the evaluation pipeline before
model outputs are ready. It can then read the real prediction file without
changing the metric code:

```powershell
python evaluation.py --demo
python evaluation.py --input data/predictions.csv --features data/features.csv --split test
```

When `data/predictions.csv` does not contain rainfall, the script joins `rfh`
from `data/features.csv` using `PCODE` and `date`. This matches the output
format of the project training scripts. The comparison's 50 mm baseline is
computed directly as `rfh >= 50`; it remains separate from any combined 50 mm
and rainfall-quantile rule saved by the training pipeline.

The fake-data output is for checking the harness only. It is not evidence about
the relative performance of the models. The real comparison should be described
with the evaluation setup, recall-first rationale, and metrics above; report
numerical results only when the brief asks for them.
