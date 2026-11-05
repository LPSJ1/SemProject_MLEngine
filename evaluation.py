"""Compare flood prediction methods without reporting accuracy.

Run a deterministic fake-data demo with ``python evaluation.py --demo`` or
evaluate a prediction file with ``python evaluation.py --input data/predictions.csv``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score


REQUIRED_COLUMNS = {"split", "y_true", "logreg_prob", "rf_prob"}
RULE_THRESHOLD_MM = 50.0


def evaluate_model(
    name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray | None,
) -> dict[str, float | int | str]:
    """Return requested metrics and confusion-matrix counts for one method."""
    tn, fp, fn, tp = _confusion_counts(y_true, y_pred)
    auc = (
        float(roc_auc_score(y_true, y_score))
        if y_score is not None and np.unique(y_true).size == 2
        else np.nan
    )
    return {
        "Model": name,
        "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "Recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "F1": float(f1_score(y_true, y_pred, zero_division=0)),
        "ROC-AUC": auc,
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
    }


def _confusion_counts(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[int, int, int, int]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.ndim != 1 or y_pred.ndim != 1 or len(y_true) != len(y_pred):
        raise ValueError("Labels and predictions must be aligned one-dimensional arrays.")
    if not np.isin(y_true, [0, 1]).all() or not np.isin(y_pred, [0, 1]).all():
        raise ValueError("Labels and predictions must contain only binary 0/1 values.")
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    return tn, fp, fn, tp


def compare_predictions(
    predictions: pd.DataFrame,
    split: str = "test",
    probability_threshold: float = 0.5,
    rainfall_data: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Evaluate the two probability models and the explicit 50 mm rule."""
    missing = REQUIRED_COLUMNS.difference(predictions.columns)
    if missing:
        raise ValueError(f"Prediction file is missing required columns: {', '.join(sorted(missing))}")
    if not 0 <= probability_threshold <= 1:
        raise ValueError("The probability threshold must be between 0 and 1.")

    selected = predictions.loc[predictions["split"] == split].copy()
    if selected.empty:
        available = ", ".join(map(str, predictions["split"].dropna().unique()))
        raise ValueError(f"No rows found for split {split!r}. Available splits: {available}")
    if "rfh" not in selected.columns:
        if rainfall_data is None or not {"PCODE", "date", "rfh"}.issubset(rainfall_data.columns):
            raise ValueError("rfh is absent; provide rainfall_data with PCODE, date, and rfh columns.")
        if not {"PCODE", "date"}.issubset(selected.columns):
            raise ValueError("PCODE and date are required to join rainfall values to predictions.")
        rainfall = rainfall_data[["PCODE", "date", "rfh"]].copy()
        if rainfall.duplicated(["PCODE", "date"]).any():
            raise ValueError("Rainfall data must have one row per PCODE and date.")
        selected = selected.merge(rainfall, on=["PCODE", "date"], how="left", validate="many_to_one")
    fields = ["y_true", "logreg_prob", "rf_prob", "rfh"]
    if selected[fields].isna().any().any():
        raise ValueError("Selected split contains missing labels, probabilities, or rainfall values.")

    y_true = selected["y_true"].to_numpy()
    if not np.isin(y_true, [0, 1]).all():
        raise ValueError("y_true must contain only binary 0/1 values.")
    for probability_column in ("logreg_prob", "rf_prob"):
        probability = selected[probability_column].to_numpy(dtype=float)
        if not np.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any():
            raise ValueError(f"{probability_column} must contain finite probabilities from 0 to 1.")
    rainfall = selected["rfh"].to_numpy(dtype=float)
    if not np.isfinite(rainfall).all():
        raise ValueError("rfh must contain finite rainfall values.")

    rule_pred = (rainfall >= RULE_THRESHOLD_MM).astype(int)
    # If the input carries a rule column, make sure it is the agreed plain threshold rule.
    if "rule_plain" in selected.columns:
        recorded_rule = selected["rule_plain"].to_numpy()
        if not np.isin(recorded_rule, [0, 1]).all() or not np.array_equal(recorded_rule, rule_pred):
            raise ValueError("rule_plain does not match the plain 50 mm rule (rfh >= 50 mm).")

    rows = [
        evaluate_model(
            "Logistic Regression",
            y_true,
            (selected["logreg_prob"].to_numpy() >= probability_threshold).astype(int),
            selected["logreg_prob"].to_numpy(dtype=float),
        ),
        evaluate_model(
            "Random Forest",
            y_true,
            (selected["rf_prob"].to_numpy() >= probability_threshold).astype(int),
            selected["rf_prob"].to_numpy(dtype=float),
        ),
        evaluate_model("50 mm Rule", y_true, rule_pred, rainfall),
    ]
    return pd.DataFrame(rows)


def synthetic_predictions(seed: int = 42, n: int = 500) -> pd.DataFrame:
    """Build reproducible fake predictions for a quick harness demo."""
    rng = np.random.default_rng(seed)
    y_true = rng.binomial(1, 0.05, size=n)
    # Random scores are deliberately fake; the demo checks wiring, not model quality.
    return pd.DataFrame(
        {
            "split": "demo",
            "y_true": y_true,
            "logreg_prob": rng.random(n),
            "rf_prob": rng.random(n),
            "rfh": rng.gamma(shape=1.5, scale=25.0, size=n),
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/predictions.csv"))
    parser.add_argument("--features", type=Path, default=Path("data/features.csv"),
                        help="Feature file used to obtain rfh when predictions omit it.")
    parser.add_argument("--split", default="test", help="Dataset split to evaluate (default: test).")
    parser.add_argument("--threshold", type=float, default=0.5, help="Probability cutoff (default: 0.5).")
    parser.add_argument("--output", type=Path, help="Optional path for the comparison CSV.")
    parser.add_argument("--demo", action="store_true", help="Use reproducible fake data instead of the input file.")
    args = parser.parse_args()

    if args.demo:
        predictions = synthetic_predictions()
        split = "demo"
        output = args.output
    else:
        predictions = pd.read_csv(args.input)
        split = args.split
        output = args.output
        rainfall_data = (
            pd.read_csv(args.features, usecols=["PCODE", "date", "rfh"])
            if "rfh" not in predictions.columns
            else None
        )

    if args.demo:
        rainfall_data = None
    comparison = compare_predictions(predictions, split, args.threshold, rainfall_data)
    print(comparison.round(3).to_string(index=False))
    print("\nConfusion matrices (rows = actual [no flood, flood]; columns = predicted [no flood, flood]):")
    for row in comparison.itertuples(index=False):
        print(f"{row.Model}: [[{row.TN}, {row.FP}], [{row.FN}, {row.TP}]]")
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        comparison.to_csv(output, index=False)
        print(f"Comparison saved to {output}")


if __name__ == "__main__":
    main()
