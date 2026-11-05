"""Compare flood prediction methods without reporting accuracy.

Run a deterministic fake-data demo with ``python evaluation.py --demo`` or
evaluate a prediction file with ``python evaluation.py --input data/predictions.csv``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


REQUIRED_COLUMNS = {"split", "y_true", "logreg_prob", "rf_prob"}
RAINFALL_COLUMNS = {"rfh", "rfq"}
RULE_THRESHOLD_MM = 50.0
RULE_THRESHOLD_PERCENT = 200.0
LOGREG_CHOSEN_THRESHOLD = 0.669
DEFAULT_PROBABILITY_THRESHOLD = 0.5


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


def prepare_split(
    predictions: pd.DataFrame,
    split: str = "test",
    rainfall_data: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Select a split and attach rainfall fields needed for both rule baselines."""
    missing = REQUIRED_COLUMNS.difference(predictions.columns)
    if missing:
        raise ValueError(f"Prediction file is missing required columns: {', '.join(sorted(missing))}")

    selected = predictions.loc[predictions["split"] == split].copy()
    if selected.empty:
        available = ", ".join(map(str, predictions["split"].dropna().unique()))
        raise ValueError(f"No rows found for split {split!r}. Available splits: {available}")

    missing_rainfall = RAINFALL_COLUMNS.difference(selected.columns)
    if missing_rainfall:
        required_keys = {"PCODE", "date"}
        if rainfall_data is None or not required_keys.union(missing_rainfall).issubset(rainfall_data.columns):
            columns = ", ".join(sorted(missing_rainfall))
            raise ValueError(
                f"{columns} absent; provide rainfall_data with PCODE, date, and those columns."
            )
        if not required_keys.issubset(selected.columns):
            raise ValueError("PCODE and date are required to join rainfall features to predictions.")
        join_columns = ["PCODE", "date", *sorted(missing_rainfall)]
        rainfall = rainfall_data[join_columns].copy()
        if rainfall.duplicated(["PCODE", "date"]).any():
            raise ValueError("Rainfall data must have one row per PCODE and date.")
        selected = selected.merge(rainfall, on=["PCODE", "date"], how="left", validate="many_to_one")

    fields = ["y_true", "logreg_prob", "rf_prob", "rfh", "rfq"]
    if selected[fields].isna().any().any():
        raise ValueError("Selected split contains missing labels, probabilities, or rainfall values.")
    if not np.isin(selected["y_true"].to_numpy(), [0, 1]).all():
        raise ValueError("y_true must contain only binary 0/1 values.")
    for probability_column in ("logreg_prob", "rf_prob"):
        probability = selected[probability_column].to_numpy(dtype=float)
        if not np.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any():
            raise ValueError(f"{probability_column} must contain finite probabilities from 0 to 1.")
    for rainfall_column in RAINFALL_COLUMNS:
        rainfall = selected[rainfall_column].to_numpy(dtype=float)
        if not np.isfinite(rainfall).all():
            raise ValueError(f"{rainfall_column} must contain finite values.")
    return selected


def compare_predictions(
    predictions: pd.DataFrame,
    split: str = "test",
    probability_threshold: float = DEFAULT_PROBABILITY_THRESHOLD,
    rainfall_data: pd.DataFrame | None = None,
    logreg_chosen_threshold: float = LOGREG_CHOSEN_THRESHOLD,
) -> pd.DataFrame:
    """Compare both models at fixed cutoffs and the plain and combined rules."""
    for threshold in (probability_threshold, logreg_chosen_threshold):
        if not 0 <= threshold <= 1:
            raise ValueError("Probability thresholds must be between 0 and 1.")

    selected = prepare_split(predictions, split, rainfall_data)
    y_true = selected["y_true"].to_numpy(dtype=int)
    logreg_score = selected["logreg_prob"].to_numpy(dtype=float)
    rf_score = selected["rf_prob"].to_numpy(dtype=float)
    rainfall = selected["rfh"].to_numpy(dtype=float)
    rainfall_percent_normal = selected["rfq"].to_numpy(dtype=float)

    plain_rule = (rainfall >= RULE_THRESHOLD_MM).astype(int)
    combined_rule = (
        (rainfall >= RULE_THRESHOLD_MM) & (rainfall_percent_normal >= RULE_THRESHOLD_PERCENT)
    ).astype(int)

    rows = [
        evaluate_model(
            f"Logistic Regression (threshold {probability_threshold:g})",
            y_true,
            (logreg_score >= probability_threshold).astype(int),
            logreg_score,
        ),
        evaluate_model(
            f"Logistic Regression (chosen threshold {logreg_chosen_threshold:g})",
            y_true,
            (logreg_score >= logreg_chosen_threshold).astype(int),
            logreg_score,
        ),
        evaluate_model(
            f"Random Forest (threshold {probability_threshold:g})",
            y_true,
            (rf_score >= probability_threshold).astype(int),
            rf_score,
        ),
        # The plain rule's ROC-AUC uses rainfall amount as a continuous risk score.
        evaluate_model("Plain 50 mm Rule", y_true, plain_rule, rainfall),
        # The combined rule supplies a fixed binary score, so its ROC-AUC describes
        # that single operating point rather than a ranking over multiple cutoffs.
        evaluate_model("Combined 50 mm + 200% Rule", y_true, combined_rule, combined_rule),
    ]
    return pd.DataFrame(rows)


def plot_precision_recall(
    predictions: pd.DataFrame,
    output: Path,
    split: str = "test",
    rainfall_data: pd.DataFrame | None = None,
    logreg_chosen_threshold: float = LOGREG_CHOSEN_THRESHOLD,
) -> None:
    """Plot model precision-recall curves and the two rules as operating points."""
    selected = prepare_split(predictions, split, rainfall_data)
    y_true = selected["y_true"].to_numpy(dtype=int)
    logreg_score = selected["logreg_prob"].to_numpy(dtype=float)
    rf_score = selected["rf_prob"].to_numpy(dtype=float)
    rainfall = selected["rfh"].to_numpy(dtype=float)
    rainfall_percent_normal = selected["rfq"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    for label, scores in (("Logistic regression", logreg_score), ("Random forest", rf_score)):
        precision, recall, _ = precision_recall_curve(y_true, scores)
        average_precision = average_precision_score(y_true, scores)
        ax.plot(recall, precision, linewidth=2, label=f"{label} (AP={average_precision:.3f})")

    rule_rows = [
        (
            "Plain 50 mm rule",
            (rainfall >= RULE_THRESHOLD_MM).astype(int),
            "o",
            "tab:green",
        ),
        (
            "Combined 50 mm + 200% rule",
            ((rainfall >= RULE_THRESHOLD_MM) & (rainfall_percent_normal >= RULE_THRESHOLD_PERCENT)).astype(int),
            "s",
            "tab:red",
        ),
    ]
    for label, rule_prediction, marker, color in rule_rows:
        precision = precision_score(y_true, rule_prediction, zero_division=0)
        recall = recall_score(y_true, rule_prediction, zero_division=0)
        ax.scatter(recall, precision, marker=marker, s=100, color=color, edgecolor="black",
                   zorder=4, label=f"{label} (P={precision:.3f}, R={recall:.3f})")

    chosen_prediction = (logreg_score >= logreg_chosen_threshold).astype(int)
    chosen_precision = precision_score(y_true, chosen_prediction, zero_division=0)
    chosen_recall = recall_score(y_true, chosen_prediction, zero_division=0)
    ax.scatter(chosen_recall, chosen_precision, marker="X", s=100, color="tab:blue",
               edgecolor="black", zorder=5,
               label=f"Logistic regression cutoff {logreg_chosen_threshold:g}")

    prevalence = float(y_true.mean())
    ax.axhline(prevalence, linestyle="--", color="gray", linewidth=1,
               label=f"Flood prevalence ({prevalence:.3f})")
    ax.set(
        title=f"Precision–recall comparison ({split} split)",
        xlabel="Recall",
        ylabel="Precision",
        xlim=(0, 1),
        ylim=(0, 1.02),
    )
    ax.grid(alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def synthetic_predictions(seed: int = 42, n: int = 500) -> pd.DataFrame:
    """Build reproducible fake predictions for a quick harness demo."""
    rng = np.random.default_rng(seed)
    y_true = rng.binomial(1, 0.05, size=n)
    # Random scores are deliberately fake; the demo checks wiring, not model quality.
    rainfall = rng.gamma(shape=1.5, scale=25.0, size=n)
    return pd.DataFrame(
        {
            "split": "demo",
            "y_true": y_true,
            "logreg_prob": rng.random(n),
            "rf_prob": rng.random(n),
            "rfh": rainfall,
            "rfq": (rainfall + 5) / 55 * 100,
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/predictions.csv"))
    parser.add_argument("--features", type=Path, default=Path("data/features.csv"),
                        help="Feature file used to obtain rfh/rfq when predictions omit them.")
    parser.add_argument("--split", default="test", help="Dataset split to evaluate (default: test).")
    parser.add_argument("--threshold", type=float, default=DEFAULT_PROBABILITY_THRESHOLD,
                        help="Default probability cutoff for the model rows (default: 0.5).")
    parser.add_argument("--logreg-threshold", type=float, default=LOGREG_CHOSEN_THRESHOLD,
                        help="Training-selected logistic regression cutoff (default: 0.669).")
    parser.add_argument("--output", type=Path, help="Optional path for the comparison CSV.")
    parser.add_argument("--plot", type=Path, help="Path for the precision-recall chart.")
    parser.add_argument("--demo", action="store_true", help="Use reproducible fake data instead of the input file.")
    args = parser.parse_args()

    if args.demo:
        predictions = synthetic_predictions()
        split = "demo"
        rainfall_data = None
        output = args.output
        plot_path = args.plot or Path("precision_recall_demo.png")
    else:
        predictions = pd.read_csv(args.input)
        split = args.split
        output = args.output
        rainfall_data = (
            pd.read_csv(args.features, usecols=["PCODE", "date", "rfh", "rfq"])
            if not RAINFALL_COLUMNS.issubset(predictions.columns)
            else None
        )
        plot_path = args.plot or Path("precision_recall_comparison.png")

    comparison = compare_predictions(
        predictions, split, args.threshold, rainfall_data, args.logreg_threshold
    )
    print(comparison.round(3).to_string(index=False))
    print("\nConfusion matrices (rows = actual [no flood, flood]; columns = predicted [no flood, flood]):")
    for row in comparison.itertuples(index=False):
        print(f"{row.Model}: [[{row.TN}, {row.FP}], [{row.FN}, {row.TP}]]")
    plot_precision_recall(
        predictions,
        plot_path,
        split,
        rainfall_data=rainfall_data,
        logreg_chosen_threshold=args.logreg_threshold,
    )
    print(f"Precision-recall chart saved to {plot_path}")
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        comparison.to_csv(output, index=False)
        print(f"Comparison saved to {output}")


if __name__ == "__main__":
    main()
