"""
model_calibration.py — calibrate match-outcome probabilities
============================================================

Fits isotonic calibration on the validation set so reported percentages
better match real frequencies (e.g. 70% picks win ~70% of the time).
"""

from __future__ import annotations

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import accuracy_score, log_loss


def calibrate_model(model, X_val, y_val):
    """Wrap a pre-trained classifier with isotonic calibration."""
    calibrated = CalibratedClassifierCV(FrozenEstimator(model), method="isotonic")
    calibrated.fit(X_val, y_val)
    return calibrated


def compute_draw_boost(y_true, proba: np.ndarray, cap: float = 2.5) -> float:
    """Ratio of actual draw frequency to mean predicted draw probability.

    Returns a multiplier (>=1.0, capped) to correct the model's chronic
    under-calling of draws. 1.0 means no adjustment needed.
    """
    y_true = np.asarray(y_true)
    actual_draw = float((y_true == 1).mean())
    pred_draw = float(proba[:, 1].mean())
    if pred_draw <= 1e-6:
        return 1.0
    boost = actual_draw / pred_draw
    return float(min(max(boost, 1.0), cap))


def apply_draw_boost(
    p_home: float, p_draw: float, p_away: float, boost: float
) -> tuple[float, float, float]:
    p_draw = p_draw * boost
    total = p_home + p_draw + p_away
    if total <= 0:
        return p_home, p_draw, p_away
    return p_home / total, p_draw / total, p_away / total


def evaluate_probabilities(name: str, y_true, proba: np.ndarray) -> dict[str, float]:
    pred = proba.argmax(axis=1)
    base = np.tile(np.bincount(y_true, minlength=3) / len(y_true), (len(y_true), 1))
    metrics = {
        "accuracy": float(accuracy_score(y_true, pred)),
        "log_loss": float(log_loss(y_true, proba, labels=[0, 1, 2])),
        "baseline_log_loss": float(log_loss(y_true, base, labels=[0, 1, 2])),
    }
    print(f"  {name} accuracy : {metrics['accuracy']:.3f}")
    print(
        f"  {name} log-loss  : {metrics['log_loss']:.3f}  "
        f"(baseline {metrics['baseline_log_loss']:.3f})"
    )
    return metrics


def reliability_bins(y_true, proba: np.ndarray, n_bins: int = 5) -> list[dict[str, float]]:
    """Bucket predicted win probabilities (max class) vs observed hit rate."""
    conf = proba.max(axis=1)
    pred = proba.argmax(axis=1)
    hits = (pred == y_true).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    rows: list[dict[str, float]] = []
    for low, high in zip(bins[:-1], bins[1:]):
        mask = (conf >= low) & (conf < high if high < 1.0 else conf <= high)
        if not mask.any():
            continue
        rows.append(
            {
                "bin_low": float(low),
                "bin_high": float(high),
                "n": int(mask.sum()),
                "avg_conf": float(conf[mask].mean()),
                "hit_rate": float(hits[mask].mean()),
            }
        )
    return rows


def print_reliability(title: str, y_true, proba: np.ndarray) -> None:
    rows = reliability_bins(y_true, proba)
    if not rows:
        return
    print(f"  {title} reliability (confidence vs hit rate):")
    for row in rows:
        print(
            f"    {row['bin_low']*100:>4.0f}-{row['bin_high']*100:>4.0f}%  "
            f"n={row['n']:>4}  conf={row['avg_conf']*100:>5.1f}%  hit={row['hit_rate']*100:>5.1f}%"
        )
