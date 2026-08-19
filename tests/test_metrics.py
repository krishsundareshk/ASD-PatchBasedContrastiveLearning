import os
import sys
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.evaluation.metrics import (
    compute_roc_auc,
    compute_pauc,
    compute_classification_metrics,
    harmonic_mean,
)


def test_roc_auc():
    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([0.1, 0.2, 0.8, 0.9])
    auc = compute_roc_auc(y_true, y_score)
    assert np.isclose(auc, 1.0)

    # Inverted scores
    auc_inv = compute_roc_auc(y_true, 1.0 - y_score)
    assert np.isclose(auc_inv, 0.0)

    # Single class edge case
    auc_single = compute_roc_auc(np.array([0, 0, 0]), np.array([0.1, 0.2, 0.3]))
    assert np.isnan(auc_single)


def test_pauc():
    y_true = np.array([0, 0, 0, 0, 1, 1])
    y_score = np.array([0.1, 0.2, 0.3, 0.4, 0.8, 0.9])
    pauc = compute_pauc(y_true, y_score, max_fpr=0.1)
    assert not np.isnan(pauc)


def test_classification_metrics():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 0, 1, 1])
    metrics = compute_classification_metrics(y_true, y_pred)
    assert metrics["accuracy"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0


def test_harmonic_mean():
    scores = [0.8, 0.8, 0.8]
    assert np.isclose(harmonic_mean(scores), 0.8)

    # Known harmonic mean: H(2, 6) = 2 / (1/2 + 1/6) = 2 / (4/6) = 3.0
    assert np.isclose(harmonic_mean([2.0, 6.0]), 3.0)

    # Ignore NaNs
    assert np.isclose(harmonic_mean([2.0, float("nan"), 6.0]), 3.0)

    # Empty / All NaNs
    assert np.isnan(harmonic_mean([]))
    assert np.isnan(harmonic_mean([float("nan")]))
