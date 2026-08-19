"""
Evaluation metrics for Anomaly Sound Detection according to DCASE specifications.
Calculates ROC-AUC, Partial-AUC (pAUC with max_fpr=0.1), binary classification metrics
(Accuracy, Precision, Recall, F1), and Harmonic Means.
"""

from typing import Dict, List, Union
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score


def compute_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute Standard Area Under ROC Curve (ROC-AUC). Returns NaN if single class present."""
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def compute_pauc(y_true: np.ndarray, y_score: np.ndarray, max_fpr: float = 0.1) -> float:
    """
    Compute Partial Area Under ROC Curve (pAUC) bounded by max_fpr (official DCASE metric).
    Returns NaN if single class or invalid range.
    """
    if len(np.unique(y_true)) < 2:
        return float("nan")
    try:
        return float(roc_auc_score(y_true, y_score, max_fpr=max_fpr))
    except (ValueError, TypeError):
        return float("nan")


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """Compute Accuracy, Precision, Recall, and F1-score for binary anomaly predictions."""
    if len(np.unique(y_true)) < 2:
        return {
            "accuracy": float("nan"),
            "precision": float("nan"),
            "recall": float("nan"),
            "f1": float("nan"),
        }

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def harmonic_mean(values: Union[List[float], np.ndarray]) -> float:
    """
    Compute harmonic mean of a list of metric scores, ignoring NaNs.
    Formula: n / sum(1 / x)
    """
    valid = [v for v in values if v is not None and not np.isnan(v) and v > 0]
    if not valid:
        return float("nan")
    return float(len(valid) / np.sum(1.0 / np.array(valid, dtype=np.float64)))
