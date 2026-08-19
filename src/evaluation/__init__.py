"""Evaluation, domain modeling, anomaly scoring, and metric modules."""

from .domain_model import DomainModel, fit_covariance, maha_sq_to_centers, cos_dist_to_centers
from .metrics import (
    compute_roc_auc,
    compute_pauc,
    compute_classification_metrics,
    harmonic_mean,
)
from .evaluator import extract_embeddings, evaluate_machine_domain

__all__ = [
    "DomainModel",
    "fit_covariance",
    "maha_sq_to_centers",
    "cos_dist_to_centers",
    "compute_roc_auc",
    "compute_pauc",
    "compute_classification_metrics",
    "harmonic_mean",
    "extract_embeddings",
    "evaluate_machine_domain",
]
