import os
import sys
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.evaluation.domain_model import (
    fit_covariance,
    maha_sq_to_centers,
    cos_dist_to_centers,
    DomainModel,
)


def test_fit_covariance():
    np.random.seed(42)
    x = np.random.randn(50, 4)

    # Test diagonal covariance
    mu, prec = fit_covariance(x, cov_type="diag")
    assert mu.shape == (4,)
    assert prec.shape == (4, 4)
    assert np.allclose(prec, np.diag(np.diag(prec)))

    # Test empirical covariance
    mu_emp, prec_emp = fit_covariance(x, cov_type="empirical")
    assert mu_emp.shape == (4,)
    assert prec_emp.shape == (4, 4)


def test_mahalanobis_distance():
    x = np.array([[0.0, 0.0], [2.0, 0.0]], dtype=np.float64)
    mus = [np.array([0.0, 0.0], dtype=np.float64)]
    precs = [np.eye(2, dtype=np.float64)]

    dists = maha_sq_to_centers(x, mus, precisions=precs)
    assert np.isclose(dists[0], 0.0)
    assert np.isclose(dists[1], 4.0)


def test_cosine_distance():
    z = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    centers = np.array([[1.0, 0.0]], dtype=np.float64)

    cos_dists = cos_dist_to_centers(z, centers)
    assert np.isclose(cos_dists[0], 0.0)
    assert np.isclose(cos_dists[1], 1.0)


def test_domain_model_fit_and_predict():
    np.random.seed(42)
    # Synthetic normal samples centered around (1, 0, 0)
    normal_train = np.random.randn(100, 8) + np.array([2.0] + [0.0] * 7)
    # Normalize to unit length
    normal_train = normal_train / np.linalg.norm(normal_train, axis=1, keepdims=True)

    model = DomainModel(k=2, cov_type="diag", use_pca=False, use_cosine=True)
    model.fit(normal_train)

    assert model.threshold is not None
    assert len(model.mus) == 2

    # Normal test point
    normal_test = np.array([[2.0] + [0.0] * 7])
    normal_test = normal_test / np.linalg.norm(normal_test, axis=1, keepdims=True)
    pred_norm, score_norm = model.predict(normal_test)

    # Anomaly test point centered far away
    anomaly_test = np.array([[-2.0] + [0.0] * 7])
    anomaly_test = anomaly_test / np.linalg.norm(anomaly_test, axis=1, keepdims=True)
    pred_anom, score_anom = model.predict(anomaly_test)

    # Score of anomaly should be strictly higher than score of normal
    assert score_anom[0] > score_norm[0]
