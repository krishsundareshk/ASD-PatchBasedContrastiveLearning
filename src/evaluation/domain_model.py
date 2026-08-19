"""
Multi-Centroid Normal Modeling and Anomaly Scoring.
Combines PCA dimensionality reduction, K-Means clustering, robust covariance estimation
(Ledoit-Wolf, OAS, Empirical, Diagonal), Mahalanobis distance, Cosine distance ensemble,
and Target False-Positive-Rate (FPR) / Percentile threshold calibration.
"""

from typing import Tuple, List, Optional
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.covariance import LedoitWolf, OAS, EmpiricalCovariance

from ..config import DomainModelConfig


def fit_covariance(x: np.ndarray, cov_type: str = "lw") -> Tuple[np.ndarray, np.ndarray]:
    """
    Fit mean location and precision (inverse covariance) matrix.

    Args:
        x: Array of shape (N, D).
        cov_type: One of 'lw' (Ledoit-Wolf), 'oas', 'empirical', or 'diag'.

    Returns:
        Tuple of (location vector of shape (D,), precision matrix of shape (D, D)).
    """
    cov_type = cov_type.lower()
    if cov_type == "lw":
        estimator = LedoitWolf().fit(x)
        return estimator.location_, estimator.precision_
    elif cov_type == "oas":
        estimator = OAS().fit(x)
        return estimator.location_, estimator.precision_
    elif cov_type == "empirical":
        estimator = EmpiricalCovariance().fit(x)
        return estimator.location_, estimator.precision_
    elif cov_type == "diag":
        mu = x.mean(axis=0)
        var = x.var(axis=0) + 1e-8
        prec = np.diag(1.0 / var)
        return mu, prec
    else:
        raise ValueError(f"Unknown covariance type '{cov_type}'. Choose 'lw', 'oas', 'empirical', or 'diag'.")


def maha_sq_to_centers(
    x: np.ndarray,
    mus: List[np.ndarray],
    precisions: List[np.ndarray],
) -> np.ndarray:
    """
    Compute the minimum squared Mahalanobis distance from each point in X to K cluster centers.

    Args:
        x: Array of shape (M, D).
        mus: List of K center vectors of shape (D,).
        precisions: List of K precision matrices of shape (D, D).

    Returns:
        Array of shape (M,) containing minimum squared Mahalanobis distances.
    """
    m = x.shape[0]
    k = len(mus)
    d_min = np.full(m, np.inf, dtype=np.float64)

    for i in range(k):
        diff = x - mus[i][None, :]
        # Efficient quadratic form: (x - mu)^T * Precision * (x - mu)
        dist_sq = np.einsum("bi,ij,bj->b", diff, precisions[i], diff, optimize=True)
        d_min = np.minimum(d_min, dist_sq)

    return d_min


def cos_dist_to_centers(z_unit: np.ndarray, centers_unit: np.ndarray) -> np.ndarray:
    """
    Compute minimum cosine distance (1 - cosine_similarity) from L2-normalized vectors to unit centers.

    Args:
        z_unit: Array of shape (M, D), L2-normalized rows.
        centers_unit: Array of shape (K, D), L2-normalized rows.

    Returns:
        Array of shape (M,) containing minimum cosine distances.
    """
    sims = z_unit @ centers_unit.T  # (M, K)
    sims = np.clip(sims, -1.0, 1.0)
    return (1.0 - sims).min(axis=1)


def zscore(scores: np.ndarray, mean: float, std: float) -> np.ndarray:
    """Standardize scores using training distribution statistics."""
    return (scores - mean) / (std + 1e-8)


def threshold_from_normals(
    train_scores: np.ndarray,
    thr_mode: str = "fpr",
    target_fpr: float = 0.05,
    perc_q: float = 99.0,
) -> float:
    """
    Calibrate anomaly decision threshold exclusively from training normal scores.
    """
    if thr_mode == "fpr":
        keep = max(0.0, min(1.0, 1.0 - float(target_fpr)))
        try:
            return float(np.quantile(train_scores, keep, method="nearest"))
        except TypeError:
            return float(np.percentile(train_scores, keep * 100.0))
    else:
        try:
            return float(np.percentile(train_scores, perc_q, method="nearest"))
        except TypeError:
            return float(np.percentile(train_scores, perc_q))


class DomainModel:
    """
    Domain-specific multi-centroid normal modeling and scoring engine.
    Learns normal operational modes from normal training embeddings and scores test samples.
    """

    def __init__(
        self,
        use_pca: bool = True,
        pca_variance: float = 0.98,
        cov_type: str = "lw",
        use_cosine: bool = True,
        w_maha: float = 0.7,
        w_cos: float = 0.3,
        k: int = 5,
        thr_mode: str = "fpr",
        target_fpr: float = 0.05,
        perc_q: float = 99.0,
        seed: int = 42,
    ):
        self.use_pca = use_pca
        self.pca_variance = pca_variance
        self.cov_type = cov_type
        self.use_cosine = use_cosine
        self.w_maha = w_maha
        self.w_cos = w_cos
        self.k = k
        self.thr_mode = thr_mode
        self.target_fpr = target_fpr
        self.perc_q = perc_q
        self.seed = seed

        # Fitted parameters
        self.pca: Optional[PCA] = None
        self.kmeans: Optional[KMeans] = None
        self.mus: List[np.ndarray] = []
        self.precs: List[np.ndarray] = []
        self.cos_centers: Optional[np.ndarray] = None

        # Normalization stats
        self.maha_mean: float = 0.0
        self.maha_std: float = 1.0
        self.cos_mean: float = 0.0
        self.cos_std: float = 1.0
        self.threshold: Optional[float] = None

    @classmethod
    def from_config(cls, config: DomainModelConfig) -> "DomainModel":
        """Instantiate DomainModel from DomainModelConfig."""
        return cls(
            use_pca=config.use_pca,
            pca_variance=config.pca_variance,
            cov_type=config.cov_type,
            use_cosine=config.use_cosine,
            w_maha=config.w_maha,
            w_cos=config.w_cos,
            k=config.k,
            thr_mode=config.thr_mode,
            target_fpr=config.target_fpr,
            perc_q=config.perc_q,
            seed=config.seed,
        )

    def fit(self, z_train_unit: np.ndarray) -> "DomainModel":
        """
        Fit normal modeling distributions on L2-normalized training normal embeddings.

        Args:
            z_train_unit: Array of shape (N, D) containing normal sample embeddings.
        """
        n_samples, feat_dim = z_train_unit.shape

        # 1) Dimensionality Reduction via PCA
        if self.use_pca and feat_dim > 1:
            self.pca = PCA(
                n_components=self.pca_variance,
                svd_solver="auto",
                random_state=self.seed
            )
            x_space = self.pca.fit_transform(z_train_unit)
        else:
            self.pca = None
            x_space = z_train_unit

        d_sub = x_space.shape[1]
        k_clusters = min(self.k, n_samples)

        # 2) Multi-Centroid Clustering via K-Means
        if k_clusters > 1:
            self.kmeans = KMeans(
                n_clusters=k_clusters,
                n_init=10,
                random_state=self.seed,
                algorithm="lloyd"
            )
            labels = self.kmeans.fit_predict(x_space)
            centers_pca = self.kmeans.cluster_centers_
        else:
            self.kmeans = None
            labels = np.zeros(n_samples, dtype=int)
            centers_pca = x_space.mean(axis=0, keepdims=True)

        def _fallback_diag(sub_x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            mu = sub_x.mean(axis=0)
            var = sub_x.var(axis=0) + 1e-6
            return mu.astype(np.float64), np.diag(1.0 / var.astype(np.float64))

        # 3) Fit Per-Cluster Covariances (Mahalanobis branch)
        self.mus = []
        self.precs = []

        for c in range(k_clusters):
            idx = np.where(labels == c)[0]

            # Merge with nearest cluster if points < dimensions
            if (idx.size < d_sub + 2) and (k_clusters > 1):
                dists = np.linalg.norm(centers_pca - centers_pca[c], axis=1)
                other_sorted = [j for j in np.argsort(dists) if j != c]
                if other_sorted:
                    nearest_cluster = other_sorted[0]
                    idx = np.where((labels == c) | (labels == nearest_cluster))[0]

            if idx.size < d_sub + 2:
                mu_c, prec_c = _fallback_diag(x_space[idx] if idx.size > 0 else x_space)
            else:
                try:
                    mu_c, prec_c = fit_covariance(x_space[idx], cov_type=self.cov_type)
                    # Check numerical condition
                    cond = np.linalg.cond(prec_c) if np.all(np.isfinite(prec_c)) else np.inf
                    if (not np.all(np.isfinite(prec_c))) or (cond > 1e8):
                        mu_c, prec_c = _fallback_diag(x_space[idx])
                except Exception:
                    mu_c, prec_c = _fallback_diag(x_space[idx] if idx.size > 0 else x_space)

            self.mus.append(mu_c.astype(np.float64))
            self.precs.append(prec_c.astype(np.float64))

        # 4) Compute Cosine Centers in Original Embedding Space
        if self.use_cosine:
            centers_cos_list = []
            for c in range(k_clusters):
                idx = np.where(labels == c)[0]
                if idx.size == 0:
                    idx = np.arange(n_samples)
                c_mean = z_train_unit[idx].mean(axis=0)
                norm_val = np.linalg.norm(c_mean)
                c_norm = c_mean / (norm_val + 1e-8) if norm_val > 0 else c_mean
                centers_cos_list.append(c_norm)
            self.cos_centers = np.stack(centers_cos_list, axis=0).astype(np.float64)

        # 5) Training Normal Score Distribution for Z-Standardization
        x_maha = x_space
        maha_scores = maha_sq_to_centers(x_maha, self.mus, self.precs)
        self.maha_mean = float(maha_scores.mean())
        self.maha_std = float(maha_scores.std() + 1e-8)

        if self.use_cosine:
            cos_scores = cos_dist_to_centers(z_train_unit, self.cos_centers)
            self.cos_mean = float(cos_scores.mean())
            self.cos_std = float(cos_scores.std() + 1e-8)

            combo = (
                self.w_maha * zscore(maha_scores, self.maha_mean, self.maha_std)
                + self.w_cos * zscore(cos_scores, self.cos_mean, self.cos_std)
            )
        else:
            combo = zscore(maha_scores, self.maha_mean, self.maha_std)

        # 6) Calibrate Decision Threshold
        self.threshold = threshold_from_normals(
            combo,
            thr_mode=self.thr_mode,
            target_fpr=self.target_fpr,
            perc_q=self.perc_q,
        )

        return self

    def score(self, z_unit: np.ndarray) -> np.ndarray:
        """
        Compute continuous anomaly scores for test embeddings (higher = more anomalous).

        Args:
            z_unit: Array of shape (M, D) containing test sample embeddings.

        Returns:
            Array of shape (M,) containing fused anomaly scores.
        """
        x_eval = self.pca.transform(z_unit) if self.pca is not None else z_unit
        maha = maha_sq_to_centers(x_eval, self.mus, self.precs)
        maha_z = zscore(maha, self.maha_mean, self.maha_std)

        if self.use_cosine and self.cos_centers is not None:
            cos = cos_dist_to_centers(z_unit, self.cos_centers)
            cos_z = zscore(cos, self.cos_mean, self.cos_std)
            return self.w_maha * maha_z + self.w_cos * cos_z

        return maha_z

    def predict(self, z_unit: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict binary labels (0=normal, 1=anomaly) and return anomaly scores.

        Args:
            z_unit: Array of shape (M, D).

        Returns:
            Tuple of (binary_predictions of shape (M,), continuous_scores of shape (M,)).
        """
        scores = self.score(z_unit)
        thr = self.threshold if self.threshold is not None else 0.0
        preds = (scores >= thr).astype(int)
        return preds, scores
