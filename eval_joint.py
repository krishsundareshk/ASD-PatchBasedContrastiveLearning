import os
import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_score, recall_score
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.covariance import LedoitWolf, OAS, EmpiricalCovariance

from astra_attn_patch_dataset import ASTRA_EvalRGBDataset, ALL_MACHINE_TYPES
from patch_attn_model import PatchAttentionCLModel

# ─── CONFIG ─────────────────────────────────────────────────────────
ROOT_DIR       = "training_data"
CHECKPOINT_DIR = "checkpoints_joint_pre24_resnet34_stride16_bs256"
EVAL_EPOCH     = 371
DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE     = 256
PATCH_SIZE     = 32
STRIDE         = 16

# Multi-centroid normal modeling
CLUSTER_K      = 5          # try 3–8
COV_TYPE       = "lw"       # "lw" (LedoitWolf), "oas", "empirical", "diag"
USE_PCA        = True       # PCA on train normals (per domain) before Mahalanobis
PCA_VARIANCE   = 0.98       # keep 98% variance (set to 1.0 to disable effect)
# Score ensemble (normalized z-scores on train normals)
USE_COSINE     = True
W_MAHA         = 0.7        # weight for Mahalanobis z-score
W_COS          = 0.3        # weight for cosine-distance z-score

# Thresholding per domain from TRAIN normals only
USE_TARGET_FPR = True
TARGET_FPR     = 0.05       # 5% false alarms on train normals
PERCENTILE_Q   = 99.0       # used when USE_TARGET_FPR=False

# Repro
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
# ─────────────────────────────────────────────────────────────────────


def infer_attr_dim(sd):
    key = "attn_pool.attr_bias.weight"
    return sd[key].shape[1] if key in sd else 0


def domain_subset(ds, domain, label_val=None):
    idxs = []
    for i, p in enumerate(ds.samples):
        b = os.path.basename(p)
        if domain in b:
            if label_val is None:
                idxs.append(i)
            else:
                is_normal = ("normal" in b)
                if (0 if is_normal else 1) == label_val:
                    idxs.append(i)
    return Subset(ds, idxs)


@torch.no_grad()
def extract_embeddings(loader, model):
    """Return L2-normalized pooled embeddings (exactly like your original)."""
    embs, labels = [], []
    model.eval()
    for batch in tqdm(loader, desc="Extract", leave=False):
        patches = batch["patches"].to(DEVICE)   # (B,N,3,224,224)
        attrs   = batch["attrs"].to(DEVICE)     # (B,attr_dim)
        labs    = batch["label"].numpy()        # (B,)
        B,N,C,H,W = patches.shape
        z = model(patches, B, N, attrs)         # (B, D_final) pooled + attrs
        z = F.normalize(z, dim=1).cpu().numpy() # same normalization as baseline
        embs.append(z)
        labels.extend(labs.tolist())
    return np.vstack(embs), np.array(labels)


# --------- modeling helpers ---------
def fit_cov(X, cov_type="lw"):
    if cov_type == "lw":
        est = LedoitWolf().fit(X)
        return est.location_, est.precision_
    if cov_type == "oas":
        est = OAS().fit(X)
        return est.location_, est.precision_
    if cov_type == "empirical":
        est = EmpiricalCovariance().fit(X)
        return est.location_, est.precision_
    if cov_type == "diag":
        mu = X.mean(0)
        var = X.var(0) + 1e-8
        prec = np.diag(1.0 / var)
        return mu, prec
    raise ValueError(f"Unknown COV_TYPE={cov_type}")


def maha_sq_to_centers(X, mus, precisions):
    """
    X: (M,D), mus: list of (D,), precisions: list of (D,D)
    returns: (M,) min squared Mahalanobis distance to the K centers
    """
    M = X.shape[0]
    K = len(mus)
    dmin = np.full(M, np.inf, dtype=np.float64)
    for k in range(K):
        d = X - mus[k][None, :]
        mk = np.einsum("bi,ij,bj->b", d, precisions[k], d, optimize=True)
        dmin = np.minimum(dmin, mk)
    return dmin


def cos_dist_to_centers(Z_unit, centers_unit):
    """
    Cosine distance = 1 - cos_sim; inputs must be L2-normalized row-wise.
    Z_unit: (M,D), centers_unit: (K,D)
    returns: (M,) min cosine distance to centers
    """
    sims = Z_unit @ centers_unit.T                 # (M,K)
    sims = np.clip(sims, -1.0, 1.0)
    d = 1.0 - sims                                 # (M,K)
    return d.min(axis=1)


def threshold_from_normals(train_scores, use_target_fpr=True, target_fpr=0.05, q=99.0):
    if use_target_fpr:
        keep = 1.0 - float(target_fpr)
        keep = min(max(keep, 0.0), 1.0)
        # NumPy ≥1.22 uses method= instead of interpolation=
        return float(np.quantile(train_scores, keep, method="nearest"))
    else:
        return float(np.percentile(train_scores, q))


def zscore(scores, mean, std):
    return (scores - mean) / (std + 1e-8)


# --------- per-domain model ---------
class DomainModel:
    def __init__(self, use_pca=True, pca_variance=0.98, cov_type="lw",
                 use_cosine=True, w_maha=0.7, w_cos=0.3, k=5):
        self.use_pca    = use_pca
        self.pca_var    = pca_variance
        self.cov_type   = cov_type
        self.use_cosine = use_cosine
        self.w_maha     = w_maha
        self.w_cos      = w_cos
        self.k          = k

        self.pca        = None
        self.kmeans     = None
        self.mus        = None
        self.precs      = None
        self.cos_centers= None   # unit-norm centers in original space (for cosine)

        # train-scores stats for z-normalization
        self.maha_mean  = 0.0
        self.maha_std   = 1.0
        self.cos_mean   = 0.0
        self.cos_std    = 1.0

        self.threshold  = None   # on combined score

    def fit(self, Z_train_unit, rng=SEED):
        """
        Z_train_unit: (N,D) pooled L2-normalized embeddings (original space).
        """
        # (1) PCA (optional) for the Mahalanobis branch
        if self.use_pca:
            self.pca = PCA(n_components=self.pca_var, svd_solver="full", random_state=rng)
            X = self.pca.fit_transform(Z_train_unit)
        else:
            X = Z_train_unit

        # (2) KMeans in the (possibly PCA) space
        self.kmeans = KMeans(n_clusters=self.k, n_init="auto", random_state=rng)
        labels = self.kmeans.fit_predict(X)
        centers_pca = self.kmeans.cluster_centers_

        # (3) Per-cluster covariance (in PCA space) + Mahalanobis params
        mus, precs = [], []
        for k in range(self.k):
            idx = np.where(labels == k)[0]
            # guard small clusters by borrowing neighbors
            if idx.size < X.shape[1] + 2:
                # merge with nearest center
                d = np.linalg.norm(centers_pca - centers_pca[k], axis=1)
                j = np.argsort(d)[1]  # nearest other center
                idx = np.where((labels == k) | (labels == j))[0]
            mu_k, prec_k = fit_cov(X[idx], cov_type=self.cov_type)
            mus.append(mu_k.astype(np.float64))
            precs.append(prec_k.astype(np.float64))
        self.mus, self.precs = mus, precs

        # (4) Cosine centers in original space (mean of unit vectors per cluster)
        if self.use_cosine:
            centers_cos = []
            for k in range(self.k):
                idx = np.where(labels == k)[0]
                c = Z_train_unit[idx].mean(0)
                c = c / (np.linalg.norm(c) + 1e-8)
                centers_cos.append(c)
            self.cos_centers = np.stack(centers_cos, axis=0).astype(np.float64)

        # (5) Compute train scores and z-normalizers
        # Mahalanobis on PCA space
        if self.use_pca:
            X_maha = X
        else:
            X_maha = Z_train_unit

        maha_scores = maha_sq_to_centers(X_maha, self.mus, self.precs)
        self.maha_mean, self.maha_std = float(maha_scores.mean()), float(maha_scores.std() + 1e-8)

        if self.use_cosine:
            cos_scores = cos_dist_to_centers(Z_train_unit, self.cos_centers)
            self.cos_mean, self.cos_std = float(cos_scores.mean()), float(cos_scores.std() + 1e-8)
            combo = self.w_maha * zscore(maha_scores, self.maha_mean, self.maha_std) \
                    + self.w_cos  * zscore(cos_scores,  self.cos_mean,  self.cos_std)
        else:
            combo = zscore(maha_scores, self.maha_mean, self.maha_std)

        # threshold from train normals
        self.threshold = threshold_from_normals(combo, USE_TARGET_FPR, TARGET_FPR, PERCENTILE_Q)

    def score(self, Z_unit):
        """
        Z_unit: (M,D) pooled L2-normalized embeddings (original space).
        returns:
          combo_score: (M,) anomaly score (higher = more anomalous)
        """
        # Mahalanobis in PCA space (or original if no PCA)
        if self.use_pca:
            X = self.pca.transform(Z_unit)
        else:
            X = Z_unit
        maha = maha_sq_to_centers(X, self.mus, self.precs)
        maha_z = zscore(maha, self.maha_mean, self.maha_std)

        if self.use_cosine:
            cos = cos_dist_to_centers(Z_unit, self.cos_centers)
            cos_z = zscore(cos, self.cos_mean, self.cos_std)
            return self.w_maha * maha_z + self.w_cos * cos_z
        else:
            return maha_z

    def predict(self, Z_unit):
        s = self.score(Z_unit)
        return (s >= self.threshold).astype(int), s


def main():
    # load model
    ckpt = torch.load(os.path.join(CHECKPOINT_DIR, f"epoch{EVAL_EPOCH}.pth"),
                      map_location=DEVICE, weights_only=False)
    attr_dim = infer_attr_dim(ckpt["model_state"])
    model = PatchAttentionCLModel(embed_dim=128, attr_dim=attr_dim).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    print(f"🔁 Loaded epoch{EVAL_EPOCH} (attr_dim={attr_dim})")

    # per machine
    for m in ALL_MACHINE_TYPES:
        print(f"\n=== {m} ===")

        # TRAIN NORMALS
        ds_train = ASTRA_EvalRGBDataset(ROOT_DIR, m, split="train",
                                        patch_size=PATCH_SIZE, stride=STRIDE,
                                        max_patches=None, global_attr_dim=attr_dim)
        src_train = domain_subset(ds_train, "source", label_val=0)
        tgt_train = domain_subset(ds_train, "target", label_val=0)
        if len(src_train)==0 or len(tgt_train)==0:
            print("  ⚠️  skipping (no source/target normals)")
            continue

        src_embs, _ = extract_embeddings(DataLoader(src_train, batch_size=BATCH_SIZE, num_workers=4), model)
        tgt_embs, _ = extract_embeddings(DataLoader(tgt_train, batch_size=BATCH_SIZE, num_workers=4), model)

        # Fit per-domain robust models
        src_model = DomainModel(use_pca=USE_PCA, pca_variance=PCA_VARIANCE,
                                cov_type=COV_TYPE, use_cosine=USE_COSINE,
                                w_maha=W_MAHA, w_cos=W_COS, k=CLUSTER_K)
        tgt_model = DomainModel(use_pca=USE_PCA, pca_variance=PCA_VARIANCE,
                                cov_type=COV_TYPE, use_cosine=USE_COSINE,
                                w_maha=W_MAHA, w_cos=W_COS, k=CLUSTER_K)

        src_model.fit(src_embs)
        tgt_model.fit(tgt_embs)
        print(f"  src: K={CLUSTER_K}, cov={COV_TYPE}, PCA={USE_PCA}({PCA_VARIANCE}), thr={src_model.threshold:.3f}")
        print(f"  tgt: K={CLUSTER_K}, cov={COV_TYPE}, PCA={USE_PCA}({PCA_VARIANCE}), thr={tgt_model.threshold:.3f}")

        # TEST
        ds_test = ASTRA_EvalRGBDataset(ROOT_DIR, m, split="test",
                                       patch_size=PATCH_SIZE, stride=STRIDE,
                                       max_patches=None, global_attr_dim=attr_dim)

        for domain, dmodel in [("source", src_model), ("target", tgt_model)]:
            subset = domain_subset(ds_test, domain, label_val=None)
            if len(subset) == 0:
                print(f"  {domain:6s} no test items.")
                continue

            loader = DataLoader(subset, batch_size=BATCH_SIZE, num_workers=4)
            te_embs, te_labels = extract_embeddings(loader, model)

            y_pred, scores = dmodel.predict(te_embs)

            # AUCs on the combined score
            auc   = roc_auc_score(te_labels, scores) if len(np.unique(te_labels))>1 else float("nan")
            try:
                pauc = roc_auc_score(te_labels, scores, max_fpr=0.1)
            except ValueError:
                pauc = float("nan")

            if len(np.unique(te_labels))>1:
                acc = accuracy_score(te_labels, y_pred)
                f1  = f1_score(te_labels, y_pred)
                pre = precision_score(te_labels, y_pred, zero_division=0)
                rec = recall_score(te_labels, y_pred)
                print(f"  {domain:6s} AUC={auc:.4f}  pAUC(≤0.1)={pauc:.4f}  |  Op: Acc={acc:.4f} F1={f1:.4f} P={pre:.4f} R={rec:.4f}")
            else:
                print(f"  {domain:6s} AUC={auc:.4f}  pAUC(≤0.1)={pauc:.4f}")

    print("\n✅ Done.")


if __name__=="__main__":
    main()