"""
Embedding extraction and machine evaluation pipeline.
Evaluates domain shift performance (Source Domain vs Target Domain) across all machine types.
"""

from typing import Tuple, Dict, Any, List, Optional
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .domain_model import DomainModel
from .metrics import compute_roc_auc, compute_pauc, compute_classification_metrics
from ..data.dataset import ASTRAEvalDataset, domain_subset


@torch.no_grad()
def extract_embeddings(
    loader: DataLoader,
    model: torch.nn.Module,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Extract L2-normalized pooled embeddings and labels from a DataLoader.

    Args:
        loader: PyTorch DataLoader wrapping an ASTRAEvalDataset.
        model: PatchAttentionCLModel instance.
        device: torch.device.

    Returns:
        Tuple of (embeddings of shape (M, D), labels of shape (M,), filenames list).
    """
    from tqdm import tqdm

    model.eval()
    embeddings_list, labels_list, filenames_list = [], [], []

    for batch in tqdm(loader, desc="Extracting Embeddings", leave=False):
        patches = batch["patches"].to(device)  # (B, N, 3, H, W)
        attrs = batch["attrs"].to(device)      # (B, attr_dim)
        labels = batch["label"].cpu().numpy()   # (B,)
        filenames = batch["filename"]

        b, n, c, h, w = patches.shape
        z = model(patches, b, n, attrs=attrs)
        z = F.normalize(z, dim=1).cpu().numpy()

        embeddings_list.append(z)
        labels_list.extend(labels.tolist())
        filenames_list.extend(filenames)

    if len(embeddings_list) == 0:
        return np.empty((0, model.final_dim)), np.empty((0,)), []

    return np.vstack(embeddings_list), np.array(labels_list), filenames_list


def evaluate_machine_domain(
    machine_type: str,
    root_dir: str,
    model: torch.nn.Module,
    device: torch.device,
    domain_model_builder: Optional[DomainModel] = None,
    batch_size: int = 64,
    patch_size: int = 32,
    stride: int = 16,
    attr_dim: int = 0,
    num_workers: int = 4,
) -> Dict[str, Any]:
    """
    Fit domain models on source & target normal training data and evaluate on test data.

    Returns dictionary containing metrics for 'source' and 'target' domains.
    """
    results: Dict[str, Any] = {"machine": machine_type, "source": {}, "target": {}}

    # 1) Load Training Normal Dataset
    train_ds = ASTRAEvalDataset(
        root_dir=root_dir,
        machine_type=machine_type,
        split="train",
        patch_size=patch_size,
        stride=stride,
        max_patches=None,
        global_attr_dim=attr_dim,
    )

    src_train_sub = domain_subset(train_ds, "source", label_val=0)
    tgt_train_sub = domain_subset(train_ds, "target", label_val=0)

    if len(src_train_sub) == 0 or len(tgt_train_sub) == 0:
        results["error"] = "Insufficient normal training samples for source/target."
        return results

    src_tr_loader = DataLoader(src_train_sub, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    tgt_tr_loader = DataLoader(tgt_train_sub, batch_size=batch_size, num_workers=num_workers, shuffle=False)

    src_tr_embs, _, _ = extract_embeddings(src_tr_loader, model, device)
    tgt_tr_embs, _, _ = extract_embeddings(tgt_tr_loader, model, device)

    # 2) Fit Domain Models
    if domain_model_builder is None:
        src_model = DomainModel()
        tgt_model = DomainModel()
    else:
        # Create copies with same parameters
        src_model = DomainModel(
            use_pca=domain_model_builder.use_pca,
            pca_variance=domain_model_builder.pca_variance,
            cov_type=domain_model_builder.cov_type,
            use_cosine=domain_model_builder.use_cosine,
            w_maha=domain_model_builder.w_maha,
            w_cos=domain_model_builder.w_cos,
            k=domain_model_builder.k,
            thr_mode=domain_model_builder.thr_mode,
            target_fpr=domain_model_builder.target_fpr,
            perc_q=domain_model_builder.perc_q,
            seed=domain_model_builder.seed,
        )
        tgt_model = DomainModel(
            use_pca=domain_model_builder.use_pca,
            pca_variance=domain_model_builder.pca_variance,
            cov_type=domain_model_builder.cov_type,
            use_cosine=domain_model_builder.use_cosine,
            w_maha=domain_model_builder.w_maha,
            w_cos=domain_model_builder.w_cos,
            k=domain_model_builder.k,
            thr_mode=domain_model_builder.thr_mode,
            target_fpr=domain_model_builder.target_fpr,
            perc_q=domain_model_builder.perc_q,
            seed=domain_model_builder.seed,
        )

    src_model.fit(src_tr_embs)
    tgt_model.fit(tgt_tr_embs)

    # 3) Load Test Dataset
    test_ds = ASTRAEvalDataset(
        root_dir=root_dir,
        machine_type=machine_type,
        split="test",
        patch_size=patch_size,
        stride=stride,
        max_patches=None,
        global_attr_dim=attr_dim,
    )

    for domain, d_model in [("source", src_model), ("target", tgt_model)]:
        sub = domain_subset(test_ds, domain, label_val=None)
        if len(sub) == 0:
            results[domain] = {"error": f"No test items for {domain}"}
            continue

        loader = DataLoader(sub, batch_size=batch_size, num_workers=num_workers, shuffle=False)
        te_embs, te_labels, _ = extract_embeddings(loader, model, device)

        y_pred, scores = d_model.predict(te_embs)

        auc = compute_roc_auc(te_labels, scores)
        pauc = compute_pauc(te_labels, scores, max_fpr=0.1)
        cls_metrics = compute_classification_metrics(te_labels, y_pred)

        results[domain] = {
            "auc": auc,
            "pauc": pauc,
            "threshold": float(d_model.threshold) if d_model.threshold is not None else 0.0,
            **cls_metrics,
        }

    return results
