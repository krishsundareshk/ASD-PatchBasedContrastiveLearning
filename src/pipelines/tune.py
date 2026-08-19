"""
Hyperparameter Grid Search & Tuning Pipeline.
Sweeps domain modeling configurations (Clusters K, Covariance Types, PCA dimensions,
Cosine ensemble weights, Thresholding policies) to optimize AUC/pAUC without retraining.
"""

import os
import argparse
import itertools
from typing import Dict, Any, Generator
import numpy as np
import torch
from torch.utils.data import DataLoader

from ..config import DatasetConfig, GridSearchConfig
from ..models.patch_model import PatchAttentionCLModel
from ..evaluation.domain_model import DomainModel
from ..evaluation.metrics import compute_roc_auc, compute_pauc, compute_classification_metrics
from ..evaluation.evaluator import extract_embeddings
from ..data.dataset import ASTRAEvalDataset, domain_subset
from ..utils.common import set_seed, get_device, setup_logger
from ..utils.checkpoint import infer_attr_dim


def generate_grid_configs(grid_cfg: GridSearchConfig) -> Generator[Dict[str, Any], None, None]:
    """Generate parameter dictionary combinations from GridSearchConfig."""
    for k, cov, (use_pca, pca_var), (use_cos, w_m, w_c), (t_mode, t_fpr, p_q) in itertools.product(
        grid_cfg.ks,
        grid_cfg.cov_types,
        grid_cfg.pca_options,
        grid_cfg.cosine_weights,
        grid_cfg.thr_options,
    ):
        yield {
            "k": k,
            "cov_type": cov,
            "use_pca": use_pca,
            "pca_variance": pca_var,
            "use_cosine": use_cos,
            "w_maha": w_m,
            "w_cos": w_c,
            "thr_mode": t_mode,
            "target_fpr": t_fpr if t_fpr is not None else 0.05,
            "perc_q": p_q if p_q is not None else 99.0,
        }


def evaluate_single_config(
    train_embs: np.ndarray,
    test_embs: np.ndarray,
    test_labels: np.ndarray,
    config_dict: Dict[str, Any],
) -> Dict[str, Any]:
    """Fit a DomainModel with a single config and compute evaluation metrics."""
    dm = DomainModel(**config_dict)
    dm.fit(train_embs)
    y_pred, scores = dm.predict(test_embs)

    auc = compute_roc_auc(test_labels, scores)
    pauc = compute_pauc(test_labels, scores, max_fpr=0.1)
    cls_metrics = compute_classification_metrics(test_labels, y_pred)

    return {
        "auc": auc,
        "pauc": pauc,
        "threshold": dm.threshold,
        "config": config_dict,
        **cls_metrics,
    }


def run_tuning(
    checkpoint_path: str,
    root_dir: str = "training_data",
    batch_size: int = 64,
    patch_size: int = 32,
    stride: int = 16,
    embed_dim: int = 128,
    num_workers: int = 4,
    top_n: int = 5,
    seed: int = 42,
    device_str: str = "auto",
) -> None:
    """
    Execute full grid search hyperparameter tuning for domain scoring models.
    """
    logger = setup_logger("ASD.Tune")
    set_seed(seed)
    device = get_device(device_str)

    if not os.path.isfile(checkpoint_path):
        logger.error(f"Checkpoint file not found: {checkpoint_path}")
        return

    logger.info(f"Loading checkpoint for tuning: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("model_state", checkpoint)
    attr_dim = infer_attr_dim(state_dict)

    model = PatchAttentionCLModel(
        embed_dim=embed_dim,
        attr_dim=attr_dim,
        backbone_name="resnet34",
        pretrained=False,
    ).to(device)
    model.load_state_dict(state_dict)

    dataset_cfg = DatasetConfig(root_dir=root_dir)
    grid_cfg = GridSearchConfig()

    for machine in dataset_cfg.machine_types:
        logger.info(f"\n{'='*30} Tuning Machine: {machine} {'='*30}")

        # 1) Load datasets
        train_ds = ASTRAEvalDataset(
            root_dir=root_dir,
            machine_type=machine,
            split="train",
            patch_size=patch_size,
            stride=stride,
            global_attr_dim=attr_dim,
        )
        test_ds = ASTRAEvalDataset(
            root_dir=root_dir,
            machine_type=machine,
            split="test",
            patch_size=patch_size,
            stride=stride,
            global_attr_dim=attr_dim,
        )

        src_tr_sub = domain_subset(train_ds, "source", label_val=0)
        tgt_tr_sub = domain_subset(train_ds, "target", label_val=0)

        if len(src_tr_sub) == 0 or len(tgt_tr_sub) == 0:
            logger.warning(f"Skipping {machine}: missing source or target normal train samples.")
            continue

        # 2) Extract embeddings once per machine
        src_tr_embs, _, _ = extract_embeddings(
            DataLoader(src_tr_sub, batch_size=batch_size, num_workers=num_workers, shuffle=False),
            model, device
        )
        tgt_tr_embs, _, _ = extract_embeddings(
            DataLoader(tgt_tr_sub, batch_size=batch_size, num_workers=num_workers, shuffle=False),
            model, device
        )

        src_te_sub = domain_subset(test_ds, "source", label_val=None)
        tgt_te_sub = domain_subset(test_ds, "target", label_val=None)

        src_te_embs, src_te_labels, _ = extract_embeddings(
            DataLoader(src_te_sub, batch_size=batch_size, num_workers=num_workers, shuffle=False),
            model, device
        )
        tgt_te_embs, tgt_te_labels, _ = extract_embeddings(
            DataLoader(tgt_te_sub, batch_size=batch_size, num_workers=num_workers, shuffle=False),
            model, device
        )

        # 3) Sweep configurations for source and target domains
        for domain, tr_embs, te_embs, te_labels in [
            ("source", src_tr_embs, src_te_embs, src_te_labels),
            ("target", tgt_tr_embs, tgt_te_embs, tgt_te_labels),
        ]:
            if len(te_embs) == 0:
                logger.info(f"[{domain}] No test items.")
                continue

            results = [
                evaluate_single_config(tr_embs, te_embs, te_labels, cfg)
                for cfg in generate_grid_configs(grid_cfg)
            ]

            # Sort descending by AUC, then pAUC
            results.sort(
                key=lambda r: (
                    -1.0 if np.isnan(r["auc"]) else r["auc"],
                    -1.0 if np.isnan(r["pauc"]) else r["pauc"],
                ),
                reverse=True
            )

            best = results[0]
            logger.info(f"[{domain.upper()}] Best Config: AUC={best['auc']:.4f} | pAUC={best['pauc']:.4f} | F1={best.get('f1', 0):.4f}")
            logger.info(f"  Configuration: {best['config']}")

            logger.info(f"  Top {min(top_n, len(results))} configurations:")
            for i, r in enumerate(results[:top_n]):
                logger.info(
                    f"    #{i+1}: AUC={r['auc']:.4f} | pAUC={r['pauc']:.4f} | K={r['config']['k']} | "
                    f"cov={r['config']['cov_type']} | PCA={r['config']['use_pca']}({r['config']['pca_variance']}) | "
                    f"w_maha={r['config']['w_maha']}"
                )


def parse_args():
    parser = argparse.ArgumentParser(description="Tune Normal Domain Modeling Hyperparameters.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint .pth file")
    parser.add_argument("--root-dir", type=str, default="training_data", help="Root dataset directory")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--patch-size", type=int, default=32, help="Patch size")
    parser.add_argument("--stride", type=int, default=16, help="Patch stride")
    parser.add_argument("--embed-dim", type=int, default=128, help="Embedding dimension")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--top-n", type=int, default=5, help="Number of top configurations to show")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="auto", help="Device ('auto', 'cuda', 'cpu')")
    return parser.parse_args()


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()

    args = parse_args()
    run_tuning(
        checkpoint_path=args.checkpoint,
        root_dir=args.root_dir,
        batch_size=args.batch_size,
        patch_size=args.patch_size,
        stride=args.stride,
        embed_dim=args.embed_dim,
        num_workers=args.num_workers,
        top_n=args.top_n,
        seed=args.seed,
        device_str=args.device,
    )
