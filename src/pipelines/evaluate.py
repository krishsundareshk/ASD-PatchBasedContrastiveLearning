"""
Evaluation Pipeline.
Evaluates a trained PatchAttentionCLModel checkpoint across all machine types and domains,
reporting ROC-AUC, pAUC (<=0.1), Precision, Recall, F1, and Harmonic Means.
"""

import os
import argparse
from typing import Optional
import torch
import numpy as np

from ..config import DatasetConfig, DomainModelConfig
from ..models.patch_model import PatchAttentionCLModel
from ..evaluation.domain_model import DomainModel
from ..evaluation.evaluator import evaluate_machine_domain
from ..evaluation.metrics import harmonic_mean
from ..utils.common import set_seed, get_device, setup_logger
from ..utils.checkpoint import infer_attr_dim


def run_evaluation(
    checkpoint_path: str,
    root_dir: str = "training_data",
    batch_size: int = 64,
    patch_size: int = 32,
    stride: int = 16,
    embed_dim: int = 128,
    num_workers: int = 4,
    seed: int = 42,
    device_str: str = "auto",
    domain_config: Optional[DomainModelConfig] = None,
) -> None:
    """
    Evaluate a checkpoint on all machine types.
    """
    logger = setup_logger("ASD.Eval")
    set_seed(seed)
    device = get_device(device_str)

    if not os.path.isfile(checkpoint_path):
        logger.error(f"Checkpoint file not found: {checkpoint_path}")
        return

    logger.info(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("model_state", checkpoint)

    attr_dim = infer_attr_dim(state_dict)
    logger.info(f"Inferred attribute dimension from checkpoint: {attr_dim}")

    model = PatchAttentionCLModel(
        embed_dim=embed_dim,
        attr_dim=attr_dim,
        backbone_name="resnet34",
        pretrained=False,
    ).to(device)
    model.load_state_dict(state_dict)

    if domain_config is None:
        domain_config = DomainModelConfig()

    domain_builder = DomainModel.from_config(domain_config)
    dataset_cfg = DatasetConfig(root_dir=root_dir)

    src_aucs, src_paucs = [], []
    tgt_aucs, tgt_paucs = [], []

    logger.info("=" * 82)
    logger.info(f"{'Machine':<12} | {'Domain':<7} | {'AUC':<7} | {'pAUC':<7} | {'Acc':<7} | {'F1':<7} | {'Thr':<7}")
    logger.info("-" * 82)

    for machine in dataset_cfg.machine_types:
        res = evaluate_machine_domain(
            machine_type=machine,
            root_dir=root_dir,
            model=model,
            device=device,
            domain_model_builder=domain_builder,
            batch_size=batch_size,
            patch_size=patch_size,
            stride=stride,
            attr_dim=attr_dim,
            num_workers=num_workers,
        )

        for domain, auc_list, pauc_list in [("source", src_aucs, src_paucs), ("target", tgt_aucs, tgt_paucs)]:
            d_res = res.get(domain, {})
            if "auc" in d_res and not np.isnan(d_res["auc"]):
                auc = d_res["auc"]
                pauc = d_res["pauc"]
                acc = d_res.get("accuracy", float("nan"))
                f1 = d_res.get("f1", float("nan"))
                thr = d_res.get("threshold", 0.0)

                auc_list.append(auc)
                pauc_list.append(pauc)

                logger.info(
                    f"{machine:<12} | {domain:<7} | {auc:.4f}  | {pauc:.4f}  | {acc:.4f}  | {f1:.4f}  | {thr:.3f}"
                )
            else:
                logger.info(f"{machine:<12} | {domain:<7} | [No Data / Skipped]")

    logger.info("=" * 82)
    src_h_auc = harmonic_mean(src_aucs)
    src_h_pauc = harmonic_mean(src_paucs)
    tgt_h_auc = harmonic_mean(tgt_aucs)
    tgt_h_pauc = harmonic_mean(tgt_paucs)

    logger.info("📊 SUMMARY RESULTS (Harmonic Means across machine types):")
    logger.info(f"  • Source Domain : Mean AUC = {src_h_auc:.4f} | Mean pAUC = {src_h_pauc:.4f}")
    logger.info(f"  • Target Domain : Mean AUC = {tgt_h_auc:.4f} | Mean pAUC = {tgt_h_pauc:.4f}")
    logger.info("=" * 82)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Anomaly Sound Detection Model Checkpoint.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint .pth file")
    parser.add_argument("--root-dir", type=str, default="training_data", help="Root dataset directory")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--patch-size", type=int, default=32, help="Patch size")
    parser.add_argument("--stride", type=int, default=16, help="Patch stride")
    parser.add_argument("--embed-dim", type=int, default=128, help="Embedding dimension")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--k-clusters", type=int, default=5, help="Number of KMeans clusters for normal modeling")
    parser.add_argument("--cov-type", type=str, default="lw", choices=["lw", "oas", "empirical", "diag"], help="Covariance estimation type")
    parser.add_argument("--target-fpr", type=float, default=0.05, help="Target FPR for threshold calibration")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="auto", help="Device ('auto', 'cuda', 'cpu')")
    return parser.parse_args()


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()

    args = parse_args()
    d_cfg = DomainModelConfig(
        k=args.k_clusters,
        cov_type=args.cov_type,
        target_fpr=args.target_fpr,
        seed=args.seed,
    )
    run_evaluation(
        checkpoint_path=args.checkpoint,
        root_dir=args.root_dir,
        batch_size=args.batch_size,
        patch_size=args.patch_size,
        stride=args.stride,
        embed_dim=args.embed_dim,
        num_workers=args.num_workers,
        seed=args.seed,
        device_str=args.device,
        domain_config=d_cfg,
    )
