"""
Unified Command-Line Interface (CLI) for Anomaly Sound Detection.
Provides subcommands:
  - preprocess : Convert .wav audio files to 224x224 RGB Log-Mel Spectrogram PNGs.
  - train      : Joint contrastive self-supervised model training across machines.
  - evaluate   : Evaluate model checkpoint on test domain datasets.
  - tune       : Run grid search hyperparameter tuning for normal domain modeling.
"""

import os
import sys
import argparse

# Ensure project root is in Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.pipelines.preprocess import run_preprocessing
from src.pipelines.train import run_training
from src.pipelines.evaluate import run_evaluation
from src.pipelines.tune import run_tuning
from src.config import DomainModelConfig


def main():
    parser = argparse.ArgumentParser(
        description="Unified CLI for Contrastive Patch-Attention Anomaly Sound Detection (ASD).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # -------------------------------------------------------------
    # 1. Preprocessing Subcommand
    # -------------------------------------------------------------
    p_prep = subparsers.add_parser("preprocess", help="Convert .wav audio files to RGB spectrogram images")
    p_prep.add_argument("--base-dir", type=str, default="training_data", help="Root directory containing machine folders")
    p_prep.add_argument("--sample-rate", type=int, default=16000, help="Audio sampling rate (Hz)")
    p_prep.add_argument("--n-mels", type=int, default=128, help="Number of Mel frequency bins")
    p_prep.add_argument("--num-workers", type=int, default=4, help="Number of worker processes")
    p_prep.add_argument("--cmap", type=str, default="plasma", help="Matplotlib colormap")

    # -------------------------------------------------------------
    # 2. Train Subcommand
    # -------------------------------------------------------------
    p_train = subparsers.add_parser("train", help="Joint self-supervised contrastive model training")
    p_train.add_argument("--root-dir", type=str, default="training_data", help="Root dataset directory")
    p_train.add_argument("--checkpoint-dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    p_train.add_argument("--batch-size", type=int, default=32, help="Batch size")
    p_train.add_argument("--epochs", type=int, default=100, help="Total training epochs")
    p_train.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    p_train.add_argument("--temperature", type=float, default=0.1, help="NT-Xent temperature")
    p_train.add_argument("--embed-dim", type=int, default=128, help="Patch projection embedding dimension")
    p_train.add_argument("--max-patches", type=int, default=64, help="Max patches per spectrogram sample")
    p_train.add_argument("--stride", type=int, default=16, help="Patch extraction stride")
    p_train.add_argument("--num-workers", type=int, default=4, help="DataLoader workers")
    p_train.add_argument("--seed", type=int, default=42, help="Random seed")
    p_train.add_argument("--device", type=str, default="auto", help="Device ('auto', 'cuda', 'cpu')")
    p_train.add_argument("--no-resume", action="store_true", help="Start training from epoch 1 instead of resuming")

    # -------------------------------------------------------------
    # 3. Evaluate Subcommand
    # -------------------------------------------------------------
    p_eval = subparsers.add_parser("evaluate", help="Evaluate model checkpoint on test dataset")
    p_eval.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint .pth file")
    p_eval.add_argument("--root-dir", type=str, default="training_data", help="Root dataset directory")
    p_eval.add_argument("--batch-size", type=int, default=64, help="Batch size")
    p_eval.add_argument("--patch-size", type=int, default=32, help="Patch size")
    p_eval.add_argument("--stride", type=int, default=16, help="Patch stride")
    p_eval.add_argument("--embed-dim", type=int, default=128, help="Embedding dimension")
    p_eval.add_argument("--num-workers", type=int, default=4, help="DataLoader workers")
    p_eval.add_argument("--k-clusters", type=int, default=5, help="Number of KMeans clusters for normal modeling")
    p_eval.add_argument("--cov-type", type=str, default="lw", choices=["lw", "oas", "empirical", "diag"], help="Covariance estimation method")
    p_eval.add_argument("--target-fpr", type=float, default=0.05, help="Target FPR for decision threshold calibration")
    p_eval.add_argument("--seed", type=int, default=42, help="Random seed")
    p_eval.add_argument("--device", type=str, default="auto", help="Device ('auto', 'cuda', 'cpu')")

    # -------------------------------------------------------------
    # 4. Tune Subcommand
    # -------------------------------------------------------------
    p_tune = subparsers.add_parser("tune", help="Grid search hyperparameter tuning for normal domain modeling")
    p_tune.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint .pth file")
    p_tune.add_argument("--root-dir", type=str, default="training_data", help="Root dataset directory")
    p_tune.add_argument("--batch-size", type=int, default=64, help="Batch size")
    p_tune.add_argument("--patch-size", type=int, default=32, help="Patch size")
    p_tune.add_argument("--stride", type=int, default=16, help="Patch stride")
    p_tune.add_argument("--embed-dim", type=int, default=128, help="Embedding dimension")
    p_tune.add_argument("--num-workers", type=int, default=4, help="DataLoader workers")
    p_tune.add_argument("--top-n", type=int, default=5, help="Number of top configurations to show")
    p_tune.add_argument("--seed", type=int, default=42, help="Random seed")
    p_tune.add_argument("--device", type=str, default="auto", help="Device ('auto', 'cuda', 'cpu')")

    args = parser.parse_args()

    if args.command == "preprocess":
        run_preprocessing(
            base_dir=args.base_dir,
            sample_rate=args.sample_rate,
            n_mels=args.n_mels,
            num_workers=args.num_workers,
            cmap=args.cmap,
        )
    elif args.command == "train":
        run_training(
            root_dir=args.root_dir,
            checkpoint_dir=args.checkpoint_dir,
            batch_size=args.batch_size,
            epochs=args.epochs,
            learning_rate=args.lr,
            temperature=args.temperature,
            embed_dim=args.embed_dim,
            max_patches=args.max_patches,
            stride=args.stride,
            num_workers=args.num_workers,
            seed=args.seed,
            device_str=args.device,
            resume=(not args.no_resume),
        )
    elif args.command == "evaluate":
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
    elif args.command == "tune":
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
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()
