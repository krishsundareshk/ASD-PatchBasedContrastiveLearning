"""
Joint Contrastive Training Pipeline.
Trains PatchAttentionCLModel jointly across all machine types using NT-Xent loss.
"""

import os
import argparse
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import ConcatDataset, DataLoader
from tqdm import tqdm

from ..config import DatasetConfig
from ..data.attributes import compute_global_attribute_dimension
from ..data.dataset import ASTRAPatchDataset
from ..models.patch_model import PatchAttentionCLModel
from ..models.losses import NTXentLoss
from ..utils.common import set_seed, get_device, setup_logger
from ..utils.checkpoint import CheckpointManager


def run_training(
    root_dir: str = "training_data",
    checkpoint_dir: str = "checkpoints",
    batch_size: int = 32,
    epochs: int = 100,
    learning_rate: float = 2e-4,
    temperature: float = 0.1,
    embed_dim: int = 128,
    max_patches: int = 64,
    stride: int = 16,
    num_workers: int = 4,
    seed: int = 42,
    device_str: str = "auto",
    resume: bool = True,
) -> None:
    """
    Execute end-to-end joint contrastive training.
    """
    logger = setup_logger("ASD.Train")
    set_seed(seed)
    device = get_device(device_str)
    logger.info(f"Target execution device: {device}")

    dataset_cfg = DatasetConfig(root_dir=root_dir, stride=stride, max_patches=max_patches)

    # 1) Compute global attribute dimension across machines
    global_attr_dim = compute_global_attribute_dimension(
        root_dir=root_dir,
        machines=dataset_cfg.machine_types,
        use_attr_machines=dataset_cfg.use_attr_machines,
    )
    logger.info(f"Aligned global attribute dimension: {global_attr_dim}")

    # 2) Assemble joint multi-machine dataset
    datasets = []
    for m in dataset_cfg.machine_types:
        ds = ASTRAPatchDataset(
            root_dir=root_dir,
            machine_type=m,
            split="train",
            patch_size=dataset_cfg.patch_size,
            stride=stride,
            max_patches=max_patches,
            global_attr_dim=global_attr_dim,
            use_attributes=(m in dataset_cfg.use_attr_machines),
        )
        if len(ds) > 0:
            datasets.append(ds)
            logger.info(f"Loaded {len(ds)} training samples for machine '{m}'")

    if not datasets:
        logger.error(f"No training data found in {root_dir}. Please check your dataset directory.")
        return

    joint_dataset = ConcatDataset(datasets)
    logger.info(f"Total joint training dataset size: {len(joint_dataset)} samples")

    loader = DataLoader(
        joint_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    # 3) Initialize Model, Optimizer, Scheduler, and Loss
    model = PatchAttentionCLModel(
        embed_dim=embed_dim,
        attr_dim=global_attr_dim,
        attn_hidden_dim=128,
        backbone_name="resnet34",
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10)
    criterion = NTXentLoss(temperature=temperature)

    ckpt_mgr = CheckpointManager(checkpoint_dir=checkpoint_dir, keep_last_k=5)

    start_epoch = 1
    best_loss = float("inf")

    # 4) Resume from existing checkpoint if available
    if resume:
        last_epoch, last_ckpt_path = ckpt_mgr.find_latest_checkpoint()
        if last_ckpt_path is not None:
            logger.info(f"🔁 Resuming from latest checkpoint: {last_ckpt_path} (epoch {last_epoch})")
            checkpoint = torch.load(last_ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint["model_state"])
            if "optim_state" in checkpoint:
                optimizer.load_state_dict(checkpoint["optim_state"])
            best_loss = checkpoint.get("avg_loss", float("inf"))
            start_epoch = last_epoch + 1

    if start_epoch > epochs:
        logger.info(f"Training already reached target epoch {epochs}. Nothing to train.")
        return

    # 5) Main Training Loop
    logger.info(f"🚀 Commencing training from epoch {start_epoch} to {epochs}...")

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        total_loss = 0.0
        pbar = tqdm(loader, desc=f"[Epoch {epoch:03d}/{epochs:03d}]", ncols=85)

        for batch in pbar:
            p1 = batch["patches_1"].to(device)
            p2 = batch["patches_2"].to(device)
            attrs = batch["attrs"].to(device)

            optimizer.zero_grad()
            z1 = model(p1, attrs=attrs)
            z2 = model(p2, attrs=attrs)
            loss = criterion(z1, z2)
            loss.backward()
            optimizer.step()

            batch_loss = loss.item()
            total_loss += batch_loss
            pbar.set_postfix(loss=f"{batch_loss:.4f}")

        avg_loss = total_loss / len(loader)
        scheduler.step(avg_loss)

        current_lr = optimizer.param_groups[0]["lr"]
        logger.info(f"Epoch {epoch:03d}/{epochs:03d} | Avg Loss: {avg_loss:.4f} | LR: {current_lr:.2e}")

        # Save rolling checkpoint
        ckpt_mgr.save_checkpoint(
            epoch=epoch,
            model_state=model.state_dict(),
            optimizer_state=optimizer.state_dict(),
            avg_loss=avg_loss,
            extra_info={"attr_dim": global_attr_dim, "embed_dim": embed_dim},
        )

    logger.info(f"✅ Training completed successfully. Checkpoints preserved in '{checkpoint_dir}'")


def parse_args():
    parser = argparse.ArgumentParser(description="Joint Contrastive Training for Anomaly Sound Detection.")
    parser.add_argument("--root-dir", type=str, default="training_data", help="Root dataset directory")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=100, help="Total training epochs")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--temperature", type=float, default=0.1, help="NT-Xent temperature")
    parser.add_argument("--embed-dim", type=int, default=128, help="Patch projection embedding dimension")
    parser.add_argument("--max-patches", type=int, default=64, help="Max patches per spectrogram sample")
    parser.add_argument("--stride", type=int, default=16, help="Patch extraction stride")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="auto", help="Device ('auto', 'cuda', 'cpu')")
    parser.add_argument("--no-resume", action="store_true", help="Start training from epoch 1 instead of resuming")
    return parser.parse_args()


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()

    args = parse_args()
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
