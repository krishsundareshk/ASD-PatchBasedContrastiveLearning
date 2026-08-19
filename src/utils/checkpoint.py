"""Checkpoint manager for saving, loading, rolling pruning, and inspecting models."""

import os
import re
from typing import Optional, Tuple, Dict, Any


def infer_attr_dim(state_dict: Dict[str, Any]) -> int:
    """
    Infer the attribute dimension from a saved PatchAttentionCLModel state dict.
    Looks for 'attn_pool.attr_bias.weight' or 'attr_mlp.0.weight'.
    """
    if "attn_pool.attr_bias.weight" in state_dict:
        return state_dict["attn_pool.attr_bias.weight"].shape[1]
    if "attr_mlp.0.weight" in state_dict:
        return state_dict["attr_mlp.0.weight"].shape[1]
    return 0


class CheckpointManager:
    """Manages training checkpoint persistence and pruning."""

    def __init__(self, checkpoint_dir: str, keep_last_k: int = 5):
        self.checkpoint_dir = checkpoint_dir
        self.keep_last_k = keep_last_k
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def find_latest_checkpoint(self) -> Tuple[int, Optional[str]]:
        """
        Scan checkpoint_dir for epoch{N}.pth files and return (N, path) of the latest.
        Returns (0, None) if no checkpoints are found.
        """
        if not os.path.isdir(self.checkpoint_dir):
            return 0, None

        best_epoch, best_path = 0, None
        for filename in os.listdir(self.checkpoint_dir):
            match = re.match(r"^epoch(\d+)\.pth$", filename)
            if match:
                epoch_num = int(match.group(1))
                if epoch_num > best_epoch:
                    best_epoch = epoch_num
                    best_path = os.path.join(self.checkpoint_dir, filename)

        return best_epoch, best_path

    def list_available_epochs(self, start_epoch: int = 1) -> list:
        """Return a sorted list of epoch numbers available in the checkpoint directory."""
        if not os.path.isdir(self.checkpoint_dir):
            return []
        epochs = []
        for filename in os.listdir(self.checkpoint_dir):
            match = re.match(r"^epoch(\d+)\.pth$", filename)
            if match:
                epoch_num = int(match.group(1))
                if epoch_num >= start_epoch:
                    epochs.append(epoch_num)
        return sorted(epochs)

    def save_checkpoint(
        self,
        epoch: int,
        model_state: Dict[str, Any],
        optimizer_state: Dict[str, Any],
        avg_loss: float,
        extra_info: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Save an epoch checkpoint and prune older checkpoints if exceeding keep_last_k."""
        import torch

        checkpoint_data = {
            "epoch": epoch,
            "model_state": model_state,
            "optim_state": optimizer_state,
            "avg_loss": avg_loss,
        }
        if extra_info:
            checkpoint_data.update(extra_info)

        save_path = os.path.join(self.checkpoint_dir, f"epoch{epoch}.pth")
        torch.save(checkpoint_data, save_path)

        # Prune old checkpoints
        if self.keep_last_k > 0:
            self._prune_old_checkpoints()

        return save_path

    def _prune_old_checkpoints(self) -> None:
        """Keep only the latest keep_last_k checkpoints."""
        filenames = [
            f for f in os.listdir(self.checkpoint_dir)
            if re.match(r"^epoch\d+\.pth$", f)
        ]
        filenames.sort(key=lambda f: int(re.findall(r"^epoch(\d+)\.pth$", f)[0]))

        if len(filenames) > self.keep_last_k:
            to_delete = filenames[:-self.keep_last_k]
            for fn in to_delete:
                try:
                    os.remove(os.path.join(self.checkpoint_dir, fn))
                except OSError:
                    pass
