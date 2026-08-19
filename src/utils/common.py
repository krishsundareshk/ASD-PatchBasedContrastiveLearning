"""Common utilities for device selection, seed control, and logging."""

import os
import random
import logging
import numpy as np


def set_seed(seed: int = 42) -> None:
    """Set random seed across python random, numpy, and torch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def get_device(device_str: str = "auto"):
    """
    Resolve device string ('auto', 'cuda', 'cpu') to a torch.device if torch is installed.
    """
    try:
        import torch
        if device_str == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device_str)
    except ImportError:
        return "cpu"


def setup_logger(name: str = "ASD", level: int = logging.INFO) -> logging.Logger:
    """Configure and return a standard console logger with clean formatting."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(level)
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger
