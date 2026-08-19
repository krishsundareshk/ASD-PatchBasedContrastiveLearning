"""
Configuration schemas for Anomaly Sound Detection.
Provides strongly typed dataclasses with sensible defaults and serialization utilities.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Set, Tuple, Optional, Any
import json
import os


ALL_DEFAULT_MACHINE_TYPES = [
    "ToyCar",
    "ToyTrain",
    "bearing",
    "valve",
    "fan",
    "gearbox",
    "slider",
]

DEFAULT_USE_ATTR_MACHINES = {"fan", "ToyCar", "valve", "gearbox"}


@dataclass
class AudioConfig:
    """Audio preprocessing and Log-Mel Spectrogram configuration."""
    sample_rate: int = 16000
    n_fft: int = 1024
    hop_length: int = 512
    n_mels: int = 128
    fmin: float = 20.0
    fmax: float = 8000.0
    power: float = 2.0
    cmap_name: str = "plasma"
    img_size: Tuple[int, int] = (224, 224)


@dataclass
class DatasetConfig:
    """Dataset loading and patch extraction configuration."""
    root_dir: str = "training_data"
    patch_size: int = 32
    stride: int = 16
    max_patches: Optional[int] = 64
    machine_types: List[str] = field(default_factory=lambda: list(ALL_DEFAULT_MACHINE_TYPES))
    use_attr_machines: Set[str] = field(default_factory=lambda: set(DEFAULT_USE_ATTR_MACHINES))
    global_attr_dim: int = 0


@dataclass
class ModelConfig:
    """Neural network architecture configuration."""
    backbone: str = "resnet34"
    embed_dim: int = 128
    attn_hidden_dim: int = 128
    attr_dim: int = 0
    pretrained: bool = True


@dataclass
class TrainConfig:
    """Joint contrastive training configuration."""
    batch_size: int = 32
    epochs: int = 100
    learning_rate: float = 2e-4
    temperature: float = 0.1
    earlystop_patience: int = 25
    lr_patience: int = 10
    lr_factor: float = 0.5
    checkpoint_dir: str = "checkpoints"
    keep_last_k: int = 5
    num_workers: int = 4
    seed: int = 42
    device: str = "auto"  # "auto", "cuda", or "cpu"


@dataclass
class DomainModelConfig:
    """Multi-centroid normal domain scoring model configuration."""
    use_pca: bool = True
    pca_variance: float = 0.98
    cov_type: str = "lw"  # "lw" (Ledoit-Wolf), "oas", "empirical", "diag"
    use_cosine: bool = True
    w_maha: float = 0.7
    w_cos: float = 0.3
    k: int = 5
    thr_mode: str = "fpr"  # "fpr" or "percentile"
    target_fpr: float = 0.05
    perc_q: float = 99.0
    seed: int = 42


@dataclass
class EvalConfig:
    """Model evaluation pipeline configuration."""
    root_dir: str = "training_data"
    checkpoint_path: str = ""
    batch_size: int = 64
    patch_size: int = 32
    stride: int = 16
    num_workers: int = 4
    device: str = "auto"
    domain_model: DomainModelConfig = field(default_factory=DomainModelConfig)


@dataclass
class GridSearchConfig:
    """Hyperparameter grid search for domain scoring."""
    ks: List[int] = field(default_factory=lambda: [1, 3, 5])
    cov_types: List[str] = field(default_factory=lambda: ["lw", "diag"])
    pca_options: List[Tuple[bool, float]] = field(
        default_factory=lambda: [(False, 1.0), (True, 0.95), (True, 0.98)]
    )
    cosine_weights: List[Tuple[bool, float, float]] = field(
        default_factory=lambda: [(False, 1.0, 0.0), (True, 0.7, 0.3)]
    )
    thr_options: List[Tuple[str, Optional[float], Optional[float]]] = field(
        default_factory=lambda: [("fpr", 0.05, None), ("percentile", None, 99.0)]
    )


def save_config(config: Any, filepath: str) -> None:
    """Serialize a configuration dataclass to a JSON file."""
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(asdict(config), f, indent=2)


def load_config(config_cls: Any, filepath: str) -> Any:
    """Deserialize a JSON file into a configuration dataclass."""
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    return config_cls(**data)
