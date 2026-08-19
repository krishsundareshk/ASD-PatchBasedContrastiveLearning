"""Tests for configuration dataclasses and serialization."""

import os
import sys
import tempfile

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import (
    AudioConfig,
    DatasetConfig,
    ModelConfig,
    TrainConfig,
    DomainModelConfig,
    save_config,
    load_config,
)


def test_default_configs():
    audio_cfg = AudioConfig()
    assert audio_cfg.sample_rate == 16000
    assert audio_cfg.n_mels == 128
    assert audio_cfg.img_size == (224, 224)

    dataset_cfg = DatasetConfig()
    assert "ToyCar" in dataset_cfg.machine_types
    assert dataset_cfg.patch_size == 32

    model_cfg = ModelConfig()
    assert model_cfg.embed_dim == 128
    assert model_cfg.backbone == "resnet34"

    train_cfg = TrainConfig()
    assert train_cfg.batch_size == 32
    assert train_cfg.temperature == 0.1

    domain_cfg = DomainModelConfig()
    assert domain_cfg.k == 5
    assert domain_cfg.cov_type == "lw"


def test_config_serialization():
    train_cfg = TrainConfig(batch_size=64, epochs=50, learning_rate=1e-3)
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_file = os.path.join(tmpdir, "train_config.json")
        save_config(train_cfg, tmp_file)
        assert os.path.isfile(tmp_file)

        loaded_cfg = load_config(TrainConfig, tmp_file)
        assert loaded_cfg.batch_size == 64
        assert loaded_cfg.epochs == 50
        assert loaded_cfg.learning_rate == 1e-3
