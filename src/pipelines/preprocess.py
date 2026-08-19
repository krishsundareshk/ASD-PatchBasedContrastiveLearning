"""
Audio Preprocessing Pipeline.
Converts raw .wav audio files to 224x224 RGB Log-Mel Spectrogram PNGs.
"""

import argparse

from ..config import AudioConfig
from ..data.audio_processing import batch_convert_audio_directory
from ..utils.common import setup_logger


def run_preprocessing(
    base_dir: str = "training_data",
    sample_rate: int = 16000,
    n_mels: int = 128,
    num_workers: int = 4,
    cmap: str = "plasma",
) -> None:
    """
    Execute batch audio preprocessing pipeline.
    """
    logger = setup_logger("ASD.Preprocess")
    logger.info(f"Starting spectrogram conversion on: {base_dir}")

    config = AudioConfig(
        sample_rate=sample_rate,
        n_mels=n_mels,
        cmap_name=cmap,
    )

    counts = batch_convert_audio_directory(
        base_dir=base_dir,
        categories=["train", "supplemental", "test"],
        config=config,
        num_workers=num_workers,
    )

    logger.info("Spectrogram conversion summary:")
    for machine, count in counts.items():
        logger.info(f"  • {machine:<12}: {count} spectrograms generated")
    logger.info("✅ Preprocessing pipeline complete.")


def parse_args():
    parser = argparse.ArgumentParser(description="Convert .wav audio files to RGB Log-Mel spectrogram PNGs.")
    parser.add_argument("--base-dir", type=str, default="training_data", help="Root directory containing machine folders")
    parser.add_argument("--sample-rate", type=int, default=16000, help="Audio sample rate (Hz)")
    parser.add_argument("--n-mels", type=int, default=128, help="Number of Mel frequency bins")
    parser.add_argument("--num-workers", type=int, default=4, help="Parallel processing workers")
    parser.add_argument("--cmap", type=str, default="plasma", help="Matplotlib colormap name")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_preprocessing(
        base_dir=args.base_dir,
        sample_rate=args.sample_rate,
        n_mels=args.n_mels,
        num_workers=args.num_workers,
        cmap=args.cmap,
    )
