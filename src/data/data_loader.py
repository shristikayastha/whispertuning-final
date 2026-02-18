"""
Data Loader for Nepali Common Voice Dataset
============================================

This module handles loading and preprocessing the Mozilla Common Voice
dataset for Nepali language ASR training.
"""

import logging
from datasets import load_dataset, DatasetDict, Audio
from transformers import WhisperProcessor
from typing import Optional

logger = logging.getLogger(__name__)


def load_common_voice_dataset(
    dataset_name: str = "mozilla-foundation/common_voice_16_1",
    language: str = "ne-NP",
    sampling_rate: int = 16000
) -> DatasetDict:
    """
    Load the Mozilla Common Voice dataset for Nepali.

    This downloads the dataset from HuggingFace (requires login).
    Audio is automatically resampled to the required sampling rate.

    Args:
        dataset_name: HuggingFace dataset identifier
        language: Language code (ne-NP for Nepali)
        sampling_rate: Target audio sampling rate (16000 for Whisper)

    Returns:
        DatasetDict with train, validation, and test splits
    """
    logger.info(f"Loading dataset: {dataset_name} ({language})")

    # Load train, validation, and test splits
    # trust_remote_code is needed for some dataset scripts
    dataset = DatasetDict()

    dataset["train"] = load_dataset(
        dataset_name,
        language,
        split="train",
        trust_remote_code=True
    )

    dataset["validation"] = load_dataset(
        dataset_name,
        language,
        split="validation",
        trust_remote_code=True
    )

    dataset["test"] = load_dataset(
        dataset_name,
        language,
        split="test",
        trust_remote_code=True
    )

    logger.info(f"Dataset loaded:")
    logger.info(f"  Train: {len(dataset['train'])} samples")
    logger.info(f"  Validation: {len(dataset['validation'])} samples")
    logger.info(f"  Test: {len(dataset['test'])} samples")

    # Cast audio column to resample automatically to 16000 Hz
    # Whisper always expects 16kHz audio
    dataset = dataset.cast_column("audio", Audio(sampling_rate=sampling_rate))
    logger.info(f"Audio resampled to {sampling_rate} Hz")

    return dataset


def prepare_dataset(
    dataset: DatasetDict,
    processor: WhisperProcessor,
    max_audio_length: float = 30.0
) -> DatasetDict:
    """
    Prepare the dataset for training.

    This converts:
      - Raw audio → mel spectrogram features (what Whisper reads)
      - Nepali text → token IDs (what the model predicts)

    Args:
        dataset: Raw DatasetDict from load_common_voice_dataset
        processor: WhisperProcessor (handles both audio and text)
        max_audio_length: Maximum audio clip length in seconds (clips longer than this are removed)

    Returns:
        Processed DatasetDict ready for training
    """
    logger.info("Preparing dataset (converting audio + text)...")

    # Calculate max samples allowed (16000 samples per second)
    max_samples = int(max_audio_length * 16000)

    def process_single_example(example):
        """
        Process one audio+text pair.

        Steps:
        1. Extract mel spectrogram from audio
        2. Tokenize the Nepali text into label IDs
        """
        # Get audio array and sampling rate
        audio = example["audio"]

        # Step 1: Convert audio to mel spectrogram features
        # This is what Whisper actually "sees" - a visual representation of sound
        example["input_features"] = processor.feature_extractor(
            audio["array"],
            sampling_rate=audio["sampling_rate"],
            return_tensors="pt"
        ).input_features[0]

        # Step 2: Tokenize the Nepali text transcript
        # This converts text like "नमस्ते" into numbers the model can predict
        example["labels"] = processor.tokenizer(
            example["sentence"]
        ).input_ids

        return example

    def filter_long_audio(example):
        """Remove audio clips that are too long (over 30 seconds)."""
        return len(example["audio"]["array"]) <= max_samples

    # Filter out clips that are too long first
    logger.info(f"Filtering clips longer than {max_audio_length} seconds...")
    dataset = dataset.filter(filter_long_audio)

    # Apply processing to all examples
    # remove_columns drops the raw columns we no longer need
    columns_to_remove = dataset["train"].column_names
    columns_to_remove = [c for c in columns_to_remove if c not in ["input_features", "labels"]]

    logger.info("Processing audio and text (this may take a few minutes)...")
    dataset = dataset.map(
        process_single_example,
        remove_columns=columns_to_remove,
        desc="Preparing dataset"
    )

    logger.info("Dataset preparation complete!")
    logger.info(f"  Train: {len(dataset['train'])} samples")
    logger.info(f"  Validation: {len(dataset['validation'])} samples")

    return dataset