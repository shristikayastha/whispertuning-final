"""
Data Loader for Nepali ASR Dataset
===================================

This module handles loading and preprocessing Nepali ASR datasets
for Whisper fine-tuning. Supports both Common Voice and OpenSLR formats.
"""

import os
import logging
from datasets import load_dataset, DatasetDict, Audio
from transformers import WhisperProcessor
from typing import Optional

logger = logging.getLogger(__name__)


def load_common_voice_dataset(
    dataset_name: str = "spktsagar/openslr-nepali-asr-cleaned",
    language: str = "ne-NP",
    sampling_rate: int = 16000,
    max_train_samples: int = 10000,
    max_eval_samples: int = 1000
) -> DatasetDict:
    """
    Load a Nepali ASR dataset from HuggingFace.

    Supports:
      - spktsagar/openslr-nepali-asr-cleaned (OpenSLR format)
      - mozilla-foundation/common_voice_* (Common Voice format)

    Args:
        dataset_name: HuggingFace dataset identifier
        language: Language code (ne-NP for Nepali)
        sampling_rate: Target audio sampling rate (16000 for Whisper)
        max_train_samples: Maximum training samples (default 10000, saves disk)
        max_eval_samples: Maximum eval/test samples (default 1000)

    Returns:
        DatasetDict with train, validation, and test splits
    """
    logger.info(f"Loading dataset: {dataset_name} ({language})")

    # Get HF token for gated datasets
    hf_token = os.environ.get("HF_TOKEN", None)

    # Detect dataset type
    is_openslr = "openslr" in dataset_name.lower() or "spktsagar" in dataset_name.lower()

    if is_openslr:
        # OpenSLR format: single unsplit dataset with 'utterance' and 'transcription' fields
        logger.info("Detected OpenSLR format dataset")
        full_dataset = load_dataset(
            dataset_name,
            split="train",
            trust_remote_code=True,
            token=hf_token
        )

        # Rename columns to match Common Voice format
        # utterance -> audio, transcription -> sentence
        full_dataset = full_dataset.rename_column("utterance", "audio")
        full_dataset = full_dataset.rename_column("transcription", "sentence")

        # Split: 90% train, 5% validation, 5% test
        logger.info("Splitting dataset: 90% train, 5% val, 5% test")
        split1 = full_dataset.train_test_split(test_size=0.1, seed=42)
        split2 = split1["test"].train_test_split(test_size=0.5, seed=42)

        dataset = DatasetDict({
            "train": split1["train"],
            "validation": split2["train"],
            "test": split2["test"],
        })
    else:
        # Common Voice format: pre-split with 'audio' and 'sentence' fields
        logger.info("Detected Common Voice format dataset")
        dataset = DatasetDict()

        dataset["train"] = load_dataset(
            dataset_name,
            language,
            split="train",
            trust_remote_code=True,
            token=hf_token
        )

        dataset["validation"] = load_dataset(
            dataset_name,
            language,
            split="validation",
            trust_remote_code=True,
            token=hf_token
        )

        dataset["test"] = load_dataset(
            dataset_name,
            language,
            split="test",
            trust_remote_code=True,
            token=hf_token
        )

    # Subset to save disk space and training time
    # 10K train + 1K val/test is plenty for Whisper fine-tuning
    if max_train_samples and len(dataset["train"]) > max_train_samples:
        logger.info(f"Subsetting train: {len(dataset['train'])} -> {max_train_samples}")
        dataset["train"] = dataset["train"].select(range(max_train_samples))
    if max_eval_samples and len(dataset["validation"]) > max_eval_samples:
        logger.info(f"Subsetting validation: {len(dataset['validation'])} -> {max_eval_samples}")
        dataset["validation"] = dataset["validation"].select(range(max_eval_samples))
    if max_eval_samples and len(dataset["test"]) > max_eval_samples:
        logger.info(f"Subsetting test: {len(dataset['test'])} -> {max_eval_samples}")
        dataset["test"] = dataset["test"].select(range(max_eval_samples))

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