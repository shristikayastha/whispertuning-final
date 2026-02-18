"""
Data Collator for Speech Seq2Seq Training
==========================================

This module handles batching audio samples together during training.
The challenge: audio clips have different lengths, so we need to
pad them all to the same length within each batch.
"""

import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from transformers import WhisperProcessor


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """
    Data collator for Whisper fine-tuning.

    What does a data collator do?
    ------------------------------
    During training, samples are grouped into "batches" (e.g. 8 at a time).
    But each audio clip has a different length. This class:
      1. Pads audio features to the same size (Whisper uses fixed 30s windows)
      2. Pads text labels to the same length (using -100 so they're ignored in loss)

    Think of it like stacking papers of different sizes —
    you add blank space at the bottom so they all match.

    Args:
        processor: WhisperProcessor that handles both audio and text
    """
    processor: WhisperProcessor

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        """
        Collate a list of examples into a batch.

        Args:
            features: List of processed examples, each with:
                      - input_features: mel spectrogram
                      - labels: tokenized text

        Returns:
            Batch dictionary with padded tensors
        """
        # ── Step 1: Handle Audio Features ──────────────────────────────────
        # Extract just the audio features from each example
        input_features = [{"input_features": f["input_features"]} for f in features]

        # Pad all audio features to the same length
        # Whisper always uses 30-second windows (3000 mel frames)
        # Short clips are padded with zeros on the right
        batch = self.processor.feature_extractor.pad(
            input_features,
            return_tensors="pt"  # Return PyTorch tensors
        )

        # ── Step 2: Handle Text Labels ──────────────────────────────────────
        # Extract just the label (text token) sequences
        label_features = [{"input_ids": f["labels"]} for f in features]

        # Pad all label sequences to the same length
        labels_batch = self.processor.tokenizer.pad(
            label_features,
            return_tensors="pt"
        )

        # Replace padding token ID with -100
        # Why -100? PyTorch's cross-entropy loss automatically ignores -100
        # This means the model isn't penalized for "predicting" padding tokens
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1),  # Where attention mask is 0 (padding)
            -100                                 # Replace with -100
        )

        # ── Step 3: Remove decoder start token if present ──────────────────
        # Whisper uses a special start token at the beginning of labels
        # during training it's handled separately, so we strip it here
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        # Add labels to batch
        batch["labels"] = labels

        return batch