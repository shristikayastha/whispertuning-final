"""
Tokenizer Utilities for Whisper
===============================

This module handles tokenizer loading and configuration
for Nepali language processing.
"""

from transformers import WhisperProcessor, WhisperTokenizer, WhisperFeatureExtractor
from typing import Optional, List
import logging

logger = logging.getLogger(__name__)


def load_tokenizer(
    model_name: str = "openai/whisper-small",
    language: str = "nepali",
    task: str = "transcribe"
) -> WhisperTokenizer:
    """
    Load and configure the Whisper tokenizer for Nepali.
    
    Args:
        model_name: HuggingFace model identifier
        language: Target language for transcription
        task: Task type ("transcribe" or "translate")
    
    Returns:
        Configured Whisper tokenizer
    """
    logger.info(f"Loading tokenizer for {language} ({task})")
    
    tokenizer = WhisperTokenizer.from_pretrained(
        model_name,
        language=language,
        task=task
    )
    
    logger.info(f"Tokenizer loaded with vocab size: {tokenizer.vocab_size}")
    
    return tokenizer


def load_feature_extractor(
    model_name: str = "openai/whisper-small"
) -> WhisperFeatureExtractor:
    """
    Load the Whisper feature extractor.
    
    The feature extractor converts raw audio to log-mel spectrograms.
    
    Args:
        model_name: HuggingFace model identifier
    
    Returns:
        Whisper feature extractor
    """
    logger.info("Loading Whisper feature extractor")
    
    feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name)
    
    logger.info(
        f"Feature extractor loaded:\n"
        f"  - Sampling rate: {feature_extractor.sampling_rate}\n"
        f"  - Num mel bins: {feature_extractor.feature_size}\n"
        f"  - Chunk length: {feature_extractor.chunk_length}s"
    )
    
    return feature_extractor


def load_processor(
    model_name: str = "openai/whisper-small",
    language: str = "nepali",
    task: str = "transcribe"
) -> WhisperProcessor:
    """
    Load the complete Whisper processor (feature extractor + tokenizer).
    
    Args:
        model_name: HuggingFace model identifier
        language: Target language for transcription
        task: Task type ("transcribe" or "translate")
    
    Returns:
        Whisper processor combining feature extractor and tokenizer
    """
    logger.info(f"Loading Whisper processor for {language}")
    
    processor = WhisperProcessor.from_pretrained(
        model_name,
        language=language,
        task=task
    )
    
    logger.info("Processor loaded successfully")
    
    return processor


def get_special_tokens(tokenizer: WhisperTokenizer) -> dict:
    """
    Get special tokens used by the tokenizer.
    
    Args:
        tokenizer: Whisper tokenizer
    
    Returns:
        Dictionary of special tokens and their IDs
    """
    special_tokens = {
        "bos_token": tokenizer.bos_token,
        "bos_token_id": tokenizer.bos_token_id,
        "eos_token": tokenizer.eos_token,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token": tokenizer.pad_token,
        "pad_token_id": tokenizer.pad_token_id,
    }
    
    return special_tokens


def decode_predictions(
    tokenizer: WhisperTokenizer,
    token_ids: List[int],
    skip_special_tokens: bool = True
) -> str:
    """
    Decode token IDs to text.
    
    Args:
        tokenizer: Whisper tokenizer
        token_ids: List of token IDs to decode
        skip_special_tokens: Whether to skip special tokens in output
    
    Returns:
        Decoded text string
    """
    return tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)


def batch_decode(
    tokenizer: WhisperTokenizer,
    token_ids_batch: List[List[int]],
    skip_special_tokens: bool = True
) -> List[str]:
    """
    Decode a batch of token ID sequences.
    
    Args:
        tokenizer: Whisper tokenizer
        token_ids_batch: Batch of token ID lists
        skip_special_tokens: Whether to skip special tokens
    
    Returns:
        List of decoded text strings
    """
    return tokenizer.batch_decode(
        token_ids_batch,
        skip_special_tokens=skip_special_tokens
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("Testing tokenizer loading...")
    
    # Test loading
    processor = load_processor()
    special_tokens = get_special_tokens(processor.tokenizer)
    
    print(f"Special tokens: {special_tokens}")
    
    # Test encoding/decoding Nepali text
    nepali_text = "नेपाली भाषा"
    tokens = processor.tokenizer(nepali_text)
    print(f"Nepali text: {nepali_text}")
    print(f"Token IDs: {tokens['input_ids']}")
    
    decoded = decode_predictions(processor.tokenizer, tokens['input_ids'])
    print(f"Decoded: {decoded}")
