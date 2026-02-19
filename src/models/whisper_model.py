"""
Whisper Model Setup with LoRA
=============================

This module handles loading the Whisper model and configuring
LoRA (Low-Rank Adaptation) for memory-efficient fine-tuning.
"""

import torch
from transformers import WhisperForConditionalGeneration
from peft import LoraConfig, get_peft_model, PeftModel
from typing import Optional, List, Dict, Any
import logging

logger = logging.getLogger(__name__)


def load_whisper_model(
    model_name: str = "openai/whisper-small",
    device: Optional[str] = None,
    torch_dtype: torch.dtype = torch.float16,
    load_in_8bit: bool = False
) -> WhisperForConditionalGeneration:
    """
    Load the base Whisper model.
    
    Args:
        model_name: HuggingFace model identifier
        device: Device to load model on (None for auto)
        torch_dtype: Torch data type for model weights
        load_in_8bit: Whether to use 8-bit quantization
    
    Returns:
        Loaded Whisper model
    """
    logger.info(f"Loading Whisper model: {model_name}")
    
    model_kwargs = {
        "torch_dtype": torch_dtype,
    }
    
    if load_in_8bit:
        model_kwargs["load_in_8bit"] = True
        model_kwargs["device_map"] = "auto"
        logger.info("Loading model with 8-bit quantization")
    elif device:
        model_kwargs["device_map"] = device
    
    model = WhisperForConditionalGeneration.from_pretrained(
        model_name,
        **model_kwargs
    )
    
    # Log model info
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model loaded with {total_params:,} parameters")
    
    return model


def setup_lora(
    model: WhisperForConditionalGeneration,
    r: int = 32,
    lora_alpha: int = 64,
    lora_dropout: float = 0.1,
    target_modules: Optional[List[str]] = None,
    bias: str = "none"
) -> WhisperForConditionalGeneration:
    """
    Apply LoRA adapters to the Whisper model.
    
    LoRA (Low-Rank Adaptation) allows efficient fine-tuning by only
    training small adapter matrices instead of the full model.
    
    Args:
        model: Base Whisper model
        r: LoRA rank (higher = more capacity, more memory)
        lora_alpha: LoRA scaling factor
        lora_dropout: Dropout probability for LoRA layers
        target_modules: List of module names to apply LoRA to
        bias: Bias training strategy ("none", "all", "lora_only")
    
    Returns:
        Model with LoRA adapters applied
    """
    if target_modules is None:
        # Default target modules for Whisper (attention layers)
        target_modules = ["q_proj", "v_proj", "k_proj", "out_proj"]
    
    logger.info(f"Setting up LoRA with rank={r}, alpha={lora_alpha}")
    logger.info(f"Target modules: {target_modules}")
    
    # Configure LoRA
    lora_config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias=bias,
        # NOTE: Do NOT set task_type here! SEQ_2_SEQ_LM and CAUSAL_LM
        # create PEFT wrappers that inject input_ids=None into forward(),
        # but Whisper expects input_features, not input_ids.
        # Without task_type, PEFT uses the base PeftModel which passes
        # kwargs through cleanly.
    )
    
    # Apply LoRA to model
    model = get_peft_model(model, lora_config)
    
    # Log trainable parameters
    trainable_params, total_params = get_trainable_params(model)
    logger.info(
        f"Trainable parameters: {trainable_params:,} / {total_params:,} "
        f"({100 * trainable_params / total_params:.2f}%)"
    )
    
    return model


def get_trainable_params(model) -> tuple:
    """
    Get the number of trainable and total parameters.
    
    Args:
        model: The model to analyze
    
    Returns:
        Tuple of (trainable_params, total_params)
    """
    trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    total_params = sum(p.numel() for p in model.parameters())
    
    return trainable_params, total_params


def load_finetuned_model(
    base_model_name: str,
    adapter_path: str,
    device: Optional[str] = None,
    torch_dtype: torch.dtype = torch.float16
) -> WhisperForConditionalGeneration:
    """
    Load a fine-tuned model with LoRA adapters.
    
    Args:
        base_model_name: HuggingFace model identifier for base model
        adapter_path: Path to saved LoRA adapter weights
        device: Device to load model on
        torch_dtype: Torch data type for model weights
    
    Returns:
        Model with loaded LoRA adapters
    """
    logger.info(f"Loading fine-tuned model from {adapter_path}")
    
    # Load base model
    base_model = load_whisper_model(
        base_model_name,
        device=device,
        torch_dtype=torch_dtype
    )
    
    # Load LoRA adapters
    model = PeftModel.from_pretrained(base_model, adapter_path)
    
    logger.info("Fine-tuned model loaded successfully")
    
    return model


def merge_lora_weights(
    model: WhisperForConditionalGeneration,
    output_path: str
) -> None:
    """
    Merge LoRA weights into the base model and save.
    
    This creates a standalone model without requiring PEFT for inference.
    
    Args:
        model: Model with LoRA adapters
        output_path: Path to save merged model
    """
    logger.info("Merging LoRA weights into base model...")
    
    # Merge weights
    merged_model = model.merge_and_unload()
    
    # Save merged model
    merged_model.save_pretrained(output_path)
    
    logger.info(f"Merged model saved to {output_path}")


def prepare_model_for_training(
    model: WhisperForConditionalGeneration
) -> WhisperForConditionalGeneration:
    """
    Prepare the model for training with gradient checkpointing.
    
    Args:
        model: The model to prepare
    
    Returns:
        Model ready for training
    """
    # Enable gradient checkpointing to reduce memory usage
    model.gradient_checkpointing_enable()
    
    # Ensure input embeddings require gradients
    model.enable_input_require_grads()
    
    logger.info("Model prepared for training with gradient checkpointing")
    
    return model


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("Testing Whisper model loading...")
    
    # Test model loading (requires GPU for full test)
    # model = load_whisper_model("openai/whisper-small")
    # model = setup_lora(model)
    # print("Model loaded and LoRA configured successfully!")
