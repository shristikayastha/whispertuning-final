"""
Main Training Script for Nepali ASR

This script explains the complete training pipeline for
fine-tuning Whisper on the Nepali Common Voice dataset.

Usage:
    python train.py
    python train.py --config configs/config.yaml
    python train.py --resume outputs/checkpoints/checkpoint-1000
"""

import argparse
import logging
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.utils.helpers import load_config, setup_logging, get_device, seed_everything, create_output_dirs
from src.data.data_loader import load_common_voice_dataset, prepare_dataset
from src.data.data_collator import DataCollatorSpeechSeq2SeqWithPadding
from src.models.whisper_model import load_whisper_model, setup_lora, prepare_model_for_training
from src.models.tokenizer import load_processor
from src.training.trainer import create_training_arguments, create_trainer, train_model, evaluate_model
from src.training.metrics import create_compute_metrics_fn


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train Nepali ASR model")
    
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode (limited data)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    
    return parser.parse_args()


def main():
    """Main training function."""
    # Parse arguments
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Setup logging
    logger = setup_logging(
        log_level="DEBUG" if args.debug else "INFO",
        log_file="outputs/logs/training.log"
    )
    
    logger.info("=" * 60)
    logger.info("Nepali ASR Training")
    logger.info("=" * 60)
    
    # Set random seed
    seed_everything(args.seed)
    
    # Create output directories
    output_dirs = create_output_dirs(config["training"]["output_dir"])
    
    # Get device
    device = get_device()
    
    # ===== Step 1: Load Processor =====
    logger.info("\n[Step 1/6] Loading processor...")
    processor = load_processor(
        model_name=config["model"]["name"],
        language=config["model"]["language"],
        task=config["model"]["task"]
    )
    
    # ===== Step 2: Load Dataset =====
    logger.info("\n[Step 2/6] Loading dataset...")
    dataset = load_common_voice_dataset(
        dataset_name=config["data"]["dataset_name"],
        language=config["data"]["language"],
        sampling_rate=config["data"]["sampling_rate"]
    )
    
    if args.debug:
        # Use small subset for debugging
        logger.info("Debug mode: using small dataset subset")
        dataset["train"] = dataset["train"].select(range(min(100, len(dataset["train"]))))
        dataset["validation"] = dataset["validation"].select(range(min(20, len(dataset["validation"]))))
    
    # ===== Step 3: Prepare Dataset =====
    logger.info("\n[Step 3/6] Preparing dataset...")
    prepared_dataset = prepare_dataset(
        dataset=dataset,
        processor=processor,
        max_audio_length=config["data"]["max_audio_length"]
    )
    
    # ===== Step 4: Load Model with LoRA =====
    logger.info("\n[Step 4/6] Loading model...")
    model = load_whisper_model(
        model_name=config["model"]["name"],
        device=str(device)
    )
    
    # Apply LoRA
    model = setup_lora(
        model=model,
        r=config["lora"]["r"],
        lora_alpha=config["lora"]["lora_alpha"],
        lora_dropout=config["lora"]["lora_dropout"],
        target_modules=config["lora"]["target_modules"],
        bias=config["lora"]["bias"]
    )
    
    # Prepare for training
    model = prepare_model_for_training(model)
    
    # Force Nepali language and transcribe task during generation
    # This prevents Whisper from trying language detection (which causes FP16/FP32 mismatch)
    model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(
        language=config["model"]["language"],
        task=config["model"]["task"]
    )
    model.generation_config.forced_decoder_ids = model.config.forced_decoder_ids
    logger.info(f"Forced decoder IDs set for language={config['model']['language']}, task={config['model']['task']}")
    
    # ===== Step 5: Setup Trainer =====
    logger.info("\n[Step 5/6] Setting up trainer...")
    
    # Create data collator
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
    )
    
    # Create training arguments (add fp16_full_eval to avoid dtype mismatch during eval)
    training_config = config["training"].copy()
    training_config["fp16_full_eval"] = True
    training_args = create_training_arguments(**training_config)
    
    # Create compute metrics function
    compute_metrics = create_compute_metrics_fn(processor)
    
    # Create trainer
    trainer = create_trainer(
        model=model,
        training_args=training_args,
        train_dataset=prepared_dataset["train"],
        eval_dataset=prepared_dataset["validation"],
        data_collator=data_collator,
        processor=processor,
        compute_metrics=compute_metrics
    )
    
    # ===== Step 6: Train =====
    logger.info("\n[Step 6/6] Starting training...")
    
    train_metrics = train_model(
        trainer=trainer,
        resume_from_checkpoint=args.resume
    )
    
    logger.info("\nTraining complete!")
    logger.info(f"Final training metrics: {train_metrics}")
    
    # ===== Summary =====
    # NOTE: We skip the redundant final evaluate_model() call here because
    # PEFT's generate() runs outside autocast and hits FP16/FP32 dtype mismatch.
    # The trainer already evaluates during training at each eval_steps checkpoint,
    # so WER/CER metrics are already recorded in the training logs.
    logger.info("\n" + "=" * 60)
    logger.info("Training Summary")
    logger.info("=" * 60)
    logger.info(f"Model saved to: {config['training']['output_dir']}")
    logger.info(f"Final training loss: {train_metrics.get('train_loss', 'N/A')}")
    logger.info("Check trainer_state.json for eval WER/CER at each checkpoint.")
    logger.info("Run: python plot_results.py to generate all figures.")
    logger.info("=" * 60)
    
    return train_metrics


if __name__ == "__main__":
    main()
