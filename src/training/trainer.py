"""
Training Pipeline for Whisper Fine-tuning
==========================================

This module provides the training loop using HuggingFace's
Seq2SeqTrainer with optimized settings for ASR training.
"""

import os
import torch
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperForConditionalGeneration,
    WhisperProcessor
)
from typing import Optional, Dict, Any, Callable
import logging

logger = logging.getLogger(__name__)


def create_training_arguments(
    output_dir: str = "./outputs/checkpoints",
    per_device_train_batch_size: int = 8,
    per_device_eval_batch_size: int = 8,
    gradient_accumulation_steps: int = 2,
    learning_rate: float = 1e-4,
    warmup_steps: int = 500,
    max_steps: int = 5000,
    fp16: bool = True,
    evaluation_strategy: str = "steps",
    eval_steps: int = 500,
    save_steps: int = 500,
    logging_steps: int = 25,
    save_total_limit: int = 3,
    load_best_model_at_end: bool = True,
    metric_for_best_model: str = "wer",
    greater_is_better: bool = False,
    predict_with_generate: bool = True,
    generation_max_length: int = 225,
    report_to: str = "tensorboard",
    **kwargs
) -> Seq2SeqTrainingArguments:
    """
    Create training arguments for Whisper fine-tuning.
    
    Args:
        output_dir: Directory to save checkpoints
        per_device_train_batch_size: Training batch size per device
        per_device_eval_batch_size: Evaluation batch size per device
        gradient_accumulation_steps: Number of gradient accumulation steps
        learning_rate: Initial learning rate
        warmup_steps: Number of warmup steps
        max_steps: Maximum training steps
        fp16: Use mixed precision training
        evaluation_strategy: When to evaluate ("steps" or "epoch")
        eval_steps: Evaluate every N steps
        save_steps: Save checkpoint every N steps
        logging_steps: Log every N steps
        save_total_limit: Maximum checkpoints to keep
        load_best_model_at_end: Load best model when training ends
        metric_for_best_model: Metric to determine best model
        greater_is_better: Whether higher metric is better
        predict_with_generate: Use generation for predictions
        generation_max_length: Maximum generation length
        report_to: Where to report metrics
        **kwargs: Additional arguments
    
    Returns:
        Configured Seq2SeqTrainingArguments
    """
    # Create output directory if needed
    os.makedirs(output_dir, exist_ok=True)
    
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        max_steps=max_steps,
        fp16=fp16 and torch.cuda.is_available(),
        evaluation_strategy=evaluation_strategy,
        eval_steps=eval_steps,
        save_steps=save_steps,
        logging_steps=logging_steps,
        save_total_limit=save_total_limit,
        load_best_model_at_end=load_best_model_at_end,
        metric_for_best_model=metric_for_best_model,
        greater_is_better=greater_is_better,
        predict_with_generate=predict_with_generate,
        generation_max_length=generation_max_length,
        report_to=report_to,
        remove_unused_columns=True,
        label_names=["labels"],
        **kwargs
    )
    
    logger.info(f"Training arguments created:")
    logger.info(f"  - Output dir: {output_dir}")
    logger.info(f"  - Batch size: {per_device_train_batch_size} x {gradient_accumulation_steps} = {per_device_train_batch_size * gradient_accumulation_steps}")
    logger.info(f"  - Learning rate: {learning_rate}")
    logger.info(f"  - Max steps: {max_steps}")
    logger.info(f"  - FP16: {fp16 and torch.cuda.is_available()}")
    
    return training_args


def create_trainer(
    model: WhisperForConditionalGeneration,
    training_args: Seq2SeqTrainingArguments,
    train_dataset,
    eval_dataset,
    data_collator,
    processor: WhisperProcessor,
    compute_metrics: Optional[Callable] = None
) -> Seq2SeqTrainer:
    """
    Create the Seq2SeqTrainer for Whisper fine-tuning.
    
    Args:
        model: Whisper model (with or without LoRA)
        training_args: Training arguments
        train_dataset: Training dataset
        eval_dataset: Evaluation dataset
        data_collator: Data collator for batching
        processor: Whisper processor for decoding
        compute_metrics: Function to compute evaluation metrics
    
    Returns:
        Configured Seq2SeqTrainer
    """
    logger.info("Creating Seq2SeqTrainer...")
    
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=processor.feature_extractor,
    )
    
    logger.info("Trainer created successfully")
    
    return trainer


def train_model(
    trainer: Seq2SeqTrainer,
    resume_from_checkpoint: Optional[str] = None
) -> Dict[str, Any]:
    """
    Execute model training.
    
    Args:
        trainer: Configured Seq2SeqTrainer
        resume_from_checkpoint: Path to checkpoint to resume from
    
    Returns:
        Training results dictionary
    """
    logger.info("Starting training...")
    
    if resume_from_checkpoint:
        logger.info(f"Resuming from checkpoint: {resume_from_checkpoint}")
    
    # Train
    train_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    
    # Log results
    logger.info(f"Training completed!")
    logger.info(f"  - Total steps: {train_result.global_step}")
    logger.info(f"  - Training loss: {train_result.training_loss:.4f}")
    
    # Save final model
    trainer.save_model()
    logger.info("Model saved successfully")
    
    # Save training metrics
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    
    return metrics


def evaluate_model(
    trainer: Seq2SeqTrainer,
    eval_dataset=None
) -> Dict[str, float]:
    """
    Evaluate the trained model.
    
    Args:
        trainer: Trained Seq2SeqTrainer
        eval_dataset: Optional evaluation dataset (uses trainer's if None)
    
    Returns:
        Evaluation metrics dictionary
    """
    logger.info("Evaluating model...")
    
    eval_result = trainer.evaluate(eval_dataset=eval_dataset)
    
    # Log results
    logger.info("Evaluation results:")
    for key, value in eval_result.items():
        logger.info(f"  - {key}: {value:.4f}" if isinstance(value, float) else f"  - {key}: {value}")
    
    # Save metrics
    trainer.log_metrics("eval", eval_result)
    trainer.save_metrics("eval", eval_result)
    
    return eval_result


def get_last_checkpoint(output_dir: str) -> Optional[str]:
    """
    Get the path to the last checkpoint in the output directory.
    
    Args:
        output_dir: Training output directory
    
    Returns:
        Path to last checkpoint or None
    """
    if not os.path.isdir(output_dir):
        return None
    
    checkpoints = [
        d for d in os.listdir(output_dir)
        if d.startswith("checkpoint-") and os.path.isdir(os.path.join(output_dir, d))
    ]
    
    if not checkpoints:
        return None
    
    # Sort by step number
    checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))
    
    return os.path.join(output_dir, checkpoints[-1])


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Test training arguments creation
    args = create_training_arguments(output_dir="./test_output")
    print(f"Training arguments created: {args.output_dir}")
