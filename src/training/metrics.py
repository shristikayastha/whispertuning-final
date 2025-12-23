"""
Evaluation Metrics for ASR
==========================

This module provides Word Error Rate (WER) and Character Error Rate (CER)
computation for evaluating speech recognition performance.
"""

import evaluate
import numpy as np
from typing import Dict, Any, Tuple
import logging

logger = logging.getLogger(__name__)

# Load evaluation metrics
wer_metric = evaluate.load("wer")
cer_metric = evaluate.load("cer")


def compute_wer(predictions: list, references: list) -> float:
    """
    Compute Word Error Rate.
    
    WER = (Substitutions + Insertions + Deletions) / Total Words
    
    Args:
        predictions: List of predicted transcriptions
        references: List of ground truth transcriptions
    
    Returns:
        WER score (0-1, lower is better)
    """
    wer = wer_metric.compute(predictions=predictions, references=references)
    return wer


def compute_cer(predictions: list, references: list) -> float:
    """
    Compute Character Error Rate.
    
    CER = (Substitutions + Insertions + Deletions) / Total Characters
    
    Args:
        predictions: List of predicted transcriptions
        references: List of ground truth transcriptions
    
    Returns:
        CER score (0-1, lower is better)
    """
    cer = cer_metric.compute(predictions=predictions, references=references)
    return cer


def create_compute_metrics_fn(processor):
    """
    Create a compute_metrics function for the Trainer.
    
    This function is called during evaluation to compute WER and CER.
    
    Args:
        processor: WhisperProcessor for decoding predictions
    
    Returns:
        Function that computes metrics from predictions
    """
    def compute_metrics(pred) -> Dict[str, float]:
        """
        Compute WER and CER metrics from predictions.
        
        Args:
            pred: EvalPrediction object containing predictions and labels
        
        Returns:
            Dictionary with 'wer' and 'cer' scores
        """
        pred_ids = pred.predictions
        label_ids = pred.label_ids
        
        # Replace -100 with pad token id for decoding
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        
        # Decode predictions and references
        pred_str = processor.tokenizer.batch_decode(
            pred_ids, 
            skip_special_tokens=True
        )
        label_str = processor.tokenizer.batch_decode(
            label_ids, 
            skip_special_tokens=True
        )
        
        # Normalize text (optional: can add more normalization)
        pred_str = [normalize_text(text) for text in pred_str]
        label_str = [normalize_text(text) for text in label_str]
        
        # Compute metrics
        wer = 100 * compute_wer(pred_str, label_str)
        cer = 100 * compute_cer(pred_str, label_str)
        
        return {
            "wer": wer,
            "cer": cer
        }
    
    return compute_metrics


def normalize_text(text: str) -> str:
    """
    Normalize text for fair comparison.
    
    Args:
        text: Input text string
    
    Returns:
        Normalized text string
    """
    # Basic normalization
    text = text.strip()
    text = " ".join(text.split())  # Normalize whitespace
    
    return text


def detailed_evaluation(
    predictions: list,
    references: list
) -> Dict[str, Any]:
    """
    Perform detailed evaluation with additional statistics.
    
    Args:
        predictions: List of predicted transcriptions
        references: List of ground truth transcriptions
    
    Returns:
        Dictionary with detailed evaluation results
    """
    # Compute main metrics
    wer = compute_wer(predictions, references)
    cer = compute_cer(predictions, references)
    
    # Compute per-sample metrics
    per_sample_wer = []
    per_sample_cer = []
    
    for pred, ref in zip(predictions, references):
        if ref.strip():  # Avoid division by zero
            sample_wer = compute_wer([pred], [ref])
            sample_cer = compute_cer([pred], [ref])
            per_sample_wer.append(sample_wer)
            per_sample_cer.append(sample_cer)
    
    # Compute statistics
    results = {
        "wer": wer * 100,  # Convert to percentage
        "cer": cer * 100,
        "wer_std": np.std(per_sample_wer) * 100 if per_sample_wer else 0,
        "cer_std": np.std(per_sample_cer) * 100 if per_sample_cer else 0,
        "wer_median": np.median(per_sample_wer) * 100 if per_sample_wer else 0,
        "cer_median": np.median(per_sample_cer) * 100 if per_sample_cer else 0,
        "num_samples": len(predictions),
        "perfect_matches": sum(1 for p, r in zip(predictions, references) if p.strip() == r.strip())
    }
    
    return results


def log_sample_predictions(
    predictions: list,
    references: list,
    num_samples: int = 5
) -> None:
    """
    Log sample predictions for qualitative analysis.
    
    Args:
        predictions: List of predicted transcriptions
        references: List of ground truth transcriptions
        num_samples: Number of samples to log
    """
    logger.info("=" * 60)
    logger.info("Sample Predictions")
    logger.info("=" * 60)
    
    for i, (pred, ref) in enumerate(zip(predictions[:num_samples], references[:num_samples])):
        sample_wer = compute_wer([pred], [ref]) * 100
        logger.info(f"\nSample {i + 1}:")
        logger.info(f"  Reference:  {ref}")
        logger.info(f"  Prediction: {pred}")
        logger.info(f"  WER: {sample_wer:.2f}%")
    
    logger.info("=" * 60)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Test metrics
    predictions = ["मेरो नाम राम हो", "नमस्ते"]
    references = ["मेरो नाम राम हो", "नमस्ते संसार"]
    
    results = detailed_evaluation(predictions, references)
    print(f"Evaluation results: {results}")
    
    log_sample_predictions(predictions, references)
