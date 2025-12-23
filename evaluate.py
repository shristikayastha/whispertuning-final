"""
Evaluation Script for Nepali ASR
================================

This script evaluates the trained model on test data and
generates detailed performance reports.

Usage:
    python evaluate.py --model outputs/checkpoints/best_model
    python evaluate.py --model outputs/checkpoints/best_model --test-file audio.wav
"""

import argparse
import logging
import json
import os
import sys
from pathlib import Path
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.utils.helpers import load_config, setup_logging, get_device
from src.data.data_loader import load_common_voice_dataset, prepare_dataset
from src.models.tokenizer import load_processor
from src.inference.transcriber import NepaliTranscriber
from src.training.metrics import detailed_evaluation, log_sample_predictions


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate Nepali ASR model")
    
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to trained model"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--test-file",
        type=str,
        default=None,
        help="Single audio file to transcribe (optional)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/predictions/evaluation_results.json",
        help="Output file for evaluation results"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Number of samples to evaluate (None for all)"
    )
    
    return parser.parse_args()


def evaluate_on_dataset(
    transcriber: NepaliTranscriber,
    dataset,
    processor,
    num_samples: int = None
) -> dict:
    """
    Evaluate model on a dataset.
    
    Args:
        transcriber: NepaliTranscriber instance
        dataset: Test dataset
        processor: Whisper processor
        num_samples: Number of samples to evaluate
    
    Returns:
        Evaluation results dictionary
    """
    logger = logging.getLogger(__name__)
    
    # Limit samples if specified
    if num_samples:
        dataset = dataset.select(range(min(num_samples, len(dataset))))
    
    logger.info(f"Evaluating on {len(dataset)} samples...")
    
    predictions = []
    references = []
    
    for sample in tqdm(dataset, desc="Transcribing"):
        # Get audio
        audio = sample["audio"]["array"]
        sr = sample["audio"]["sampling_rate"]
        
        # Get reference text
        reference = sample["sentence"]
        
        # Transcribe
        prediction = transcriber.transcribe(audio, sampling_rate=sr)
        
        predictions.append(prediction)
        references.append(reference)
    
    # Compute metrics
    results = detailed_evaluation(predictions, references)
    
    # Add sample predictions
    results["sample_predictions"] = [
        {"prediction": p, "reference": r}
        for p, r in zip(predictions[:10], references[:10])
    ]
    
    return results


def transcribe_single_file(
    transcriber: NepaliTranscriber,
    audio_path: str
) -> str:
    """
    Transcribe a single audio file.
    
    Args:
        transcriber: NepaliTranscriber instance
        audio_path: Path to audio file
    
    Returns:
        Transcription text
    """
    logger = logging.getLogger(__name__)
    
    logger.info(f"Transcribing: {audio_path}")
    
    transcription = transcriber.transcribe(audio_path)
    
    return transcription


def main():
    """Main evaluation function."""
    # Parse arguments
    args = parse_args()
    
    # Setup logging
    logger = setup_logging(log_level="INFO")
    
    logger.info("=" * 60)
    logger.info("Nepali ASR Evaluation")
    logger.info("=" * 60)
    
    # Load configuration
    config = load_config(args.config)
    
    # Get device
    device = get_device()
    
    # Load transcriber
    logger.info(f"\nLoading model from: {args.model}")
    transcriber = NepaliTranscriber(
        model_path=args.model,
        base_model_name=config["model"]["name"],
        device=str(device)
    )
    
    # Print model info
    model_info = transcriber.get_model_info()
    logger.info(f"Model info: {model_info}")
    
    # Single file transcription
    if args.test_file:
        logger.info("\n" + "-" * 40)
        logger.info("Single File Transcription")
        logger.info("-" * 40)
        
        transcription = transcribe_single_file(transcriber, args.test_file)
        
        logger.info(f"\nFile: {args.test_file}")
        logger.info(f"Transcription: {transcription}")
        
        return
    
    # Dataset evaluation
    logger.info("\n" + "-" * 40)
    logger.info("Dataset Evaluation")
    logger.info("-" * 40)
    
    # Load processor
    processor = load_processor(
        model_name=config["model"]["name"],
        language=config["model"]["language"]
    )
    
    # Load test dataset
    logger.info("\nLoading test dataset...")
    dataset = load_common_voice_dataset(
        dataset_name=config["data"]["dataset_name"],
        language=config["data"]["language"],
        sampling_rate=config["data"]["sampling_rate"]
    )
    
    # Evaluate
    results = evaluate_on_dataset(
        transcriber=transcriber,
        dataset=dataset["test"],
        processor=processor,
        num_samples=args.num_samples
    )
    
    # Log results
    logger.info("\n" + "=" * 60)
    logger.info("Evaluation Results")
    logger.info("=" * 60)
    logger.info(f"Word Error Rate (WER): {results['wer']:.2f}%")
    logger.info(f"Character Error Rate (CER): {results['cer']:.2f}%")
    logger.info(f"WER Std Dev: {results['wer_std']:.2f}%")
    logger.info(f"CER Std Dev: {results['cer_std']:.2f}%")
    logger.info(f"Perfect Matches: {results['perfect_matches']}/{results['num_samples']}")
    logger.info("=" * 60)
    
    # Log sample predictions
    logger.info("\nSample Predictions:")
    for i, sample in enumerate(results["sample_predictions"][:5]):
        logger.info(f"\n[{i+1}]")
        logger.info(f"  Reference:  {sample['reference']}")
        logger.info(f"  Prediction: {sample['prediction']}")
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    logger.info(f"\nResults saved to: {output_path}")
    
    return results


if __name__ == "__main__":
    main()
