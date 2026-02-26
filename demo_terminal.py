"""
Simple Terminal Demo — Nepali ASR (Whisper + LoRA)
==================================================

Loads the fine-tuned model and transcribes audio from the test dataset.
No UI, just clean terminal output.

Usage:
    python demo_terminal.py
    python demo_terminal.py --samples 10
    python demo_terminal.py --audio path/to/audio.wav
"""

import sys
import argparse
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.utils.helpers import load_config, get_device
from src.models.tokenizer import load_processor
from src.data.data_loader import load_common_voice_dataset


def parse_args():
    parser = argparse.ArgumentParser(description="Nepali ASR Demo")
    parser.add_argument("--model", type=str, default="./outputs/checkpoints",
                        help="Path to trained model/checkpoint")
    parser.add_argument("--audio", type=str, default=None,
                        help="Path to a single audio file to transcribe")
    parser.add_argument("--samples", type=int, default=5,
                        help="Number of test samples to demo (default: 5)")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    return parser.parse_args()


def load_model(model_path, config):
    """Load the fine-tuned Whisper + LoRA model."""
    import torch
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    from peft import PeftModel

    device = get_device()
    print(f"  Device: {device}")

    # Load processor
    processor = WhisperProcessor.from_pretrained(
        config["model"]["name"],
        language=config["model"]["language"],
        task=config["model"]["task"]
    )

    # Load base model
    model = WhisperForConditionalGeneration.from_pretrained(
        config["model"]["name"],
        torch_dtype=torch.float16
    )

    # Try loading LoRA adapters
    try:
        model = PeftModel.from_pretrained(model, model_path)
        print("  LoRA adapters loaded ✓")
    except Exception:
        # Maybe it's a full model save
        try:
            model = WhisperForConditionalGeneration.from_pretrained(
                model_path, torch_dtype=torch.float16
            )
            print("  Full model loaded ✓")
        except Exception as e:
            print(f"  Warning: Could not load adapters from {model_path}: {e}")
            print("  Using base model only.")

    model = model.to(device)
    model.eval()

    return model, processor, device


def transcribe_audio(model, processor, audio_array, sampling_rate, device):
    """Transcribe a single audio array."""
    import torch

    # Process audio to mel spectrogram
    inputs = processor.feature_extractor(
        audio_array,
        sampling_rate=sampling_rate,
        return_tensors="pt"
    )
    input_features = inputs.input_features.to(device, dtype=torch.float16)

    # Generate
    with torch.no_grad():
        predicted_ids = model.generate(
            input_features,
            language="ne",
            task="transcribe",
            max_new_tokens=225
        )

    # Decode
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    return transcription.strip()


def demo_single_file(model, processor, device, audio_path):
    """Demo: transcribe a single audio file."""
    import librosa

    print(f"\n{'='*60}")
    print(f"  Transcribing: {audio_path}")
    print(f"{'='*60}")

    audio, sr = librosa.load(audio_path, sr=16000)
    duration = len(audio) / sr

    print(f"  Duration: {duration:.1f}s")

    text = transcribe_audio(model, processor, audio, sr, device)

    print(f"\n  📝 Transcription:")
    print(f"  {text}")
    print(f"{'='*60}")


def demo_test_dataset(model, processor, device, config, num_samples=5):
    """Demo: transcribe samples from the test dataset."""

    print(f"\n{'='*60}")
    print(f"  Loading test dataset...")
    print(f"{'='*60}")

    dataset = load_common_voice_dataset(
        dataset_name=config["data"]["dataset_name"],
        language=config["data"]["language"],
        sampling_rate=config["data"]["sampling_rate"]
    )

    test = dataset["test"]
    num_samples = min(num_samples, len(test))

    print(f"  Test set size: {len(test)}")
    print(f"  Demoing {num_samples} samples\n")

    correct_chars = 0
    total_chars = 0

    for i in range(num_samples):
        sample = test[i]
        audio = sample["audio"]["array"]
        sr = sample["audio"]["sampling_rate"]
        reference = sample["sentence"]
        duration = len(audio) / sr

        prediction = transcribe_audio(model, processor, audio, sr, device)

        # Simple character accuracy
        total_chars += len(reference)
        correct_chars += sum(1 for a, b in zip(reference, prediction) if a == b)

        print(f"  ┌─ Sample {i+1}/{num_samples} ({duration:.1f}s)")
        print(f"  │  Reference : {reference}")
        print(f"  │  Prediction: {prediction}")

        # Quick match indicator
        if reference.strip() == prediction.strip():
            print(f"  │  Result    : ✅ EXACT MATCH")
        else:
            print(f"  │  Result    : 🔶 Partial match")
        print(f"  └{'─'*55}")
        print()

    # Summary
    char_acc = (correct_chars / total_chars * 100) if total_chars > 0 else 0
    print(f"  {'='*55}")
    print(f"  Summary: {num_samples} samples transcribed")
    print(f"  Approximate Character Accuracy: {char_acc:.1f}%")
    print(f"  {'='*55}")


def main():
    args = parse_args()

    print()
    print("  ╔══════════════════════════════════════════════╗")
    print("  ║   🇳🇵  Nepali ASR Demo (Whisper + LoRA)      ║")
    print("  ╚══════════════════════════════════════════════╝")
    print()

    # Load config
    config = load_config(args.config)

    # Load model
    print("  Loading model...")
    model, processor, device = load_model(args.model, config)
    print("  Model ready!\n")

    if args.audio:
        # Single file mode
        demo_single_file(model, processor, device, args.audio)
    else:
        # Test dataset mode
        demo_test_dataset(model, processor, device, config, args.samples)


if __name__ == "__main__":
    main()
