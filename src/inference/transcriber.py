"""
Nepali Speech Transcriber
=========================

Production-ready inference pipeline for Nepali ASR using
the fine-tuned Whisper model.
"""

import os
import torch
import numpy as np
import librosa
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    pipeline
)
from peft import PeftModel
from typing import Optional, Union, List, Dict
import logging

logger = logging.getLogger(__name__)


class NepaliTranscriber:
    """
    End-to-end Nepali speech transcription using fine-tuned Whisper.
    
    This class provides a simple interface for transcribing Nepali
    audio files or raw audio arrays.
    
    Attributes:
        model: The fine-tuned Whisper model
        processor: Whisper processor for feature extraction and decoding
        device: Device to run inference on
    """
    
    def __init__(
        self,
        model_path: str,
        base_model_name: str = "openai/whisper-small",
        device: Optional[str] = None,
        use_lora: bool = True,
        torch_dtype: torch.dtype = torch.float16
    ):
        """
        Initialize the transcriber.
        
        Args:
            model_path: Path to the fine-tuned model or LoRA adapters
            base_model_name: Base Whisper model name
            device: Device to use ("cuda", "cpu", or None for auto)
            use_lora: Whether the model uses LoRA adapters
            torch_dtype: Data type for model weights
        """
        # Set device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        logger.info(f"Initializing NepaliTranscriber on {self.device}")
        
        self.torch_dtype = torch_dtype
        self.sampling_rate = 16000
        
        # Load processor
        self.processor = WhisperProcessor.from_pretrained(
            base_model_name,
            language="nepali",
            task="transcribe"
        )
        
        # Load model
        if use_lora and os.path.exists(os.path.join(model_path, "adapter_config.json")):
            logger.info("Loading model with LoRA adapters")
            base_model = WhisperForConditionalGeneration.from_pretrained(
                base_model_name,
                torch_dtype=torch_dtype
            )
            self.model = PeftModel.from_pretrained(base_model, model_path)
            self.model = self.model.merge_and_unload()  # Merge for faster inference
        else:
            logger.info("Loading full model")
            self.model = WhisperForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=torch_dtype
            )
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Configure generation
        self.model.config.forced_decoder_ids = self.processor.get_decoder_prompt_ids(
            language="nepali",
            task="transcribe"
        )
        
        logger.info("Transcriber initialized successfully")
    
    def load_audio(
        self,
        audio_path: str,
        sampling_rate: int = 16000
    ) -> np.ndarray:
        """
        Load audio from file.
        
        Args:
            audio_path: Path to audio file
            sampling_rate: Target sampling rate
        
        Returns:
            Audio waveform as numpy array
        """
        audio, sr = librosa.load(audio_path, sr=sampling_rate, mono=True)
        return audio
    
    def preprocess(
        self,
        audio: Union[str, np.ndarray],
        sampling_rate: Optional[int] = None
    ) -> torch.Tensor:
        """
        Preprocess audio for model input.
        
        Args:
            audio: Audio file path or numpy array
            sampling_rate: Sampling rate (required if audio is array)
        
        Returns:
            Input features tensor
        """
        if isinstance(audio, str):
            audio = self.load_audio(audio, self.sampling_rate)
            sampling_rate = self.sampling_rate
        else:
            sampling_rate = sampling_rate or self.sampling_rate
        
        # Extract features
        inputs = self.processor(
            audio,
            sampling_rate=sampling_rate,
            return_tensors="pt"
        )
        
        return inputs.input_features.to(self.device, dtype=self.torch_dtype)
    
    @torch.no_grad()
    def transcribe(
        self,
        audio: Union[str, np.ndarray],
        sampling_rate: Optional[int] = None,
        return_timestamps: bool = False
    ) -> str:
        """
        Transcribe audio to Nepali text.
        
        Args:
            audio: Audio file path or numpy array
            sampling_rate: Sampling rate (required if audio is array)
            return_timestamps: Whether to return timestamps
        
        Returns:
            Transcribed Nepali text
        """
        # Preprocess
        input_features = self.preprocess(audio, sampling_rate)
        
        # Generate
        generated_ids = self.model.generate(
            input_features,
            max_new_tokens=225,
            num_beams=5,
            return_timestamps=return_timestamps
        )
        
        # Decode
        transcription = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )[0]
        
        return transcription.strip()
    
    def transcribe_batch(
        self,
        audio_paths: List[str],
        batch_size: int = 4
    ) -> List[str]:
        """
        Transcribe multiple audio files.
        
        Args:
            audio_paths: List of paths to audio files
            batch_size: Number of files to process at once
        
        Returns:
            List of transcriptions
        """
        transcriptions = []
        
        for i in range(0, len(audio_paths), batch_size):
            batch_paths = audio_paths[i:i + batch_size]
            
            # Load and preprocess batch
            batch_audio = [self.load_audio(p) for p in batch_paths]
            inputs = self.processor(
                batch_audio,
                sampling_rate=self.sampling_rate,
                return_tensors="pt",
                padding=True
            )
            input_features = inputs.input_features.to(self.device, dtype=self.torch_dtype)
            
            # Generate
            with torch.no_grad():
                generated_ids = self.model.generate(
                    input_features,
                    max_new_tokens=225,
                    num_beams=5
                )
            
            # Decode
            batch_transcriptions = self.processor.batch_decode(
                generated_ids,
                skip_special_tokens=True
            )
            transcriptions.extend([t.strip() for t in batch_transcriptions])
        
        return transcriptions
    
    def get_model_info(self) -> Dict:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary with model information
        """
        return {
            "model_type": self.model.config.model_type,
            "device": str(self.device),
            "dtype": str(self.torch_dtype),
            "num_parameters": sum(p.numel() for p in self.model.parameters()),
            "language": "nepali"
        }


def create_asr_pipeline(
    model_path: str,
    base_model_name: str = "openai/whisper-small",
    device: Optional[str] = None
):
    """
    Create a HuggingFace ASR pipeline for quick inference.
    
    Args:
        model_path: Path to fine-tuned model
        base_model_name: Base model name
        device: Device to use
    
    Returns:
        HuggingFace ASR pipeline
    """
    if device is None:
        device = 0 if torch.cuda.is_available() else -1
    elif device == "cuda":
        device = 0
    elif device == "cpu":
        device = -1
    
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model_path,
        tokenizer=WhisperProcessor.from_pretrained(base_model_name).tokenizer,
        feature_extractor=WhisperProcessor.from_pretrained(base_model_name).feature_extractor,
        device=device,
        torch_dtype=torch.float16
    )
    
    return pipe


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Example usage (requires a trained model)
    print("NepaliTranscriber module loaded successfully!")
    print("\nExample usage:")
    print("  transcriber = NepaliTranscriber('./outputs/checkpoints/best_model')")
    print("  text = transcriber.transcribe('audio.wav')")
    print("  print(text)")
