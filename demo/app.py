"""
Gradio Demo for Nepali ASR
==========================

Professional web interface for demonstrating Nepali speech recognition.
Supports both microphone recording and file upload.

Usage:
    python demo/app.py
    python demo/app.py --model outputs/checkpoints/best_model --share
"""

import argparse
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import gradio as gr
import numpy as np
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from peft import PeftModel
import librosa
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ===== Model Loading =====

def load_model(model_path: str, base_model: str = "openai/whisper-small"):
    """Load the fine-tuned model and processor."""
    logger.info(f"Loading model from: {model_path}")
    
    # Load processor
    processor = WhisperProcessor.from_pretrained(
        base_model,
        language="nepali",
        task="transcribe"
    )
    
    # Check if path exists and has LoRA adapters
    if os.path.exists(model_path) and os.path.exists(os.path.join(model_path, "adapter_config.json")):
        logger.info("Loading LoRA model")
        base = WhisperForConditionalGeneration.from_pretrained(
            base_model,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        model = PeftModel.from_pretrained(base, model_path)
        model = model.merge_and_unload()
    elif os.path.exists(model_path):
        logger.info("Loading full model")
        model = WhisperForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
    else:
        logger.warning(f"Model path not found: {model_path}, using base model")
        model = WhisperForConditionalGeneration.from_pretrained(
            base_model,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
    
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    
    # Configure for Nepali
    model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(
        language="nepali",
        task="transcribe"
    )
    
    logger.info(f"Model loaded on {device}")
    
    return model, processor, device


# ===== Transcription Function =====

def transcribe_audio(audio, model, processor, device):
    """
    Transcribe audio to Nepali text.
    
    Args:
        audio: Tuple of (sample_rate, audio_array) or file path
        model: Whisper model
        processor: Whisper processor
        device: Device to use
    
    Returns:
        Transcribed text
    """
    if audio is None:
        return "⚠️ कृपया अडियो रेकर्ड गर्नुहोस् वा फाइल अपलोड गर्नुहोस्।"
    
    try:
        # Handle different input types
        if isinstance(audio, tuple):
            # Microphone input: (sample_rate, audio_array)
            sr, audio_array = audio
            audio_array = audio_array.astype(np.float32)
            
            # Normalize if needed
            if audio_array.max() > 1.0:
                audio_array = audio_array / 32768.0
            
            # Convert stereo to mono
            if len(audio_array.shape) > 1:
                audio_array = audio_array.mean(axis=1)
            
            # Resample to 16kHz
            if sr != 16000:
                audio_array = librosa.resample(audio_array, orig_sr=sr, target_sr=16000)
        else:
            # File input: path string
            audio_array, sr = librosa.load(audio, sr=16000, mono=True)
        
        # Extract features
        inputs = processor(
            audio_array,
            sampling_rate=16000,
            return_tensors="pt"
        )
        input_features = inputs.input_features.to(device)
        
        if torch.cuda.is_available():
            input_features = input_features.half()
        
        # Generate transcription
        with torch.no_grad():
            generated_ids = model.generate(
                input_features,
                max_new_tokens=225,
                num_beams=5
            )
        
        # Decode
        transcription = processor.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )[0]
        
        return transcription.strip() if transcription.strip() else "🔇 कुनै आवाज पत्ता लागेन।"
    
    except Exception as e:
        logger.error(f"Transcription error: {e}")
        return f"❌ त्रुटि: {str(e)}"


# ===== Demo Interface =====

def create_demo(model, processor, device):
    """Create the Gradio demo interface."""
    
    def process_microphone(audio):
        return transcribe_audio(audio, model, processor, device)
    
    def process_file(audio_file):
        if audio_file is None:
            return "⚠️ कृपया अडियो फाइल अपलोड गर्नुहोस्।"
        return transcribe_audio(audio_file, model, processor, device)
    
    # Custom CSS for beautiful UI
    custom_css = """
    .gradio-container {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }
    
    .main-title {
        text-align: center;
        padding: 20px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 16px;
        color: white;
        margin-bottom: 24px;
    }
    
    .main-title h1 {
        font-size: 2.5rem;
        margin: 0;
        font-weight: 700;
    }
    
    .main-title p {
        font-size: 1.1rem;
        opacity: 0.9;
        margin: 8px 0 0 0;
    }
    
    .feature-box {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 16px;
        border-radius: 12px;
        margin: 10px 0;
    }
    
    footer {
        text-align: center;
        padding: 20px;
        color: #666;
    }
    """
    
    with gr.Blocks(
        title="🇳🇵 Nepali ASR",
        css=custom_css,
        theme=gr.themes.Soft(
            primary_hue="purple",
            secondary_hue="blue"
        )
    ) as demo:
        
        # Header
        gr.HTML("""
            <div class="main-title">
                <h1>🇳🇵 नेपाली वाक् पहिचान</h1>
                <p>Nepali Speech Recognition | Fine-tuned Whisper Model</p>
            </div>
        """)
        
        gr.Markdown("""
        ### 📋 कसरी प्रयोग गर्ने (How to Use)
        
        1. **🎤 माइक्रोफोन** - रेकर्ड बटन थिच्नुहोस् र नेपालीमा बोल्नुहोस्
        2. **📁 फाइल अपलोड** - अडियो फाइल (MP3, WAV, FLAC) अपलोड गर्नुहोस्
        """)
        
        with gr.Tabs():
            
            # Tab 1: Microphone Recording
            with gr.TabItem("🎤 माइक्रोफोन (Microphone)"):
                with gr.Row():
                    with gr.Column(scale=1):
                        mic_input = gr.Audio(
                            sources=["microphone"],
                            type="numpy",
                            label="🎙️ रेकर्डिङ (Recording)",
                            elem_id="mic-input"
                        )
                        mic_button = gr.Button(
                            "✨ ट्रान्सक्राइब गर्नुहोस् (Transcribe)",
                            variant="primary",
                            size="lg"
                        )
                    
                    with gr.Column(scale=1):
                        mic_output = gr.Textbox(
                            label="📝 नेपाली पाठ (Nepali Text)",
                            lines=5,
                            placeholder="ट्रान्सक्रिप्सन यहाँ देखिनेछ...",
                            elem_id="mic-output"
                        )
                
                mic_button.click(
                    fn=process_microphone,
                    inputs=mic_input,
                    outputs=mic_output
                )
            
            # Tab 2: File Upload
            with gr.TabItem("📁 फाइल अपलोड (File Upload)"):
                with gr.Row():
                    with gr.Column(scale=1):
                        file_input = gr.Audio(
                            sources=["upload"],
                            type="filepath",
                            label="📂 अडियो फाइल (Audio File)",
                            elem_id="file-input"
                        )
                        file_button = gr.Button(
                            "✨ ट्रान्सक्राइब गर्नुहोस् (Transcribe)",
                            variant="primary",
                            size="lg"
                        )
                    
                    with gr.Column(scale=1):
                        file_output = gr.Textbox(
                            label="📝 नेपाली पाठ (Nepali Text)",
                            lines=5,
                            placeholder="ट्रान्सक्रिप्सन यहाँ देखिनेछ...",
                            elem_id="file-output"
                        )
                
                file_button.click(
                    fn=process_file,
                    inputs=file_input,
                    outputs=file_output
                )
        
        # Model Info
        with gr.Accordion("ℹ️ मोडेल जानकारी (Model Information)", open=False):
            device_info = "GPU (CUDA)" if device == "cuda" else "CPU"
            gr.Markdown(f"""
            | विशेषता | मान |
            |---------|-----|
            | **आधार मोडेल** | OpenAI Whisper Small |
            | **Fine-tuning** | LoRA (Low-Rank Adaptation) |
            | **Dataset** | Mozilla Common Voice (Nepali) |
            | **Device** | {device_info} |
            | **भाषा** | नेपाली (Nepali) |
            """)
        
        # Footer
        gr.HTML("""
            <footer>
                <p>🎓 Final Year Project | Nepali Speech Recognition using Whisper</p>
                <p style="font-size: 0.9em; opacity: 0.7;">Powered by OpenAI Whisper & HuggingFace Transformers</p>
            </footer>
        """)
    
    return demo


def main():
    """Main function to run the demo."""
    parser = argparse.ArgumentParser(description="Nepali ASR Demo")
    parser.add_argument(
        "--model",
        type=str,
        default="./outputs/checkpoints/best_model",
        help="Path to trained model"
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="openai/whisper-small",
        help="Base Whisper model name"
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create public shareable link"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port to run the demo on"
    )
    
    args = parser.parse_args()
    
    # Load model
    model, processor, device = load_model(args.model, args.base_model)
    
    # Create and launch demo
    demo = create_demo(model, processor, device)
    
    logger.info(f"Launching demo on port {args.port}")
    if args.share:
        logger.info("Creating public shareable link...")
    
    demo.launch(
        server_port=args.port,
        share=args.share,
        show_error=True
    )


if __name__ == "__main__":
    main()
