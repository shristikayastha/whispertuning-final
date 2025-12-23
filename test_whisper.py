import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration

# Load model from current directory
print("Loading model...")
model = WhisperForConditionalGeneration.from_pretrained("./")
processor = WhisperProcessor.from_pretrained("./")

print("Model loaded successfully!")
print(f"Model type: {model.config.model_type}")
print(f"Model size: {model.config._name_or_path}")

# Test with dummy input
print("\nGenerating test transcription...")
input_features = processor(
    [0] * 16000,  # 1 second of silence at 16kHz
    sampling_rate=16000, 
    return_tensors="pt"
).input_features

with torch.no_grad():
    predicted_ids = model.generate(input_features)
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)

print(f"Test output: {transcription[0]}")
