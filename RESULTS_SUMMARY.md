# Nepali ASR: Whisper-Small + LoRA Fine-tuning — Comprehensive Summary

> **Project**: Fine-tuning OpenAI's Whisper-Small model for Nepali Speech Recognition using LoRA  
> **Author**: Ashish Pandey | Khwopa College of Engineering  
> **Date**: February 26, 2026  
> **GPU**: NVIDIA RTX 3080 Ti (12GB VRAM)

---

## 1. Project Overview

This project fine-tunes OpenAI's **Whisper-Small** (244M parameters) for **Nepali automatic speech recognition (ASR)** using **Low-Rank Adaptation (LoRA)**. Instead of training all 244M parameters, LoRA freezes the base model and injects small trainable matrices into the attention layers — training only **2.84% of total parameters** while achieving strong performance.

### Why This Matters
- **Nepali is a low-resource language** — few ASR systems exist for it
- **LoRA makes fine-tuning practical** — can be done on a single GPU in hours, not days
- **Whisper provides strong multilingual foundation** — pre-trained on 680,000 hours of audio

---

## 2. Model Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Whisper-Small + LoRA                      │
│                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐   │
│  │   Audio      │───▶│   Encoder    │───▶│   LoRA       │   │
│  │   Input      │    │   (Frozen)   │    │  Adapters    │   │
│  │   16kHz      │    │              │    │  r=32, α=64  │   │
│  └──────────────┘    └──────────────┘    └──────────────┘   │
│                             │                    │           │
│                             ▼                    ▼           │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐   │
│  │   Nepali     │◀───│   Decoder    │◀───│    LoRA      │   │
│  │    Text      │    │   (Frozen)   │    │  Adapters    │   │
│  └──────────────┘    └──────────────┘    └──────────────┘   │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

| Component | Value |
|-----------|-------|
| Base Model | `openai/whisper-small` (244M params) |
| Trainable Params | 7.1M (2.84% of total) |
| LoRA Rank (r) | 32 |
| LoRA Alpha (α) | 64 |
| Target Modules | q_proj, v_proj, k_proj, out_proj |
| LoRA Dropout | 0.1 |

---

## 3. Dataset

| Split | Samples | Source |
|-------|---------|--------|
| Train | 10,000 | `spktsagar/openslr-nepali-asr-cleaned` |
| Validation | 1,000 | (5% split) |
| Test | 1,000 | (5% split) |
| Full Dataset | 142,114 | Available for future training |

- Audio resampled to 16kHz (Whisper requirement)
- Max clip length: 30 seconds
- Language: Nepali (ne-NP)

---

## 4. Training Configuration

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| Learning Rate | 1e-4 |
| Batch Size | 8 × 2 (effective 16) |
| Max Steps | 5,000 |
| Warmup Steps | 500 |
| FP16 | Enabled |
| Eval Strategy | Every 500 steps |
| Gradient Checkpointing | Enabled |
| Save Strategy | Best model (lowest WER) |

---

## 5. Results

### Final Metrics

| Metric | Value |
|--------|-------|
| **Word Error Rate (WER)** | **51.51%** |
| **Character Error Rate (CER)** | **16.68%** |
| **Eval Loss** | 0.2803 |
| **Train Loss** | 0.3142 |
| **Total Training Time** | 5 hours 31 minutes |
| **Epochs Completed** | 8.0 |

### Training Progression

| Epoch | WER (%) | CER (%) | Eval Loss |
|-------|---------|---------|-----------|
| 2.4 | 59.99 | 19.93 | 0.329 |
| 3.2 | 56.83 | 18.28 | 0.298 |
| 4.0 | 55.31 | 18.63 | 0.284 |
| 4.8 | 53.60 | 17.02 | 0.278 |
| 5.6 | 53.40 | 18.18 | 0.279 |
| 6.4 | 52.66 | 17.04 | 0.279 |
| 7.2 | 52.79 | 17.01 | 0.280 |
| **8.0** | **51.51** | **16.68** | **0.280** |

### Key Observations

1. **WER improved by 8.5 percentage points** (59.99% → 51.51%), a **14.1% relative improvement**
2. **CER improved by 3.3 percentage points** (19.93% → 16.68%), a **16.3% relative improvement**
3. **No overfitting detected**: Train loss (0.314) and eval loss (0.280) remain close throughout
4. **Loss plateau after epoch 5**: Eval loss stabilizes at ~0.280, suggesting the model has extracted maximum information from the 10k subset
5. **Consistent WER decline**: Every evaluation checkpoint showed improvement except a minor 0.13pp fluctuation at epoch 7.2

---

## 6. Efficiency Analysis

| Metric | Value |
|--------|-------|
| Parameters Trained | 7.1M / 248.8M (2.84%) |
| Memory Saved | ~97% (vs full fine-tuning) |
| Training Speed | ~2.85 seconds/step |
| Throughput | 4.03 samples/second |
| GPU Memory Used | ~11.8 GB |
| Total FLOPS | 22,260,662,841 GF |

### Why LoRA?
- **97% fewer trainable parameters** → faster training, less memory
- **Frozen backbone** → prevents catastrophic forgetting of Whisper's multilingual knowledge
- **Modular adapters** → can swap LoRA weights for different languages without retraining base model

---

## 7. Generated Figures

All figures are saved in `outputs/figures/`:

| Figure | Description |
|--------|-------------|
| `fig1_wer_cer_progression.png` | WER & CER dual-axis line chart showing training progression |
| `fig2_loss_curves.png` | Train vs eval loss curves with overfitting analysis |
| `fig3_wer_improvement.png` | WER improvement bar chart across epochs |
| `fig4_config_summary.png` | Three-panel summary with parameter pie chart, config table, and results |
| `fig5_combined_dashboard.png` | Complete results dashboard combining all metrics |

---

## 8. Future Improvements

1. **Full dataset training** (142k samples) — expected to reduce WER to ~30-35%
2. **Increase max_steps** to 15,000-20,000 for larger dataset
3. **Data augmentation** — speed perturbation, noise injection
4. **Beam search tuning** — optimize beam width for better decoding
5. **Language model integration** — add Nepali n-gram LM for post-processing

---

## 9. How to Reproduce

```bash
# Install dependencies
pip install -r requirements.txt

# Train from scratch
python train.py

# Resume from checkpoint
python train.py --resume outputs/checkpoints/checkpoint-1000

# Generate result figures
python generate_results.py

# Run inference
python evaluate.py
```

---

## 10. Repository Structure

```
whispertuning-final/
├── train.py                  # Main training script
├── evaluate.py               # Evaluation & inference
├── generate_results.py       # Results visualization
├── test_whisper.py           # Quick sanity test
├── configs/
│   └── config.yaml           # All hyperparameters
├── src/
│   ├── data/
│   │   ├── data_loader.py    # Dataset loading & preprocessing
│   │   └── data_collator.py  # Batch collation for Seq2Seq
│   ├── models/
│   │   ├── whisper_model.py  # Model loading & LoRA setup
│   │   └── tokenizer.py      # Processor/tokenizer loading
│   ├── training/
│   │   ├── trainer.py        # Trainer configuration
│   │   └── metrics.py        # WER/CER computation
│   └── utils/
│       └── helpers.py        # Utilities (config, logging, etc.)
├── outputs/
│   ├── checkpoints/          # Model checkpoints
│   ├── figures/              # Generated visualizations
│   └── logs/                 # Training logs
└── requirements.txt
```
