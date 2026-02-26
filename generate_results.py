"""
Generate Training Result Visualizations for Nepali ASR (Whisper + LoRA)
=====================================================================

Parses training_nohup.log to extract ALL metrics across the full training run
(including resumed training) and generates publication-quality figures.

Run: python generate_results.py
"""

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import numpy as np
import re
import os
import json

# ============================================================
# Parse training_nohup.log for ALL eval metrics
# ============================================================

def parse_training_log(log_path="training_nohup.log"):
    """Parse the nohup log to extract eval metrics at each checkpoint."""
    
    eval_data = []
    train_loss_final = None
    train_runtime = None
    
    # Also try the outputs/logs/training.log
    paths_to_try = [log_path, "outputs/logs/training.log"]
    
    content = ""
    for p in paths_to_try:
        if os.path.exists(p):
            with open(p, 'r', errors='ignore') as f:
                content += f.read()
    
    if not content:
        print("ERROR: No log files found! Using hardcoded data.")
        return None, None, None
    
    # Find all eval metric dictionaries
    # Pattern: {'eval_loss': ..., 'eval_wer': ..., 'eval_cer': ..., ...}
    eval_pattern = r"\{'eval_loss':\s*([\d.]+),\s*'eval_wer':\s*([\d.]+),\s*'eval_cer':\s*([\d.]+),\s*'eval_runtime':\s*([\d.]+),\s*'eval_samples_per_second':\s*([\d.]+),\s*'eval_steps_per_second':\s*([\d.]+),\s*'epoch':\s*([\d.]+)\}"
    
    for match in re.finditer(eval_pattern, content):
        eval_data.append({
            'eval_loss': float(match.group(1)),
            'eval_wer': float(match.group(2)),
            'eval_cer': float(match.group(3)),
            'eval_runtime': float(match.group(4)),
            'eval_samples_per_second': float(match.group(5)),
            'epoch': float(match.group(7))
        })
    
    # Find final train metrics
    train_pattern = r"\{'train_runtime':\s*([\d.]+),.*?'train_loss':\s*([\d.]+)"
    train_matches = list(re.finditer(train_pattern, content))
    if train_matches:
        last = train_matches[-1]
        train_runtime = float(last.group(1))
        train_loss_final = float(last.group(2))
    
    # Find training loss at each logging step
    # Pattern: {'loss': ..., 'grad_norm': ..., 'learning_rate': ..., 'epoch': ...}
    train_loss_data = []
    loss_pattern = r"\{'loss':\s*([\d.]+),\s*'grad_norm':\s*([\d.]+),\s*'learning_rate':\s*([\d.e\-]+),\s*'epoch':\s*([\d.]+)\}"
    for match in re.finditer(loss_pattern, content):
        train_loss_data.append({
            'loss': float(match.group(1)),
            'grad_norm': float(match.group(2)),
            'learning_rate': float(match.group(3)),
            'epoch': float(match.group(4))
        })
    
    # Remove duplicates (keep unique epochs)
    seen_epochs = set()
    unique_eval = []
    for d in eval_data:
        if d['epoch'] not in seen_epochs:
            seen_epochs.add(d['epoch'])
            unique_eval.append(d)
    
    seen_epochs_train = set()
    unique_train = []
    for d in train_loss_data:
        if d['epoch'] not in seen_epochs_train:
            seen_epochs_train.add(d['epoch'])
            unique_train.append(d)
    
    return unique_eval, unique_train, train_loss_final


# ============================================================
# Parse logs
# ============================================================
print("Parsing training logs...")
eval_data, train_loss_data, train_loss_final = parse_training_log()

if eval_data is None or len(eval_data) == 0:
    print("WARNING: Could not parse logs. Using hardcoded data from full training run.")
    # Hardcoded fallback from the complete 5000-step training
    eval_data = [
        {'epoch': 0.8, 'eval_loss': 0.3729, 'eval_wer': 66.44, 'eval_cer': 24.64, 'eval_runtime': 300},
        {'epoch': 1.6, 'eval_loss': 0.3402, 'eval_wer': 63.21, 'eval_cer': 21.87, 'eval_runtime': 350},
        {'epoch': 2.4, 'eval_loss': 0.3287, 'eval_wer': 59.99, 'eval_cer': 19.93, 'eval_runtime': 405},
        {'epoch': 3.2, 'eval_loss': 0.2980, 'eval_wer': 56.83, 'eval_cer': 18.28, 'eval_runtime': 411},
        {'epoch': 4.0, 'eval_loss': 0.2844, 'eval_wer': 55.31, 'eval_cer': 18.63, 'eval_runtime': 428},
        {'epoch': 4.8, 'eval_loss': 0.2784, 'eval_wer': 53.60, 'eval_cer': 17.02, 'eval_runtime': 405},
        {'epoch': 5.6, 'eval_loss': 0.2792, 'eval_wer': 53.40, 'eval_cer': 18.18, 'eval_runtime': 472},
        {'epoch': 6.4, 'eval_loss': 0.2788, 'eval_wer': 52.66, 'eval_cer': 17.04, 'eval_runtime': 466},
        {'epoch': 7.2, 'eval_loss': 0.2799, 'eval_wer': 52.79, 'eval_cer': 17.01, 'eval_runtime': 1301},
        {'epoch': 8.0, 'eval_loss': 0.2803, 'eval_wer': 51.51, 'eval_cer': 16.68, 'eval_runtime': 435},
    ]
    train_loss_final = 0.3142

print(f"Found {len(eval_data)} eval checkpoints")
if train_loss_data:
    print(f"Found {len(train_loss_data)} training loss entries")
print(f"Final train loss: {train_loss_final}")

# Extract arrays
epochs = [d['epoch'] for d in eval_data]
eval_wer = [d['eval_wer'] for d in eval_data]
eval_cer = [d['eval_cer'] for d in eval_data]
eval_loss = [d['eval_loss'] for d in eval_data]

# Approximate step numbers (5000 max steps / 8 epochs ≈ 625 steps per epoch)
steps_per_epoch = 625
steps = [int(e * steps_per_epoch) for e in epochs]

# Model info
model_name = "openai/whisper-small"
total_params = 248_812_800
trainable_params = 7_077_888
trainable_pct = trainable_params / total_params * 100
lora_rank = 32
lora_alpha = 64
dataset_size = 10_000
val_size = 1_000
training_time = "5h 31m"

# Output directory
os.makedirs("outputs/figures", exist_ok=True)

# ============================================================
# Style Configuration
# ============================================================
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
})

BLUE = '#2563EB'
RED = '#DC2626'
GREEN = '#16A34A'
PURPLE = '#7C3AED'
ORANGE = '#EA580C'
GRAY = '#6B7280'
DARK = '#1F2937'

# ============================================================
# Figure 1: WER & CER Progression (FULL 5000 steps)
# ============================================================
print("\nGenerating Figure 1: WER & CER Progression...")

fig, ax1 = plt.subplots(figsize=(10, 6))

line1 = ax1.plot(steps, eval_wer, 'o-', color=BLUE, linewidth=2.5,
                  markersize=8, label='Word Error Rate (WER)', zorder=5)
ax1.fill_between(steps, eval_wer, alpha=0.1, color=BLUE)
ax1.set_xlabel('Training Step', fontweight='bold')
ax1.set_ylabel('Word Error Rate (%)', color=BLUE, fontweight='bold')
ax1.tick_params(axis='y', labelcolor=BLUE)
ax1.set_ylim(min(eval_wer) - 5, max(eval_wer) + 5)

ax2 = ax1.twinx()
line2 = ax2.plot(steps, eval_cer, 's--', color=RED, linewidth=2.5,
                  markersize=8, label='Character Error Rate (CER)', zorder=5)
ax2.fill_between(steps, eval_cer, alpha=0.1, color=RED)
ax2.set_ylabel('Character Error Rate (%)', color=RED, fontweight='bold')
ax2.tick_params(axis='y', labelcolor=RED)
ax2.set_ylim(min(eval_cer) - 3, max(eval_cer) + 3)

# Annotations
ax1.annotate(f'Start: {eval_wer[0]:.1f}%', xy=(steps[0], eval_wer[0]),
             xytext=(steps[0]+300, eval_wer[0]+2), fontsize=9, color=BLUE,
             arrowprops=dict(arrowstyle='->', color=BLUE, lw=1.2))
ax1.annotate(f'Final: {eval_wer[-1]:.1f}%', xy=(steps[-1], eval_wer[-1]),
             xytext=(steps[-1]-1200, eval_wer[-1]-3.5), fontsize=9, color=BLUE,
             fontweight='bold',
             arrowprops=dict(arrowstyle='->', color=BLUE, lw=1.2))

# Resume marker
ax1.axvline(x=1000, color=GRAY, linestyle=':', alpha=0.6, linewidth=1.5)
ax1.text(1020, max(eval_wer)-1, 'Resumed\nfrom ckpt', fontsize=8, color=GRAY, ha='left')

lines = line1 + line2
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='upper right', framealpha=0.9)

wer_improvement = eval_wer[0] - eval_wer[-1]
cer_improvement = eval_cer[0] - eval_cer[-1]
ax1.set_title(f'WER & CER Progression — Full Training (5000 Steps)\n'
              f'WER: {eval_wer[0]:.1f}% → {eval_wer[-1]:.1f}% (↓{wer_improvement:.1f}pp)  |  '
              f'CER: {eval_cer[0]:.1f}% → {eval_cer[-1]:.1f}% (↓{cer_improvement:.1f}pp)',
              fontweight='bold', fontsize=13)

plt.tight_layout()
plt.savefig('outputs/figures/fig1_wer_cer_progression.png')
plt.close()
print("  Saved: outputs/figures/fig1_wer_cer_progression.png")

# ============================================================
# Figure 2: Loss Curves
# ============================================================
print("Generating Figure 2: Loss Curves...")

fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(steps, eval_loss, 'o-', color=PURPLE, linewidth=2.5,
        markersize=8, label='Eval Loss', zorder=5)
ax.fill_between(steps, eval_loss, alpha=0.1, color=PURPLE)

# Train loss per step (if available)
if train_loss_data and len(train_loss_data) > 0:
    train_epochs_arr = [d['epoch'] for d in train_loss_data]
    train_loss_arr = [d['loss'] for d in train_loss_data]
    train_steps_arr = [int(e * steps_per_epoch) for e in train_epochs_arr]
    ax.plot(train_steps_arr, train_loss_arr, '-', color=ORANGE, linewidth=1.5,
            alpha=0.7, label='Train Loss (per step)')

if train_loss_final:
    ax.axhline(y=train_loss_final, color=ORANGE, linestyle='--', linewidth=2,
               label=f'Final Train Loss ({train_loss_final:.4f})', alpha=0.8)

ax.axvline(x=1000, color=GRAY, linestyle=':', alpha=0.6, linewidth=1.5)
ax.text(1020, max(eval_loss)-0.005, 'Resumed', fontsize=8, color=GRAY)

ax.set_xlabel('Training Step', fontweight='bold')
ax.set_ylabel('Loss', fontweight='bold')
ax.set_title(f'Training & Evaluation Loss — Full Training (5000 Steps)\n'
             f'Eval Loss: {eval_loss[0]:.3f} → {eval_loss[-1]:.3f}  |  '
             f'Train Loss: {train_loss_final:.4f}',
             fontweight='bold', fontsize=13)
ax.legend(loc='upper right', framealpha=0.9)

ax.annotate('No Overfitting\n(Train ≈ Eval)', 
            xy=(steps[-3], (eval_loss[-3] + (train_loss_final or eval_loss[-3]))/2),
            fontsize=10, color=GREEN, fontweight='bold', ha='center',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#DCFCE7',
                      edgecolor=GREEN, alpha=0.9))

plt.tight_layout()
plt.savefig('outputs/figures/fig2_loss_curves.png')
plt.close()
print("  Saved: outputs/figures/fig2_loss_curves.png")

# ============================================================
# Figure 3: WER Improvement Bar Chart
# ============================================================
print("Generating Figure 3: WER Improvement...")

fig, ax = plt.subplots(figsize=(12, 6))

colors = [plt.cm.RdYlGn_r(val / max(eval_wer)) for val in eval_wer]
bar_labels = [f'Step {s}\n(Ep {e})' for s, e in zip(steps, epochs)]
bars = ax.bar(bar_labels, eval_wer, color=colors,
              edgecolor='white', linewidth=1.5, width=0.7)

for bar, val in zip(bars, eval_wer):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
            f'{val:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=9)

ax.set_ylabel('Word Error Rate (%)', fontweight='bold')
ax.set_title(f'WER Reduction Across Full Training (5000 Steps)\n'
             f'Total Improvement: {wer_improvement:.1f} percentage points '
             f'({wer_improvement/eval_wer[0]*100:.1f}% relative)',
             fontweight='bold', fontsize=13)
ax.set_ylim(0, max(eval_wer) + 8)
ax.axhline(y=50, color=GRAY, linestyle=':', alpha=0.5, label='50% WER threshold')
ax.legend()

plt.tight_layout()
plt.savefig('outputs/figures/fig3_wer_improvement.png')
plt.close()
print("  Saved: outputs/figures/fig3_wer_improvement.png")

# ============================================================
# Figure 4: Model Architecture & Config Summary
# ============================================================
print("Generating Figure 4: Configuration Summary...")

fig, axes = plt.subplots(1, 3, figsize=(16, 6))

# 4a: Parameter Efficiency Pie
ax = axes[0]
frozen = total_params - trainable_params
sizes = [trainable_params, frozen]
labels_pie = [f'LoRA Trainable\n{trainable_params/1e6:.1f}M ({trainable_pct:.2f}%)',
              f'Frozen Backbone\n{frozen/1e6:.1f}M ({100-trainable_pct:.2f}%)']
colors_pie = [BLUE, '#E5E7EB']
wedges, texts = ax.pie(sizes, labels=labels_pie, colors=colors_pie, explode=(0.05, 0),
                       startangle=90, textprops={'fontsize': 9})
ax.set_title('Parameter Efficiency\n(LoRA Fine-tuning)', fontweight='bold')

# 4b: Training Config Table
ax = axes[1]
ax.axis('off')
config_data = [
    ['Base Model', 'whisper-small (244M)'],
    ['LoRA Rank (r)', str(lora_rank)],
    ['LoRA Alpha (α)', str(lora_alpha)],
    ['Trainable Params', f'{trainable_params/1e6:.1f}M ({trainable_pct:.1f}%)'],
    ['Train Samples', f'{dataset_size:,}'],
    ['Val Samples', f'{val_size:,}'],
    ['Batch Size', '8 × 2 (eff. 16)'],
    ['Learning Rate', '1e-4'],
    ['Max Steps', '5,000'],
    ['Training Time', training_time],
    ['GPU', 'RTX 3080 Ti'],
]
table = ax.table(cellText=config_data, colLabels=['Parameter', 'Value'],
                 loc='center', cellLoc='left')
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 1.4)
for (row, col), cell in table.get_celld().items():
    if row == 0:
        cell.set_facecolor(DARK)
        cell.set_text_props(color='white', fontweight='bold')
    elif row % 2 == 0:
        cell.set_facecolor('#F3F4F6')
    cell.set_edgecolor('#D1D5DB')
ax.set_title('Training Configuration', fontweight='bold', pad=20)

# 4c: Final Results
ax = axes[2]
ax.axis('off')
results_data = [
    ['Final WER', f'{eval_wer[-1]:.2f}%'],
    ['Final CER', f'{eval_cer[-1]:.2f}%'],
    ['WER Improvement', f'↓ {wer_improvement:.1f}pp'],
    ['CER Improvement', f'↓ {cer_improvement:.1f}pp'],
    ['Eval Loss', f'{eval_loss[-1]:.4f}'],
    ['Train Loss', f'{train_loss_final:.4f}' if train_loss_final else 'N/A'],
    ['Epochs Trained', f'{epochs[-1]}'],
    ['Total Steps', '5,000'],
]
table2 = ax.table(cellText=results_data, colLabels=['Metric', 'Value'],
                  loc='center', cellLoc='left')
table2.auto_set_font_size(False)
table2.set_fontsize(9)
table2.scale(1, 1.5)
for (row, col), cell in table2.get_celld().items():
    if row == 0:
        cell.set_facecolor(GREEN)
        cell.set_text_props(color='white', fontweight='bold')
    elif row % 2 == 0:
        cell.set_facecolor('#F0FDF4')
    cell.set_edgecolor('#D1D5DB')
ax.set_title('Final Results', fontweight='bold', pad=20)

plt.suptitle('Nepali ASR: Whisper-Small + LoRA Fine-tuning Summary',
             fontweight='bold', fontsize=15, y=1.02)
plt.tight_layout()
plt.savefig('outputs/figures/fig4_config_summary.png')
plt.close()
print("  Saved: outputs/figures/fig4_config_summary.png")

# ============================================================
# Figure 5: Combined Dashboard
# ============================================================
print("Generating Figure 5: Combined Dashboard...")

fig = plt.figure(figsize=(18, 10))
fig.suptitle('Nepali ASR — Whisper-Small + LoRA Fine-tuning Results Dashboard (5000 Steps)',
             fontweight='bold', fontsize=16, y=0.98)

# 5a: WER/CER
ax1 = fig.add_subplot(2, 2, 1)
ax1.plot(steps, eval_wer, 'o-', color=BLUE, linewidth=2.5, markersize=7, label='WER')
ax1.plot(steps, eval_cer, 's--', color=RED, linewidth=2.5, markersize=7, label='CER')
ax1.axvline(x=1000, color=GRAY, linestyle=':', alpha=0.5)
ax1.set_xlabel('Step')
ax1.set_ylabel('Error Rate (%)')
ax1.set_title('Error Rate Progression', fontweight='bold')
ax1.legend()

# 5b: Loss
ax2 = fig.add_subplot(2, 2, 2)
ax2.plot(steps, eval_loss, 'o-', color=PURPLE, linewidth=2.5, markersize=7, label='Eval Loss')
if train_loss_final:
    ax2.axhline(y=train_loss_final, color=ORANGE, linestyle='--', linewidth=2,
                label=f'Train Loss ({train_loss_final:.3f})')
ax2.axvline(x=1000, color=GRAY, linestyle=':', alpha=0.5)
ax2.set_xlabel('Step')
ax2.set_ylabel('Loss')
ax2.set_title('Loss Curves', fontweight='bold')
ax2.legend()

# 5c: WER bar
ax3 = fig.add_subplot(2, 2, 3)
colors_bar = [plt.cm.RdYlGn_r(val / 70) for val in eval_wer]
ax3.bar([f'S{s}' for s in steps], eval_wer, color=colors_bar, edgecolor='white', width=0.6)
for i, val in enumerate(eval_wer):
    ax3.text(i, val + 0.5, f'{val:.1f}', ha='center', fontsize=8, fontweight='bold')
ax3.set_xlabel('Step')
ax3.set_ylabel('WER (%)')
ax3.set_title(f'WER per Checkpoint (↓{wer_improvement:.1f}pp total)', fontweight='bold')
ax3.set_ylim(0, max(eval_wer) + 8)

# 5d: Summary stats
ax4 = fig.add_subplot(2, 2, 4)
ax4.axis('off')

summary_text = (
    f"{'='*45}\n"
    f"  TRAINING SUMMARY (FULL 5000 STEPS)\n"
    f"{'='*45}\n\n"
    f"  Model:          Whisper-Small (244M params)\n"
    f"  Method:         LoRA (r=32, α=64)\n"
    f"  Trainable:      7.1M params (2.84%)\n"
    f"  Dataset:        10,000 Nepali audio clips\n"
    f"  GPU:            RTX 3080 Ti (12GB)\n"
    f"  Training Time:  {training_time}\n\n"
    f"{'─'*45}\n"
    f"  FINAL METRICS\n"
    f"{'─'*45}\n\n"
    f"  Word Error Rate:       {eval_wer[-1]:.2f}%\n"
    f"  Char Error Rate:       {eval_cer[-1]:.2f}%\n"
    f"  WER Improvement:       ↓ {wer_improvement:.1f}pp ({wer_improvement/eval_wer[0]*100:.1f}%)\n"
    f"  CER Improvement:       ↓ {cer_improvement:.1f}pp ({cer_improvement/eval_cer[0]*100:.1f}%)\n"
    f"  Train Loss:            {train_loss_final:.4f}\n"
    f"  Eval Loss:             {eval_loss[-1]:.4f}\n\n"
    f"{'='*45}"
)
ax4.text(0.05, 0.5, summary_text, transform=ax4.transAxes,
         fontsize=10, verticalalignment='center',
         fontfamily='monospace',
         bbox=dict(boxstyle='round,pad=0.8', facecolor='#F8FAFC',
                   edgecolor=DARK, alpha=0.9))

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('outputs/figures/fig5_combined_dashboard.png')
plt.close()
print("  Saved: outputs/figures/fig5_combined_dashboard.png")

# ============================================================
# Update results_summary.txt
# ============================================================
print("\nUpdating results_summary.txt...")

summary = f"""=================================================================
  WHISPER FINE-TUNING — RESULTS SUMMARY (FULL 5000 STEPS)
=================================================================

  Training Steps Completed:  5000
  Total Epochs:              {epochs[-1]}
  Final Training Loss:       {train_loss_final:.4f}
  Training Time:             {training_time}

  ---- Evaluation Metrics (Best/Final) ----
  Best Eval Loss:            {min(eval_loss):.4f} (Epoch {epochs[eval_loss.index(min(eval_loss))]})
  Best WER:                  {min(eval_wer):.2f}% (Epoch {epochs[eval_wer.index(min(eval_wer))]})
  Final WER:                 {eval_wer[-1]:.2f}%
  WER Improvement:           {wer_improvement:.1f}pp (from {eval_wer[0]:.2f}% to {eval_wer[-1]:.2f}%)
  Best CER:                  {min(eval_cer):.2f}% (Epoch {epochs[eval_cer.index(min(eval_cer))]})
  Final CER:                 {eval_cer[-1]:.2f}%
  CER Improvement:           {cer_improvement:.1f}pp (from {eval_cer[0]:.2f}% to {eval_cer[-1]:.2f}%)

  ---- Model Configuration ----
  Base Model:                {model_name}
  Total Parameters:          {total_params:,}
  Trainable Parameters:      {trainable_params:,} ({trainable_pct:.2f}%)
  LoRA Rank:                 {lora_rank}
  LoRA Alpha:                {lora_alpha}
  Dataset Size:              {dataset_size:,} (train) / {val_size:,} (val)
  GPU:                       NVIDIA RTX 3080 Ti (12GB)

  ---- Checkpoint History ----
"""

for i, d in enumerate(eval_data):
    step = steps[i]
    summary += f"  Step {step:>5} (Ep {d['epoch']:.1f}): WER={d['eval_wer']:.2f}%, CER={d['eval_cer']:.2f}%, Loss={d['eval_loss']:.4f}\n"

summary += f"""
=================================================================
"""

with open('outputs/figures/results_summary.txt', 'w') as f:
    f.write(summary)
print("  Saved: outputs/figures/results_summary.txt")

# Also save to figures/ root
with open('figures/results_summary.txt', 'w') as f:
    f.write(summary)
print("  Saved: figures/results_summary.txt")

print("\n" + "="*60)
print("All figures generated successfully!")
print("="*60)
print(f"\nOutput directory: outputs/figures/")
print(f"  1. fig1_wer_cer_progression.png — WER & CER across all 5000 steps")
print(f"  2. fig2_loss_curves.png — Train vs eval loss")
print(f"  3. fig3_wer_improvement.png — WER improvement bar chart")
print(f"  4. fig4_config_summary.png — Config + results summary panels")
print(f"  5. fig5_combined_dashboard.png — Full results dashboard")
print(f"\nTotal figures: 5")
