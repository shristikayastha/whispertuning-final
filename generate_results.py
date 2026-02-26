"""
Generate Training Result Visualizations for Nepali ASR (Whisper + LoRA)
=====================================================================

Creates publication-quality figures from training metrics.
Run: python generate_results.py
"""

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server
import numpy as np
import os

# ============================================================
# Training Data (from training_nohup.log)
# ============================================================

epochs = [2.4, 3.2, 4.0, 4.8, 5.6, 6.4, 7.2, 8.0]

eval_wer = [59.99, 56.83, 55.31, 53.60, 53.40, 52.66, 52.79, 51.51]
eval_cer = [19.93, 18.28, 18.63, 17.02, 18.18, 17.04, 17.01, 16.68]
eval_loss = [0.329, 0.298, 0.284, 0.278, 0.279, 0.279, 0.280, 0.280]

# Final train loss
train_loss_final = 0.3142

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
gpu = "NVIDIA RTX 3080 Ti (12GB)"

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

# Color palette
BLUE = '#2563EB'
RED = '#DC2626'
GREEN = '#16A34A'
PURPLE = '#7C3AED'
ORANGE = '#EA580C'
GRAY = '#6B7280'
DARK = '#1F2937'

# ============================================================
# Figure 1: WER & CER Progression
# ============================================================
print("Generating Figure 1: WER & CER Progression...")

fig, ax1 = plt.subplots(figsize=(10, 6))

# WER on primary axis
line1 = ax1.plot(epochs, eval_wer, 'o-', color=BLUE, linewidth=2.5, 
                  markersize=8, label='Word Error Rate (WER)', zorder=5)
ax1.fill_between(epochs, eval_wer, alpha=0.1, color=BLUE)
ax1.set_xlabel('Epoch', fontweight='bold')
ax1.set_ylabel('Word Error Rate (%)', color=BLUE, fontweight='bold')
ax1.tick_params(axis='y', labelcolor=BLUE)
ax1.set_ylim(45, 65)

# CER on secondary axis
ax2 = ax1.twinx()
line2 = ax2.plot(epochs, eval_cer, 's--', color=RED, linewidth=2.5,
                  markersize=8, label='Character Error Rate (CER)', zorder=5)
ax2.fill_between(epochs, eval_cer, alpha=0.1, color=RED)
ax2.set_ylabel('Character Error Rate (%)', color=RED, fontweight='bold')
ax2.tick_params(axis='y', labelcolor=RED)
ax2.set_ylim(14, 22)

# Annotations
ax1.annotate(f'Start: {eval_wer[0]:.1f}%', xy=(epochs[0], eval_wer[0]),
             xytext=(epochs[0]+0.3, eval_wer[0]+2), fontsize=9, color=BLUE,
             arrowprops=dict(arrowstyle='->', color=BLUE, lw=1.2))
ax1.annotate(f'Final: {eval_wer[-1]:.1f}%', xy=(epochs[-1], eval_wer[-1]),
             xytext=(epochs[-1]-1.5, eval_wer[-1]-3), fontsize=9, color=BLUE,
             fontweight='bold',
             arrowprops=dict(arrowstyle='->', color=BLUE, lw=1.2))

# Combined legend
lines = line1 + line2
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='upper right', framealpha=0.9)

# Title with improvement stats
wer_improvement = eval_wer[0] - eval_wer[-1]
cer_improvement = eval_cer[0] - eval_cer[-1]
ax1.set_title(f'WER & CER Progression During Training\n'
              f'WER: {eval_wer[0]:.1f}% → {eval_wer[-1]:.1f}% (↓{wer_improvement:.1f}pp)  |  '
              f'CER: {eval_cer[0]:.1f}% → {eval_cer[-1]:.1f}% (↓{cer_improvement:.1f}pp)',
              fontweight='bold', fontsize=13)

plt.tight_layout()
plt.savefig('outputs/figures/fig1_wer_cer_progression.png')
plt.close()
print("  Saved: outputs/figures/fig1_wer_cer_progression.png")

# ============================================================
# Figure 2: Loss Curve
# ============================================================
print("Generating Figure 2: Loss Curves...")

fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(epochs, eval_loss, 'o-', color=PURPLE, linewidth=2.5, 
        markersize=8, label='Eval Loss', zorder=5)
ax.fill_between(epochs, eval_loss, alpha=0.1, color=PURPLE)

# Train loss reference line
ax.axhline(y=train_loss_final, color=ORANGE, linestyle='--', linewidth=2, 
           label=f'Final Train Loss ({train_loss_final:.4f})', alpha=0.8)

ax.set_xlabel('Epoch', fontweight='bold')
ax.set_ylabel('Loss', fontweight='bold')
ax.set_title('Training & Evaluation Loss\n'
             f'Eval Loss: {eval_loss[0]:.3f} → {eval_loss[-1]:.3f}  |  '
             f'Train Loss: {train_loss_final:.4f}',
             fontweight='bold', fontsize=13)
ax.legend(loc='upper right', framealpha=0.9)
ax.set_ylim(0.25, 0.36)

# No overfitting annotation
ax.annotate('No Overfitting\n(Train ≈ Eval)', 
            xy=(6.0, 0.30), fontsize=10, color=GREEN,
            fontweight='bold', ha='center',
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

fig, ax = plt.subplots(figsize=(10, 6))

colors = [plt.cm.RdYlGn_r(val/max(eval_wer)) for val in eval_wer]
bars = ax.bar([f'Ep {e}' for e in epochs], eval_wer, color=colors, 
              edgecolor='white', linewidth=1.5, width=0.7)

# Value labels on bars
for bar, val in zip(bars, eval_wer):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
            f'{val:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=10)

ax.set_ylabel('Word Error Rate (%)', fontweight='bold')
ax.set_title(f'WER Reduction Across Training Epochs\n'
             f'Total Improvement: {wer_improvement:.1f} percentage points ({wer_improvement/eval_wer[0]*100:.1f}% relative)',
             fontweight='bold', fontsize=13)
ax.set_ylim(0, 68)
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
labels = [f'LoRA Trainable\n{trainable_params/1e6:.1f}M ({trainable_pct:.2f}%)',
          f'Frozen Backbone\n{frozen/1e6:.1f}M ({100-trainable_pct:.2f}%)']
colors_pie = [BLUE, '#E5E7EB']
explode = (0.05, 0)
wedges, texts = ax.pie(sizes, labels=labels, colors=colors_pie, explode=explode,
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
    ['Train Loss', f'{train_loss_final:.4f}'],
    ['Epochs Trained', '8.0'],
    ['Best Epoch (WER)', '8.0'],
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
fig.suptitle('Nepali ASR — Whisper-Small + LoRA Fine-tuning Results Dashboard',
             fontweight='bold', fontsize=16, y=0.98)

# 5a: WER/CER (top left)
ax1 = fig.add_subplot(2, 2, 1)
ax1.plot(epochs, eval_wer, 'o-', color=BLUE, linewidth=2.5, markersize=7, label='WER')
ax1.plot(epochs, eval_cer, 's--', color=RED, linewidth=2.5, markersize=7, label='CER')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Error Rate (%)')
ax1.set_title(f'Error Rate Progression', fontweight='bold')
ax1.legend()
ax1.set_ylim(10, 65)

# 5b: Loss (top right)
ax2 = fig.add_subplot(2, 2, 2)
ax2.plot(epochs, eval_loss, 'o-', color=PURPLE, linewidth=2.5, markersize=7, label='Eval Loss')
ax2.axhline(y=train_loss_final, color=ORANGE, linestyle='--', linewidth=2, 
            label=f'Train Loss ({train_loss_final:.3f})')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')
ax2.set_title('Loss Curves', fontweight='bold')
ax2.legend()

# 5c: WER bar (bottom left)
ax3 = fig.add_subplot(2, 2, 3)
colors_bar = [plt.cm.RdYlGn_r(val/65) for val in eval_wer]
ax3.bar([f'{e}' for e in epochs], eval_wer, color=colors_bar, edgecolor='white', width=0.6)
for i, val in enumerate(eval_wer):
    ax3.text(i, val + 0.5, f'{val:.1f}', ha='center', fontsize=9, fontweight='bold')
ax3.set_xlabel('Epoch')
ax3.set_ylabel('WER (%)')
ax3.set_title(f'WER per Epoch (↓{wer_improvement:.1f}pp total)', fontweight='bold')
ax3.set_ylim(0, 68)

# 5d: Summary stats (bottom right)
ax4 = fig.add_subplot(2, 2, 4)
ax4.axis('off')

summary_text = (
    f"{'='*45}\n"
    f"  TRAINING SUMMARY\n"
    f"{'='*45}\n\n"
    f"  Model:          Whisper-Small (244M params)\n"
    f"  Method:         LoRA (r=32, α=64)\n"
    f"  Trainable:      7.1M params (2.84%)\n"
    f"  Dataset:        10,000 Nepali audio clips\n"
    f"  GPU:            RTX 3080 Ti (12GB)\n"
    f"  Training Time:  5h 31m\n\n"
    f"{'─'*45}\n"
    f"  FINAL METRICS\n"
    f"{'─'*45}\n\n"
    f"  Word Error Rate:       51.51%\n"
    f"  Char Error Rate:       16.68%\n"
    f"  WER Improvement:       ↓ 8.5 pp (14.1%)\n"
    f"  CER Improvement:       ↓ 3.3 pp (16.3%)\n"
    f"  Train Loss:            0.3142\n"
    f"  Eval Loss:             0.2803\n\n"
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

print("\n" + "="*60)
print("All figures generated successfully!")
print("="*60)
print(f"\nOutput directory: outputs/figures/")
print(f"  1. fig1_wer_cer_progression.png")
print(f"  2. fig2_loss_curves.png")
print(f"  3. fig3_wer_improvement.png")
print(f"  4. fig4_config_summary.png")
print(f"  5. fig5_combined_dashboard.png")
print(f"\nTotal figures: 5")
