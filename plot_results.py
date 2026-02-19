"""
Whisper Fine-Tuning Results Visualization
==========================================

Generates ALL important figures from training logs.
Reads trainer_state.json from the checkpoint directory.

Usage:
    python plot_results.py                          # Auto-detect checkpoint dir
    python plot_results.py --checkpoint_dir ./outputs/checkpoints
    python plot_results.py --output_dir ./figures    # Custom output dir
"""

import json
import os
import argparse
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for servers
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from pathlib import Path

# ============================================================
# Style Configuration - Publication Quality
# ============================================================
plt.rcParams.update({
    'figure.figsize': (10, 6),
    'figure.dpi': 150,
    'font.size': 12,
    'font.family': 'sans-serif',
    'axes.titlesize': 16,
    'axes.titleweight': 'bold',
    'axes.labelsize': 14,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
    'legend.fontsize': 11,
    'legend.framealpha': 0.9,
    'lines.linewidth': 2.0,
    'lines.markersize': 8,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.2,
})

# Color palette (professional, colorblind-friendly)
COLORS = {
    'train_loss': '#2196F3',     # Blue
    'eval_loss': '#FF5722',      # Deep Orange
    'wer': '#4CAF50',            # Green
    'cer': '#9C27B0',            # Purple
    'lr': '#FF9800',             # Orange
    'baseline': '#E91E63',       # Pink
    'finetuned': '#00BCD4',      # Cyan
    'fill': '#BBDEFB',           # Light Blue
    'fill2': '#FFCCBC',          # Light Orange
}


def load_trainer_state(checkpoint_dir):
    """Load trainer_state.json from checkpoint directory."""
    
    # Try multiple locations
    candidates = [
        os.path.join(checkpoint_dir, "trainer_state.json"),
    ]
    
    # Also check inside checkpoint subdirectories
    if os.path.isdir(checkpoint_dir):
        for d in sorted(os.listdir(checkpoint_dir)):
            if d.startswith("checkpoint-"):
                candidates.append(
                    os.path.join(checkpoint_dir, d, "trainer_state.json")
                )
    
    for path in candidates:
        if os.path.exists(path):
            print(f"[✓] Found trainer_state.json at: {path}")
            with open(path, 'r') as f:
                return json.load(f)
    
    raise FileNotFoundError(
        f"Could not find trainer_state.json in {checkpoint_dir}\n"
        f"Searched: {candidates}"
    )


def extract_metrics(state):
    """Extract all metrics from trainer state log history."""
    log_history = state.get("log_history", [])
    
    # Separate training logs from eval logs
    train_logs = []
    eval_logs = []
    
    for entry in log_history:
        if "loss" in entry and "eval_loss" not in entry:
            train_logs.append(entry)
        if "eval_loss" in entry:
            eval_logs.append(entry)
    
    # Extract training metrics
    train_steps = [e["step"] for e in train_logs]
    train_loss = [e["loss"] for e in train_logs]
    train_lr = [e.get("learning_rate", 0) for e in train_logs]
    train_epoch = [e.get("epoch", 0) for e in train_logs]
    
    # Extract eval metrics
    eval_steps = [e["step"] for e in eval_logs]
    eval_loss = [e["eval_loss"] for e in eval_logs]
    eval_wer = [e.get("eval_wer", None) for e in eval_logs]
    eval_cer = [e.get("eval_cer", None) for e in eval_logs]
    eval_runtime = [e.get("eval_runtime", None) for e in eval_logs]
    
    return {
        "train_steps": train_steps,
        "train_loss": train_loss,
        "train_lr": train_lr,
        "train_epoch": train_epoch,
        "eval_steps": eval_steps,
        "eval_loss": eval_loss,
        "eval_wer": eval_wer,
        "eval_cer": eval_cer,
        "eval_runtime": eval_runtime,
    }


# ============================================================
# Figure 1: Training Loss Curve
# ============================================================
def plot_training_loss(metrics, output_dir):
    """Training loss over steps with smoothed trend line."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    steps = metrics["train_steps"]
    loss = metrics["train_loss"]
    
    # Raw loss (semi-transparent)
    ax.plot(steps, loss, color=COLORS['train_loss'], alpha=0.3, linewidth=1,
            label='Raw Loss')
    
    # Smoothed loss (exponential moving average)
    if len(loss) > 5:
        alpha = 0.15
        smoothed = [loss[0]]
        for i in range(1, len(loss)):
            smoothed.append(alpha * loss[i] + (1 - alpha) * smoothed[-1])
        ax.plot(steps, smoothed, color=COLORS['train_loss'], linewidth=2.5,
                label='Smoothed Loss (EMA)')
    
    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss Curve')
    ax.legend(loc='upper right')
    
    # Add epoch markers on secondary x-axis
    if metrics["train_epoch"]:
        ax2 = ax.twiny()
        epoch_changes = []
        prev_epoch = -1
        for s, e in zip(steps, metrics["train_epoch"]):
            epoch = int(e)
            if epoch != prev_epoch:
                epoch_changes.append((s, epoch))
                prev_epoch = epoch
        if epoch_changes:
            ax2.set_xlim(ax.get_xlim())
            ax2.set_xticks([s for s, _ in epoch_changes])
            ax2.set_xticklabels([f'Epoch {e}' for _, e in epoch_changes],
                               fontsize=9, rotation=0)
            ax2.tick_params(length=0)
    
    path = os.path.join(output_dir, "01_training_loss.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  [✓] Saved: {path}")


# ============================================================
# Figure 2: Evaluation Loss Curve
# ============================================================
def plot_eval_loss(metrics, output_dir):
    """Evaluation loss at each eval step."""
    if not metrics["eval_steps"]:
        print("  [!] No eval data found, skipping eval loss plot")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    steps = metrics["eval_steps"]
    loss = metrics["eval_loss"]
    
    ax.plot(steps, loss, 'o-', color=COLORS['eval_loss'], linewidth=2.5,
            markersize=10, markerfacecolor='white', markeredgewidth=2,
            label='Eval Loss')
    
    # Highlight best checkpoint
    best_idx = np.argmin(loss)
    ax.scatter(steps[best_idx], loss[best_idx], s=200, color=COLORS['eval_loss'],
               zorder=5, marker='*', edgecolors='black', linewidths=1)
    ax.annotate(f'Best: {loss[best_idx]:.4f}\n(Step {steps[best_idx]})',
                xy=(steps[best_idx], loss[best_idx]),
                xytext=(15, 15), textcoords='offset points',
                fontsize=11, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='black'),
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.8))
    
    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Eval Loss')
    ax.set_title('Evaluation Loss Over Training')
    ax.legend(loc='upper right')
    
    path = os.path.join(output_dir, "02_eval_loss.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  [✓] Saved: {path}")


# ============================================================
# Figure 3: WER (Word Error Rate) Curve
# ============================================================
def plot_wer_curve(metrics, output_dir):
    """WER progression over eval steps."""
    wer = [w for w in metrics["eval_wer"] if w is not None]
    if not wer:
        print("  [!] No WER data found, skipping WER plot")
        return
    
    steps = metrics["eval_steps"][:len(wer)]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(steps, wer, 's-', color=COLORS['wer'], linewidth=2.5,
            markersize=10, markerfacecolor='white', markeredgewidth=2,
            label='WER (%)')
    
    # Fill area under curve
    ax.fill_between(steps, wer, alpha=0.15, color=COLORS['wer'])
    
    # Highlight best WER
    best_idx = np.argmin(wer)
    ax.scatter(steps[best_idx], wer[best_idx], s=200, color=COLORS['wer'],
               zorder=5, marker='*', edgecolors='black', linewidths=1)
    ax.annotate(f'Best WER: {wer[best_idx]:.2f}%\n(Step {steps[best_idx]})',
                xy=(steps[best_idx], wer[best_idx]),
                xytext=(15, 15), textcoords='offset points',
                fontsize=11, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='black'),
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.8))
    
    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Word Error Rate (%)')
    ax.set_title('Word Error Rate (WER) During Training')
    ax.legend(loc='upper right')
    
    path = os.path.join(output_dir, "03_wer_curve.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  [✓] Saved: {path}")


# ============================================================
# Figure 4: CER (Character Error Rate) Curve
# ============================================================
def plot_cer_curve(metrics, output_dir):
    """CER progression over eval steps."""
    cer = [c for c in metrics["eval_cer"] if c is not None]
    if not cer:
        print("  [!] No CER data found, skipping CER plot")
        return
    
    steps = metrics["eval_steps"][:len(cer)]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(steps, cer, 'D-', color=COLORS['cer'], linewidth=2.5,
            markersize=10, markerfacecolor='white', markeredgewidth=2,
            label='CER (%)')
    
    # Fill area under curve
    ax.fill_between(steps, cer, alpha=0.15, color=COLORS['cer'])
    
    # Highlight best CER
    best_idx = np.argmin(cer)
    ax.scatter(steps[best_idx], cer[best_idx], s=200, color=COLORS['cer'],
               zorder=5, marker='*', edgecolors='black', linewidths=1)
    ax.annotate(f'Best CER: {cer[best_idx]:.2f}%\n(Step {steps[best_idx]})',
                xy=(steps[best_idx], cer[best_idx]),
                xytext=(15, 15), textcoords='offset points',
                fontsize=11, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='black'),
                bbox=dict(boxstyle='round,pad=0.3', facecolor='plum', alpha=0.8))
    
    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Character Error Rate (%)')
    ax.set_title('Character Error Rate (CER) During Training')
    ax.legend(loc='upper right')
    
    path = os.path.join(output_dir, "04_cer_curve.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  [✓] Saved: {path}")


# ============================================================
# Figure 5: Train vs Eval Loss Comparison
# ============================================================
def plot_train_vs_eval_loss(metrics, output_dir):
    """Overlay training and evaluation loss to detect overfitting."""
    if not metrics["eval_steps"]:
        print("  [!] No eval data found, skipping train vs eval plot")
        return
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Training loss (smoothed)
    steps = metrics["train_steps"]
    loss = metrics["train_loss"]
    
    if len(loss) > 5:
        alpha = 0.15
        smoothed = [loss[0]]
        for i in range(1, len(loss)):
            smoothed.append(alpha * loss[i] + (1 - alpha) * smoothed[-1])
        ax.plot(steps, smoothed, color=COLORS['train_loss'], linewidth=2.5,
                label='Train Loss (smoothed)')
        ax.fill_between(steps, smoothed, alpha=0.1, color=COLORS['train_loss'])
    else:
        ax.plot(steps, loss, color=COLORS['train_loss'], linewidth=2.5,
                label='Train Loss')
    
    # Eval loss
    ax.plot(metrics["eval_steps"], metrics["eval_loss"], 'o-',
            color=COLORS['eval_loss'], linewidth=2.5, markersize=10,
            markerfacecolor='white', markeredgewidth=2,
            label='Eval Loss')
    
    # Potential overfitting zone
    if len(metrics["eval_loss"]) >= 3:
        eval_losses = metrics["eval_loss"]
        if eval_losses[-1] > eval_losses[-2] > eval_losses[-3]:
            overfitting_start = metrics["eval_steps"][-3]
            ax.axvspan(overfitting_start, max(steps),
                       alpha=0.1, color='red', label='Possible Overfitting Zone')
    
    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Loss')
    ax.set_title('Training vs Evaluation Loss (Overfitting Detection)')
    ax.legend(loc='upper right')
    
    path = os.path.join(output_dir, "05_train_vs_eval_loss.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  [✓] Saved: {path}")


# ============================================================
# Figure 6: WER vs CER Comparison
# ============================================================
def plot_wer_vs_cer(metrics, output_dir):
    """WER and CER on same plot for comparison."""
    wer = [w for w in metrics["eval_wer"] if w is not None]
    cer = [c for c in metrics["eval_cer"] if c is not None]
    
    if not wer or not cer:
        print("  [!] No WER/CER data found, skipping comparison plot")
        return
    
    min_len = min(len(wer), len(cer))
    wer = wer[:min_len]
    cer = cer[:min_len]
    steps = metrics["eval_steps"][:min_len]
    
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    # WER on left y-axis
    line1 = ax1.plot(steps, wer, 's-', color=COLORS['wer'], linewidth=2.5,
                     markersize=10, markerfacecolor='white', markeredgewidth=2,
                     label='WER (%)')
    ax1.set_xlabel('Training Steps')
    ax1.set_ylabel('WER (%)', color=COLORS['wer'])
    ax1.tick_params(axis='y', labelcolor=COLORS['wer'])
    ax1.fill_between(steps, wer, alpha=0.1, color=COLORS['wer'])
    
    # CER on right y-axis
    ax2 = ax1.twinx()
    line2 = ax2.plot(steps, cer, 'D-', color=COLORS['cer'], linewidth=2.5,
                     markersize=10, markerfacecolor='white', markeredgewidth=2,
                     label='CER (%)')
    ax2.set_ylabel('CER (%)', color=COLORS['cer'])
    ax2.tick_params(axis='y', labelcolor=COLORS['cer'])
    
    # Combined legend
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper right')
    
    ax1.set_title('WER vs CER Comparison Over Training')
    
    path = os.path.join(output_dir, "06_wer_vs_cer.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  [✓] Saved: {path}")


# ============================================================
# Figure 7: Learning Rate Schedule
# ============================================================
def plot_learning_rate(metrics, output_dir):
    """Learning rate schedule visualization."""
    steps = metrics["train_steps"]
    lr = metrics["train_lr"]
    
    if not any(lr):
        print("  [!] No learning rate data found, skipping LR plot")
        return
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    ax.plot(steps, lr, color=COLORS['lr'], linewidth=2.5)
    ax.fill_between(steps, lr, alpha=0.15, color=COLORS['lr'])
    
    # Mark warmup phase
    warmup_end = None
    for i in range(1, len(lr)):
        if lr[i] < lr[i-1]:
            warmup_end = steps[i]
            break
    
    if warmup_end:
        ax.axvline(x=warmup_end, color='gray', linestyle='--', alpha=0.7)
        ax.annotate('Warmup End', xy=(warmup_end, max(lr) * 0.95),
                    fontsize=10, ha='right',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))
    
    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Learning Rate')
    ax.set_title('Learning Rate Schedule')
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
    ax.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))
    
    path = os.path.join(output_dir, "07_learning_rate.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  [✓] Saved: {path}")


# ============================================================
# Figure 8: Final Summary Dashboard (4-in-1)
# ============================================================
def plot_summary_dashboard(metrics, output_dir):
    """4-panel summary dashboard for reports/papers."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Whisper Fine-Tuning — Training Summary Dashboard',
                 fontsize=18, fontweight='bold', y=0.98)
    
    # Panel 1: Training Loss (top-left)
    ax = axes[0, 0]
    steps = metrics["train_steps"]
    loss = metrics["train_loss"]
    ax.plot(steps, loss, color=COLORS['train_loss'], alpha=0.3, linewidth=1)
    if len(loss) > 5:
        alpha_ema = 0.15
        smoothed = [loss[0]]
        for i in range(1, len(loss)):
            smoothed.append(alpha_ema * loss[i] + (1 - alpha_ema) * smoothed[-1])
        ax.plot(steps, smoothed, color=COLORS['train_loss'], linewidth=2.5)
    ax.set_xlabel('Steps')
    ax.set_ylabel('Loss')
    ax.set_title('(a) Training Loss')
    
    # Panel 2: Eval Loss (top-right)
    ax = axes[0, 1]
    if metrics["eval_steps"]:
        ax.plot(metrics["eval_steps"], metrics["eval_loss"], 'o-',
                color=COLORS['eval_loss'], linewidth=2.5,
                markersize=8, markerfacecolor='white', markeredgewidth=2)
        best_idx = np.argmin(metrics["eval_loss"])
        ax.scatter(metrics["eval_steps"][best_idx], metrics["eval_loss"][best_idx],
                   s=150, color=COLORS['eval_loss'], zorder=5, marker='*',
                   edgecolors='black')
    ax.set_xlabel('Steps')
    ax.set_ylabel('Loss')
    ax.set_title('(b) Evaluation Loss')
    
    # Panel 3: WER (bottom-left)
    ax = axes[1, 0]
    wer = [w for w in metrics["eval_wer"] if w is not None]
    if wer:
        eval_steps = metrics["eval_steps"][:len(wer)]
        ax.plot(eval_steps, wer, 's-', color=COLORS['wer'], linewidth=2.5,
                markersize=8, markerfacecolor='white', markeredgewidth=2)
        ax.fill_between(eval_steps, wer, alpha=0.15, color=COLORS['wer'])
        best_idx = np.argmin(wer)
        ax.scatter(eval_steps[best_idx], wer[best_idx],
                   s=150, color=COLORS['wer'], zorder=5, marker='*',
                   edgecolors='black')
        ax.annotate(f'{wer[best_idx]:.1f}%',
                    xy=(eval_steps[best_idx], wer[best_idx]),
                    xytext=(10, 10), textcoords='offset points',
                    fontsize=11, fontweight='bold')
    ax.set_xlabel('Steps')
    ax.set_ylabel('WER (%)')
    ax.set_title('(c) Word Error Rate')
    
    # Panel 4: CER (bottom-right)
    ax = axes[1, 1]
    cer = [c for c in metrics["eval_cer"] if c is not None]
    if cer:
        eval_steps = metrics["eval_steps"][:len(cer)]
        ax.plot(eval_steps, cer, 'D-', color=COLORS['cer'], linewidth=2.5,
                markersize=8, markerfacecolor='white', markeredgewidth=2)
        ax.fill_between(eval_steps, cer, alpha=0.15, color=COLORS['cer'])
        best_idx = np.argmin(cer)
        ax.scatter(eval_steps[best_idx], cer[best_idx],
                   s=150, color=COLORS['cer'], zorder=5, marker='*',
                   edgecolors='black')
        ax.annotate(f'{cer[best_idx]:.1f}%',
                    xy=(eval_steps[best_idx], cer[best_idx]),
                    xytext=(10, 10), textcoords='offset points',
                    fontsize=11, fontweight='bold')
    ax.set_xlabel('Steps')
    ax.set_ylabel('CER (%)')
    ax.set_title('(d) Character Error Rate')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    path = os.path.join(output_dir, "08_summary_dashboard.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  [✓] Saved: {path}")


# ============================================================
# Figure 9: Metrics Table (text-based summary)
# ============================================================
def save_metrics_table(metrics, state, output_dir):
    """Save a text summary table of all metrics."""
    lines = []
    lines.append("=" * 65)
    lines.append("  WHISPER FINE-TUNING — RESULTS SUMMARY")
    lines.append("=" * 65)
    lines.append("")
    
    # Training info
    total_steps = max(metrics["train_steps"]) if metrics["train_steps"] else 0
    final_loss = metrics["train_loss"][-1] if metrics["train_loss"] else 0
    min_loss = min(metrics["train_loss"]) if metrics["train_loss"] else 0
    lines.append(f"  Training Steps Completed:  {total_steps}")
    lines.append(f"  Final Training Loss:       {final_loss:.4f}")
    lines.append(f"  Min Training Loss:         {min_loss:.4f}")
    lines.append("")
    
    # Eval info
    if metrics["eval_loss"]:
        best_eval_idx = np.argmin(metrics["eval_loss"])
        lines.append(f"  Best Eval Loss:            {metrics['eval_loss'][best_eval_idx]:.4f} (Step {metrics['eval_steps'][best_eval_idx]})")
    
    wer = [w for w in metrics["eval_wer"] if w is not None]
    if wer:
        best_wer_idx = np.argmin(wer)
        lines.append(f"  Best WER:                  {wer[best_wer_idx]:.2f}% (Step {metrics['eval_steps'][best_wer_idx]})")
        lines.append(f"  Final WER:                 {wer[-1]:.2f}%")
        if len(wer) > 1:
            improvement = wer[0] - wer[best_wer_idx]
            lines.append(f"  WER Improvement:           {improvement:.2f}% (from {wer[0]:.2f}% to {wer[best_wer_idx]:.2f}%)")
    
    cer = [c for c in metrics["eval_cer"] if c is not None]
    if cer:
        best_cer_idx = np.argmin(cer)
        lines.append(f"  Best CER:                  {cer[best_cer_idx]:.2f}% (Step {metrics['eval_steps'][best_cer_idx]})")
        lines.append(f"  Final CER:                 {cer[-1]:.2f}%")
    
    lines.append("")
    
    # Best checkpoint
    best_checkpoint = state.get("best_model_checkpoint", "N/A")
    best_metric = state.get("best_metric", "N/A")
    lines.append(f"  Best Checkpoint:           {best_checkpoint}")
    lines.append(f"  Best Metric Value:         {best_metric}")
    lines.append("")
    lines.append("=" * 65)
    
    # Print and save
    text = "\n".join(lines)
    print("\n" + text + "\n")
    
    path = os.path.join(output_dir, "results_summary.txt")
    with open(path, 'w') as f:
        f.write(text)
    print(f"  [✓] Saved: {path}")


# ============================================================
# Figure 10: Per-Step Loss Distribution (Histogram)
# ============================================================
def plot_loss_distribution(metrics, output_dir):
    """Distribution of training loss values (early vs late)."""
    loss = metrics["train_loss"]
    if len(loss) < 20:
        print("  [!] Not enough data for loss distribution, skipping")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Split into early and late training
    mid = len(loss) // 2
    early = loss[:mid]
    late = loss[mid:]
    
    ax.hist(early, bins=30, alpha=0.6, color=COLORS['baseline'],
            label=f'First Half (Steps 0-{metrics["train_steps"][mid]})', edgecolor='white')
    ax.hist(late, bins=30, alpha=0.6, color=COLORS['finetuned'],
            label=f'Second Half (Steps {metrics["train_steps"][mid]}+)', edgecolor='white')
    
    ax.axvline(np.mean(early), color=COLORS['baseline'], linestyle='--', linewidth=2,
               label=f'First Half Mean: {np.mean(early):.4f}')
    ax.axvline(np.mean(late), color=COLORS['finetuned'], linestyle='--', linewidth=2,
               label=f'Second Half Mean: {np.mean(late):.4f}')
    
    ax.set_xlabel('Loss Value')
    ax.set_ylabel('Frequency')
    ax.set_title('Loss Distribution: Early vs Late Training')
    ax.legend(loc='upper right')
    
    path = os.path.join(output_dir, "09_loss_distribution.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  [✓] Saved: {path}")


# ============================================================
# Figure 11: Eval Runtime per Step
# ============================================================
def plot_eval_runtime(metrics, output_dir):
    """Evaluation runtime to show computational cost."""
    runtime = [r for r in metrics["eval_runtime"] if r is not None]
    if not runtime:
        print("  [!] No eval runtime data, skipping")
        return
    
    steps = metrics["eval_steps"][:len(runtime)]
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    bars = ax.bar(range(len(steps)), runtime, color=COLORS['lr'],
                  edgecolor='white', alpha=0.8, width=0.7)
    ax.set_xticks(range(len(steps)))
    ax.set_xticklabels([f'Step\n{s}' for s in steps])
    ax.set_xlabel('Evaluation Checkpoint')
    ax.set_ylabel('Runtime (seconds)')
    ax.set_title('Evaluation Runtime per Checkpoint')
    
    # Add value labels on bars
    for bar, val in zip(bars, runtime):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f'{val:.1f}s', ha='center', va='bottom', fontsize=10)
    
    path = os.path.join(output_dir, "10_eval_runtime.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  [✓] Saved: {path}")


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Generate all training figures")
    parser.add_argument("--checkpoint_dir", type=str,
                        default="./outputs/checkpoints",
                        help="Path to checkpoint directory")
    parser.add_argument("--output_dir", type=str,
                        default="./figures",
                        help="Directory to save figures")
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 60)
    print("  Whisper Fine-Tuning — Results Visualization")
    print("=" * 60)
    print(f"  Checkpoint dir: {args.checkpoint_dir}")
    print(f"  Output dir:     {args.output_dir}")
    print("=" * 60)
    
    # Load data
    print("\n[1] Loading trainer state...")
    state = load_trainer_state(args.checkpoint_dir)
    
    print("[2] Extracting metrics...")
    metrics = extract_metrics(state)
    
    print(f"    Found {len(metrics['train_steps'])} training log entries")
    print(f"    Found {len(metrics['eval_steps'])} evaluation entries")
    
    # Generate all figures
    print("\n[3] Generating figures...\n")
    
    plot_training_loss(metrics, args.output_dir)
    plot_eval_loss(metrics, args.output_dir)
    plot_wer_curve(metrics, args.output_dir)
    plot_cer_curve(metrics, args.output_dir)
    plot_train_vs_eval_loss(metrics, args.output_dir)
    plot_wer_vs_cer(metrics, args.output_dir)
    plot_learning_rate(metrics, args.output_dir)
    plot_summary_dashboard(metrics, args.output_dir)
    plot_loss_distribution(metrics, args.output_dir)
    plot_eval_runtime(metrics, args.output_dir)
    
    # Save text summary
    print("")
    save_metrics_table(metrics, state, args.output_dir)
    
    print(f"\n{'=' * 60}")
    print(f"  ✅ All figures saved to: {args.output_dir}/")
    print(f"{'=' * 60}")
    
    # List all generated files
    print("\n  Generated files:")
    for f in sorted(os.listdir(args.output_dir)):
        size = os.path.getsize(os.path.join(args.output_dir, f))
        print(f"    📊 {f} ({size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
