#!/usr/bin/env python3
"""
Publication-Quality Visualizations for E∆-MHC-Geo

Creates figures following mHC paper (arXiv:2512.24880) style:
1. Training dynamics with gradient norms (Fig 5 style)
2. Propagation stability (Fig 3/7 style)
3. Ablation study results (Table 1 style)
4. Comprehensive summary figure
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import ScalarFormatter
import json

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'legend.fontsize': 9,
    'lines.linewidth': 1.5,
    'figure.dpi': 150,
    'font.family': 'sans-serif',
})

COLORS = {
    'GPT': '#4285F4',      # Blue
    'DDL': '#EA4335',      # Red
    'mHC': '#FBBC05',      # Yellow
    'E∆-MHC-Geo': '#34A853',  # Green
    'Cayley-only': '#9C27B0',  # Purple
    'Householder-only': '#FF9800',  # Orange
}

# Directory mappings for new gradient norm training
GRADNORM_DIRS = {
    'gyroscope': {
        'GPT': 'out-gradnorm/gyroscope-baseline',
        'DDL': 'out-gradnorm/gyroscope-ddl',
        'mHC': 'out-gradnorm/gyroscope-mhc',
        'E∆-MHC-Geo': 'out-gradnorm/gyroscope-proposed',
    },
    'correction': {
        'GPT': 'out-gradnorm/correction-baseline',
        'DDL': 'out-gradnorm/correction-ddl',
        'mHC': 'out-gradnorm/correction-mhc',
        'E∆-MHC-Geo': 'out-gradnorm/correction-proposed',
    },
    'stability': {
        'GPT': 'out-gradnorm/stability-baseline',
        'DDL': 'out-gradnorm/stability-ddl',
        'mHC': 'out-gradnorm/stability-mhc',
        'E∆-MHC-Geo': 'out-gradnorm/stability-proposed',
    },
}

# Ablation directories
ABLATION_DIRS = {
    'gyroscope': {
        'Cayley-only': 'out-ablation/cayley_only-gyroscope',
        'Householder-only': 'out-ablation/householder_only-gyroscope',
    },
    'correction': {
        'Cayley-only': 'out-ablation/cayley_only-correction',
        'Householder-only': 'out-ablation/householder_only-correction',
    },
    'stability': {
        'Cayley-only': 'out-ablation/cayley_only-stability',
        'Householder-only': 'out-ablation/householder_only-stability',
    },
}


def load_training_log(out_dir):
    """Load training log from output directory."""
    log_path = os.path.join(out_dir, 'train_log.npy')
    if os.path.exists(log_path):
        return np.load(log_path, allow_pickle=True).item()
    return None


def smooth_curve(values, window=5):
    """Apply simple moving average smoothing."""
    if len(values) < window:
        return values
    return np.convolve(values, np.ones(window)/window, mode='valid')


def create_figure_5_style(dataset='gyroscope', include_ablation=False):
    """
    Create Figure 5 style from mHC paper:
    (a) Training/Validation Loss vs Steps
    (b) Gradient Norm vs Steps
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Load logs
    logs = {}
    for model_name, out_dir in GRADNORM_DIRS[dataset].items():
        log = load_training_log(out_dir)
        if log:
            logs[model_name] = log
    
    # Add ablation logs if requested
    if include_ablation:
        for model_name, out_dir in ABLATION_DIRS.get(dataset, {}).items():
            log = load_training_log(out_dir)
            if log:
                logs[model_name] = log
    
    if not logs:
        print(f"No logs found for {dataset}")
        return None
    
    # (a) Validation Loss vs Steps
    ax1 = axes[0]
    for model_name, log in logs.items():
        color = COLORS.get(model_name, '#666666')
        iters = np.array(log['iter'])
        val_loss = np.array(log['val_loss'])
        ax1.plot(iters, val_loss, color=color, label=model_name, linewidth=2)
    
    ax1.set_xlabel('Training Steps')
    ax1.set_ylabel('Validation Loss')
    ax1.set_title('(a) Validation Loss vs Training Steps')
    ax1.legend(loc='upper right', framealpha=0.9)
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3)
    
    # (b) Gradient Norm vs Steps
    ax2 = axes[1]
    has_grad_data = False
    
    for model_name, log in logs.items():
        if 'grad_norm' in log and len(log['grad_norm']) > 0:
            has_grad_data = True
            color = COLORS.get(model_name, '#666666')
            
            # Gradient norms are logged at log_interval (every 50 steps)
            grad_norms = np.array(log['grad_norm'])
            grad_iters = np.arange(len(grad_norms)) * 50  # log_interval = 50
            
            # Apply smoothing
            if len(grad_norms) > 5:
                smoothed = smooth_curve(grad_norms, window=5)
                smooth_iters = grad_iters[:len(smoothed)]
                ax2.plot(smooth_iters, smoothed, color=color, label=model_name, linewidth=2, alpha=0.8)
            else:
                ax2.plot(grad_iters, grad_norms, color=color, label=model_name, linewidth=2, alpha=0.8)
    
    if has_grad_data:
        ax2.set_xlabel('Training Steps')
        ax2.set_ylabel('Gradient Norm')
        ax2.set_title('(b) Gradient Norm vs Training Steps')
        ax2.legend(loc='upper right', framealpha=0.9)
        ax2.set_yscale('log')
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, 'Gradient norm data not available',
                ha='center', va='center', transform=ax2.transAxes, fontsize=12)
    
    fig.suptitle(f'Training Dynamics: {dataset.capitalize()} Dataset\n(Following mHC Paper Figure 5 Style)',
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    filename = f'pub_fig5_{dataset}.png'
    if include_ablation:
        filename = f'pub_fig5_{dataset}_ablation.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: {filename}")
    return fig


def create_gradient_analysis():
    """Create comprehensive gradient analysis across all datasets."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    
    datasets = ['gyroscope', 'correction', 'stability']
    
    # Row 1: Validation Loss
    for i, dataset in enumerate(datasets):
        ax = axes[0, i]
        
        for model_name, out_dir in GRADNORM_DIRS[dataset].items():
            log = load_training_log(out_dir)
            if log:
                color = COLORS.get(model_name, '#666666')
                iters = np.array(log['iter'])
                val_loss = np.array(log['val_loss'])
                ax.plot(iters, val_loss, color=color, label=model_name, linewidth=1.5)
        
        ax.set_xlabel('Steps')
        ax.set_ylabel('Val Loss')
        ax.set_title(f'{dataset.capitalize()}')
        ax.set_yscale('log')
        if i == 0:
            ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
    
    # Row 2: Gradient Norms
    for i, dataset in enumerate(datasets):
        ax = axes[1, i]
        
        for model_name, out_dir in GRADNORM_DIRS[dataset].items():
            log = load_training_log(out_dir)
            if log and 'grad_norm' in log and len(log['grad_norm']) > 0:
                color = COLORS.get(model_name, '#666666')
                grad_norms = np.array(log['grad_norm'])
                grad_iters = np.arange(len(grad_norms)) * 50
                
                if len(grad_norms) > 5:
                    smoothed = smooth_curve(grad_norms, window=5)
                    ax.plot(grad_iters[:len(smoothed)], smoothed, color=color, 
                           label=model_name, linewidth=1.5, alpha=0.8)
                else:
                    ax.plot(grad_iters, grad_norms, color=color, label=model_name, linewidth=1.5)
        
        ax.set_xlabel('Steps')
        ax.set_ylabel('Grad Norm')
        ax.set_title(f'Gradient Norm')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
    
    fig.suptitle('E∆-MHC-Geo: Training Dynamics Analysis\n(Loss and Gradient Norm Comparison)',
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('pub_gradient_analysis.png', dpi=300, bbox_inches='tight', facecolor='white')
    print("Saved: pub_gradient_analysis.png")
    return fig


def create_ablation_table():
    """Create ablation study results table and bar chart."""
    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.25)
    
    datasets = ['gyroscope', 'correction', 'stability']
    
    # Collect results
    results = {dataset: {} for dataset in datasets}
    
    # Main models
    for dataset in datasets:
        for model_name, out_dir in GRADNORM_DIRS[dataset].items():
            log = load_training_log(out_dir)
            if log and 'val_loss' in log:
                results[dataset][model_name] = log['val_loss'][-1]
    
    # Ablation models
    for dataset in datasets:
        for model_name, out_dir in ABLATION_DIRS.get(dataset, {}).items():
            log = load_training_log(out_dir)
            if log and 'val_loss' in log:
                results[dataset][model_name] = log['val_loss'][-1]
    
    # Row 1: Bar charts for each dataset
    for i, dataset in enumerate(datasets):
        ax = fig.add_subplot(gs[0, i])
        
        models = list(results[dataset].keys())
        losses = [results[dataset][m] for m in models]
        colors = [COLORS.get(m, '#666666') for m in models]
        
        # Sort by loss (best to worst)
        sorted_idx = np.argsort(losses)
        models = [models[i] for i in sorted_idx]
        losses = [losses[i] for i in sorted_idx]
        colors = [colors[i] for i in sorted_idx]
        
        # Shorten labels
        short_labels = [m.replace('E∆-MHC-Geo', 'Ours').replace('-only', '') for m in models]
        
        bars = ax.bar(short_labels, losses, color=colors, edgecolor='black', linewidth=0.5)
        ax.set_ylabel('Final Val Loss')
        ax.set_title(f'{dataset.capitalize()}')
        ax.set_yscale('log')
        ax.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, loss in zip(bars, losses):
            height = bar.get_height()
            ax.annotate(f'{loss:.2e}',
                       xy=(bar.get_x() + bar.get_width()/2, height),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom', fontsize=7, rotation=0)
    
    # Row 2: Summary table
    ax_table = fig.add_subplot(gs[1, :])
    ax_table.axis('off')
    
    # Create table data
    all_models = ['GPT', 'DDL', 'mHC', 'E∆-MHC-Geo', 'Cayley-only', 'Householder-only']
    table_data = []
    
    for model in all_models:
        row = [model]
        for dataset in datasets:
            if model in results[dataset]:
                val = results[dataset][model]
                # Highlight best
                row.append(f'{val:.2e}')
            else:
                row.append('-')
        table_data.append(row)
    
    table = ax_table.table(
        cellText=table_data,
        colLabels=['Model'] + [d.capitalize() for d in datasets],
        loc='center',
        cellLoc='center',
        colWidths=[0.25, 0.25, 0.25, 0.25]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)
    
    # Style the table
    for i in range(len(all_models) + 1):
        for j in range(4):
            cell = table[i, j]
            if i == 0:  # Header
                cell.set_facecolor('#4285F4')
                cell.set_text_props(color='white', weight='bold')
            elif j == 0:  # Model names
                cell.set_facecolor('#f0f0f0')
                cell.set_text_props(weight='bold')
            else:
                cell.set_facecolor('white')
    
    fig.suptitle('E∆-MHC-Geo Ablation Study Results\n(Following mHC Paper Table 1 Style)',
                fontsize=14, fontweight='bold')
    
    plt.savefig('pub_ablation_results.png', dpi=300, bbox_inches='tight', facecolor='white')
    print("Saved: pub_ablation_results.png")
    return fig


def create_comprehensive_figure():
    """Create a comprehensive multi-panel figure for the paper."""
    fig = plt.figure(figsize=(18, 16))
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.25)
    
    datasets = ['gyroscope', 'correction', 'stability']
    
    # Row 1: Loss curves
    for i, dataset in enumerate(datasets):
        ax = fig.add_subplot(gs[0, i])
        
        for model_name, out_dir in GRADNORM_DIRS[dataset].items():
            log = load_training_log(out_dir)
            if log:
                color = COLORS.get(model_name, '#666666')
                iters = np.array(log['iter'])
                val_loss = np.array(log['val_loss'])
                ax.plot(iters, val_loss, color=color, label=model_name, linewidth=1.5)
        
        ax.set_xlabel('Steps')
        ax.set_ylabel('Val Loss')
        ax.set_title(f'{dataset.capitalize()} - Loss')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(loc='upper right', fontsize=7)
    
    # Row 2: Gradient norms
    for i, dataset in enumerate(datasets):
        ax = fig.add_subplot(gs[1, i])
        
        for model_name, out_dir in GRADNORM_DIRS[dataset].items():
            log = load_training_log(out_dir)
            if log and 'grad_norm' in log and len(log['grad_norm']) > 0:
                color = COLORS.get(model_name, '#666666')
                grad_norms = np.array(log['grad_norm'])
                grad_iters = np.arange(len(grad_norms)) * 50
                
                if len(grad_norms) > 5:
                    smoothed = smooth_curve(grad_norms, window=5)
                    ax.plot(grad_iters[:len(smoothed)], smoothed, color=color, 
                           label=model_name, linewidth=1.5, alpha=0.8)
        
        ax.set_xlabel('Steps')
        ax.set_ylabel('Grad Norm')
        ax.set_title(f'{dataset.capitalize()} - Gradient Norm')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
    
    # Row 3: Final results summary
    ax_summary = fig.add_subplot(gs[2, :])
    
    # Collect final losses
    final_losses = {}
    for dataset in datasets:
        final_losses[dataset] = {}
        for model_name, out_dir in GRADNORM_DIRS[dataset].items():
            log = load_training_log(out_dir)
            if log and 'val_loss' in log:
                final_losses[dataset][model_name] = log['val_loss'][-1]
    
    # Create grouped bar chart
    models = ['GPT', 'DDL', 'mHC', 'E∆-MHC-Geo']
    x = np.arange(len(datasets))
    width = 0.2
    
    for i, model in enumerate(models):
        losses = [final_losses[d].get(model, 0) for d in datasets]
        offset = (i - 1.5) * width
        bars = ax_summary.bar(x + offset, losses, width, 
                             label=model, color=COLORS.get(model, '#666666'),
                             edgecolor='black', linewidth=0.5)
    
    ax_summary.set_xlabel('Dataset')
    ax_summary.set_ylabel('Final Validation Loss')
    ax_summary.set_title('Final Performance Comparison')
    ax_summary.set_xticks(x)
    ax_summary.set_xticklabels([d.capitalize() for d in datasets])
    ax_summary.legend(loc='upper right')
    ax_summary.set_yscale('log')
    ax_summary.grid(True, alpha=0.3, axis='y')
    
    fig.suptitle('E∆-MHC-Geo: Complete Experimental Analysis\n(Training Dynamics, Gradient Stability, and Performance)',
                fontsize=16, fontweight='bold', y=0.98)
    
    plt.savefig('pub_comprehensive.png', dpi=300, bbox_inches='tight', facecolor='white')
    print("Saved: pub_comprehensive.png")
    return fig


def print_results_summary():
    """Print a summary of all results."""
    print("\n" + "=" * 80)
    print("EXPERIMENTAL RESULTS SUMMARY")
    print("=" * 80)
    
    datasets = ['gyroscope', 'correction', 'stability']
    
    for dataset in datasets:
        print(f"\n{dataset.upper()}")
        print("-" * 40)
        
        results = []
        for model_name, out_dir in GRADNORM_DIRS[dataset].items():
            log = load_training_log(out_dir)
            if log and 'val_loss' in log:
                final_loss = log['val_loss'][-1]
                results.append((model_name, final_loss))
        
        # Add ablation
        for model_name, out_dir in ABLATION_DIRS.get(dataset, {}).items():
            log = load_training_log(out_dir)
            if log and 'val_loss' in log:
                final_loss = log['val_loss'][-1]
                results.append((model_name, final_loss))
        
        # Sort by loss
        results.sort(key=lambda x: x[1])
        
        for model, loss in results:
            print(f"  {model:20s}: {loss:.6e}")
    
    print("\n" + "=" * 80)


if __name__ == '__main__':
    print("Creating publication-quality figures...")
    print()
    
    # Print results summary
    print_results_summary()
    
    # Create individual dataset figures
    for dataset in ['gyroscope', 'correction', 'stability']:
        print(f"\nDataset: {dataset}")
        create_figure_5_style(dataset)
    
    # Create gradient analysis
    print("\nCreating gradient analysis...")
    create_gradient_analysis()
    
    # Create ablation results (if available)
    print("\nCreating ablation results...")
    try:
        create_ablation_table()
    except Exception as e:
        print(f"  Ablation results not available yet: {e}")
    
    # Create comprehensive figure
    print("\nCreating comprehensive figure...")
    create_comprehensive_figure()
    
    print("\nAll figures created!")
