#!/usr/bin/env python3
"""
Publication-Quality Figures for Top-Tier Journal (NeurIPS/ICML/ICLR Style)

Design principles:
1. Clear, unobstructed data visualization
2. Consistent, readable fonts (minimum 10pt for print)
3. Color-blind friendly palette
4. High resolution (300 DPI minimum)
5. Proper whitespace and margins
6. No overlapping labels
"""

import os
import sys
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import LogLocator, LogFormatterMathtext
import warnings
warnings.filterwarnings('ignore')

# Publication-quality settings (NeurIPS style)
plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'lines.linewidth': 2.0,
    'lines.markersize': 6,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
    'axes.linewidth': 1.0,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linewidth': 0.5,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# Color-blind friendly palette (Wong, 2011)
COLORS = {
    'GPT': '#0072B2',        # Blue
    'DDL': '#D55E00',        # Vermillion/Orange
    'mHC': '#E69F00',        # Orange/Amber (more visible than yellow)
    'E∆-MHC-Geo': '#009E73', # Bluish Green
}

# Line styles for additional differentiation
LINE_STYLES = {
    'GPT': '-',
    'DDL': '-',
    'mHC': '-',
    'E∆-MHC-Geo': '-',
}

# Data directories (gyroscope and stability only - correction handled separately in experiments/)
GRADNORM_DIRS = {
    'gyroscope': {
        'GPT': 'out-gradnorm/gyroscope-baseline',
        'DDL': 'out-gradnorm/gyroscope-ddl',
        'mHC': 'out-gradnorm/gyroscope-mhc',
        'E∆-MHC-Geo': 'out-gradnorm/gyroscope-proposed',
    },
    'stability': {
        'GPT': 'out-gradnorm/stability-baseline',
        'DDL': 'out-gradnorm/stability-ddl',
        'mHC': 'out-gradnorm/stability-mhc',
        'E∆-MHC-Geo': 'out-gradnorm/stability-proposed',
    },
}


def load_training_log(out_dir):
    """Load training log from output directory."""
    log_path = os.path.join(out_dir, 'train_log.npy')
    if os.path.exists(log_path):
        return np.load(log_path, allow_pickle=True).item()
    return None


def smooth_curve(values, window=7):
    """Apply exponential moving average smoothing."""
    if len(values) < window:
        return values
    alpha = 2.0 / (window + 1)
    result = np.zeros_like(values, dtype=float)
    result[0] = values[0]
    for i in range(1, len(values)):
        result[i] = alpha * values[i] + (1 - alpha) * result[i-1]
    return result


def create_figure_1_training_dynamics():
    """
    Figure 1: Training Dynamics
    - Row 1: Validation Loss vs Steps (3 datasets)
    - Row 2: Gradient Norm vs Steps (3 datasets)
    """
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    
    datasets = ['gyroscope', 'stability']
    dataset_titles = ['Gyroscope\n(Manifold Precision)', 
                     'Stability\n(Unconditional Isometry)']
    
    model_order = ['GPT', 'DDL', 'mHC', 'E∆-MHC-Geo']
    
    for col, (dataset, title) in enumerate(zip(datasets, dataset_titles)):
        logs = {}
        for model_name, out_dir in GRADNORM_DIRS[dataset].items():
            log = load_training_log(out_dir)
            if log:
                logs[model_name] = log
        
        # Row 1: Validation Loss
        ax1 = axes[0, col]
        for model_name in model_order:
            if model_name in logs:
                log = logs[model_name]
                iters = np.array(log['iter'])
                val_loss = np.array(log['val_loss'])
                ax1.plot(iters, val_loss, color=COLORS[model_name], 
                        label=model_name, linewidth=2, linestyle=LINE_STYLES[model_name])
        
        ax1.set_xlabel('Training Steps')
        if col == 0:
            ax1.set_ylabel('Validation Loss')
        ax1.set_title(title, fontweight='bold', fontsize=11)
        ax1.set_yscale('log')
        ax1.set_xlim(0, 2000)
        ax1.tick_params(axis='both', which='major', labelsize=9)
        
        # Legend only on first plot, positioned outside data area
        if col == 0:
            ax1.legend(loc='upper right', framealpha=0.95, fontsize=9,
                      handlelength=1.5, handletextpad=0.5)
        
        # Row 2: Gradient Norm
        ax2 = axes[1, col]
        for model_name in model_order:
            if model_name in logs:
                log = logs[model_name]
                if 'grad_norm' in log and len(log['grad_norm']) > 0:
                    grad_norms = np.array(log['grad_norm'])
                    grad_iters = np.arange(len(grad_norms)) * 50
                    smoothed = smooth_curve(grad_norms, window=7)
                    ax2.plot(grad_iters, smoothed, color=COLORS[model_name],
                            label=model_name, linewidth=2, linestyle=LINE_STYLES[model_name])
        
        ax2.set_xlabel('Training Steps')
        if col == 0:
            ax2.set_ylabel('Gradient Norm')
        ax2.set_yscale('log')
        ax2.set_xlim(0, 2000)
        ax2.tick_params(axis='both', which='major', labelsize=9)
        
        # Add subtle annotation for mHC instability (stability dataset only)
        if dataset == 'stability':
            ax2.annotate('mHC\ninstability', xy=(900, 8), fontsize=8, 
                        color=COLORS['mHC'], ha='center', style='italic',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                                 edgecolor=COLORS['mHC'], alpha=0.8))
    
    # Row labels on the left side
    fig.text(0.01, 0.73, '(a)', fontsize=12, fontweight='bold', va='center')
    fig.text(0.01, 0.27, '(b)', fontsize=12, fontweight='bold', va='center')
    
    plt.tight_layout(rect=[0.02, 0, 1, 0.96])
    fig.suptitle('Training Dynamics: Loss and Gradient Norm Evolution', 
                fontsize=14, fontweight='bold')
    
    plt.savefig('results/journal_fig1_training.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    print("Saved: results/journal_fig1_training.png")
    plt.close()


def create_figure_2_stability_analysis():
    """
    Figure 2: Stability Analysis (Norm Preservation)
    """
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    from src.training.train_continuous import ContinuousModelWrapper
    from src.models.baseline_gpt import GPT as BaselineGPT, GPTConfig as BaselineConfig
    from src.models.ddl import GPT as DDLGPT, GPTConfig as DDLConfig
    from src.models.mhc import GPT as mHCGPT, GPTConfig as mHCConfig
    from src.models.edelta_hybrid import GPT as EdeltaGPT, GPTConfig as EdeltaConfig
    
    results = {}
    model_order = ['GPT', 'DDL', 'mHC', 'E∆-MHC-Geo']
    
    for model_name, out_dir in GRADNORM_DIRS['stability'].items():
        ckpt_path = os.path.join(out_dir, 'ckpt.pt')
        if not os.path.exists(ckpt_path):
            continue
        
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        saved_config = checkpoint['config']
        model_type = saved_config['model_type']
        
        for k, v in checkpoint['model'].items():
            if 'pos_emb' in k:
                block_size = v.shape[1]
                break
        
        model_map = {
            'gpt2': (BaselineGPT, BaselineConfig),
            'ddl': (DDLGPT, DDLConfig),
            'mhc': (mHCGPT, mHCConfig),
            'edelta': (EdeltaGPT, EdeltaConfig),
        }
        
        GPTClass, ConfigClass = model_map[model_type]
        config_args = {
            'n_layer': saved_config['n_layer'],
            'n_head': saved_config['n_head'],
            'n_embd': saved_config['n_embd'],
            'dropout': saved_config['dropout'],
            'bias': False,
            'block_size': block_size,
            'vocab_size': 1
        }
        if model_type in ['mhc', 'edelta']:
            config_args['n_streams'] = saved_config.get('n_streams', 4)
        
        config = ConfigClass(**config_args)
        core_model = GPTClass(config)
        
        input_dim = 64
        model = ContinuousModelWrapper(core_model, config, input_dim, block_size)
        model.load_state_dict(checkpoint['model'])
        model = model.to(device)
        model.eval()
        
        seq_len = 100
        n_sequences = 50
        
        all_output_norms = []
        with torch.no_grad():
            for _ in range(n_sequences):
                x = torch.randn(1, seq_len, input_dim, device=device)
                x = F.normalize(x, dim=-1)
                y = x.clone()
                pred, _ = model(x, y)
                output_norms = torch.norm(pred, dim=-1).squeeze().cpu().numpy()
                all_output_norms.append(output_norms)
        
        all_output_norms = np.array(all_output_norms)
        mean_norms = all_output_norms.mean(axis=0)
        std_norms = all_output_norms.std(axis=0)
        
        results[model_name] = {
            'mean_norms': mean_norms,
            'std_norms': std_norms,
            'norm_deviation': np.abs(mean_norms - 1.0).mean(),
        }
        
        del model
        torch.cuda.empty_cache()
    
    # Panel A: Norm vs Position
    ax1 = axes[0]
    positions = np.arange(100)
    
    for model_name in model_order:
        if model_name not in results:
            continue
        mean = results[model_name]['mean_norms']
        std = results[model_name]['std_norms']
        
        ax1.plot(positions, mean, color=COLORS[model_name], label=model_name, linewidth=2)
        ax1.fill_between(positions, mean-std, mean+std, color=COLORS[model_name], alpha=0.15)
    
    ax1.axhline(y=1.0, color='#666666', linestyle='--', linewidth=1.5, label='Target', zorder=0)
    ax1.set_xlabel('Sequence Position')
    ax1.set_ylabel('Output Norm ‖pred‖')
    ax1.set_title('(a) Norm During Processing', fontweight='bold')
    ax1.legend(loc='lower left', framealpha=0.95, fontsize=9)
    ax1.set_ylim(0.2, 1.6)
    ax1.set_xlim(0, 100)
    
    # Panel B: Norm Deviation Bar Chart
    ax2 = axes[1]
    models = [m for m in model_order if m in results]
    deviations = [results[m]['norm_deviation'] for m in models]
    colors = [COLORS[m] for m in models]
    labels = ['GPT', 'DDL', 'mHC', 'Ours']
    
    x_pos = np.arange(len(models))
    bars = ax2.bar(x_pos, deviations, color=colors, edgecolor='black', linewidth=0.8, width=0.7)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(labels, fontweight='bold')
    ax2.set_ylabel('Mean |Norm − 1.0|')
    ax2.set_title('(b) Norm Deviation', fontweight='bold')
    ax2.set_ylim(0, max(deviations) * 1.25)
    
    # Value labels above bars (not on bars)
    for bar, dev in zip(bars, deviations):
        height = bar.get_height()
        ax2.annotate(f'{dev:.3f}',
                    xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 4), textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Panel C: Final Loss Comparison
    ax3 = axes[2]
    
    final_losses = {}
    for model_name, out_dir in GRADNORM_DIRS['stability'].items():
        log = load_training_log(out_dir)
        if log and 'val_loss' in log:
            final_losses[model_name] = log['val_loss'][-1]
    
    models = [m for m in model_order if m in final_losses]
    losses = [final_losses[m] for m in models]
    colors = [COLORS[m] for m in models]
    labels = ['GPT', 'DDL', 'mHC', 'Ours']
    
    x_pos = np.arange(len(models))
    bars = ax3.bar(x_pos, losses, color=colors, edgecolor='black', linewidth=0.8, width=0.7)
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(labels, fontweight='bold')
    ax3.set_ylabel('Final Validation Loss')
    ax3.set_title('(c) Task Performance', fontweight='bold')
    ax3.set_yscale('log')
    
    # Value labels - position carefully to avoid overlap
    for bar, loss in zip(bars, losses):
        height = bar.get_height()
        ax3.annotate(f'{loss:.0e}',
                    xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 4), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    fig.suptitle('Stability Analysis: Norm Preservation Test', 
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('results/journal_fig2_stability.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    print("Saved: results/journal_fig2_stability.png")
    plt.close()


def create_figure_3_ablation():
    """
    Figure 3: Model Comparison (GPT, DDL, mHC, Ours)
    """
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    
    datasets = ['gyroscope', 'stability']
    dataset_titles = ['(a) Gyroscope', '(b) Stability']
    
    all_results = {d: {} for d in datasets}
    
    for dataset in datasets:
        for model_name, out_dir in GRADNORM_DIRS[dataset].items():
            log = load_training_log(out_dir)
            if log and 'val_loss' in log:
                all_results[dataset][model_name] = log['val_loss'][-1]
    
    model_order = ['GPT', 'DDL', 'mHC', 'E∆-MHC-Geo']
    model_labels = ['GPT', 'DDL', 'mHC', 'Ours']
    
    for i, (dataset, title) in enumerate(zip(datasets, dataset_titles)):
        ax = axes[i]
        
        models = [m for m in model_order if m in all_results[dataset]]
        losses = [all_results[dataset][m] for m in models]
        labels = [model_labels[model_order.index(m)] for m in models]
        colors = [COLORS[m] for m in models]
        
        x_pos = np.arange(len(models))
        bars = ax.bar(x_pos, losses, color=colors, edgecolor='black', linewidth=0.8, width=0.7)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, fontsize=11, fontweight='bold')
        ax.set_ylabel('Final Validation Loss' if i == 0 else '')
        ax.set_title(title, fontweight='bold', fontsize=12)
        ax.set_yscale('log')
        
        # Smart label positioning - above bars with enough space
        for bar, loss in zip(bars, losses):
            height = bar.get_height()
            # Position label with offset that scales with log scale
            ax.annotate(f'{loss:.0e}',
                       xy=(bar.get_x() + bar.get_width()/2, height),
                       xytext=(0, 5), textcoords="offset points",
                       ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        # Ensure y-axis has room for labels
        ymin, ymax = ax.get_ylim()
        ax.set_ylim(ymin, ymax * 2)
    
    fig.suptitle('Final Performance Comparison', fontsize=14, fontweight='bold', y=1.0)
    plt.tight_layout()
    plt.savefig('results/journal_fig3_ablation.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    print("Saved: results/journal_fig3_ablation.png")
    plt.close()


def create_figure_4_comprehensive():
    """
    Figure 4: Comprehensive Summary (Main Figure)
    """
    fig = plt.figure(figsize=(10, 11))
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.4, wspace=0.3,
                          height_ratios=[1, 1, 0.9])
    
    datasets = ['gyroscope', 'stability']
    dataset_short = ['Gyroscope', 'Stability']
    model_order = ['GPT', 'DDL', 'mHC', 'E∆-MHC-Geo']
    
    # Row 1: Loss Curves
    for i, (dataset, short) in enumerate(zip(datasets, dataset_short)):
        ax = fig.add_subplot(gs[0, i])
        
        for model_name in model_order:
            out_dir = GRADNORM_DIRS[dataset].get(model_name)
            if not out_dir:
                continue
            log = load_training_log(out_dir)
            if log:
                iters = np.array(log['iter'])
                val_loss = np.array(log['val_loss'])
                ax.plot(iters, val_loss, color=COLORS[model_name], 
                       label=model_name, linewidth=1.8)
        
        ax.set_xlabel('Steps', fontsize=10)
        ax.set_ylabel('Val Loss' if i == 0 else '', fontsize=10)
        ax.set_title(short, fontweight='bold', fontsize=11)
        ax.set_yscale('log')
        ax.set_xlim(0, 2000)
        ax.tick_params(labelsize=9)
        if i == 0:
            ax.legend(loc='upper right', fontsize=8, framealpha=0.9)
    
    fig.text(0.01, 0.88, '(a) Validation Loss', fontsize=11, fontweight='bold')
    
    # Row 2: Gradient Norms
    for i, (dataset, short) in enumerate(zip(datasets, dataset_short)):
        ax = fig.add_subplot(gs[1, i])
        
        for model_name in model_order:
            out_dir = GRADNORM_DIRS[dataset].get(model_name)
            if not out_dir:
                continue
            log = load_training_log(out_dir)
            if log and 'grad_norm' in log and len(log['grad_norm']) > 0:
                grad_norms = np.array(log['grad_norm'])
                grad_iters = np.arange(len(grad_norms)) * 50
                smoothed = smooth_curve(grad_norms, window=7)
                ax.plot(grad_iters, smoothed, color=COLORS[model_name], 
                       label=model_name, linewidth=1.8)
        
        ax.set_xlabel('Steps', fontsize=10)
        ax.set_ylabel('Grad Norm' if i == 0 else '', fontsize=10)
        ax.set_yscale('log')
        ax.set_xlim(0, 2000)
        ax.tick_params(labelsize=9)
    
    fig.text(0.01, 0.57, '(b) Gradient Norm', fontsize=11, fontweight='bold')
    
    # Row 3: Final Performance (grouped bar chart)
    ax_bottom = fig.add_subplot(gs[2, :])
    
    x = np.arange(len(datasets))
    width = 0.18
    
    for i, model_name in enumerate(model_order):
        losses = []
        for dataset in datasets:
            out_dir = GRADNORM_DIRS[dataset].get(model_name)
            if out_dir:
                log = load_training_log(out_dir)
                if log and 'val_loss' in log:
                    losses.append(log['val_loss'][-1])
                else:
                    losses.append(np.nan)
            else:
                losses.append(np.nan)
        
        offset = (i - 1.5) * width
        label = 'Ours' if model_name == 'E∆-MHC-Geo' else model_name
        ax_bottom.bar(x + offset, losses, width, label=label, 
                     color=COLORS[model_name], edgecolor='black', linewidth=0.5)
    
    ax_bottom.set_xlabel('Dataset', fontweight='bold', fontsize=11)
    ax_bottom.set_ylabel('Final Validation Loss', fontweight='bold', fontsize=11)
    ax_bottom.set_xticks(x)
    ax_bottom.set_xticklabels(dataset_short, fontsize=11)
    ax_bottom.legend(loc='upper right', fontsize=10, ncol=4)
    ax_bottom.set_yscale('log')
    ax_bottom.tick_params(labelsize=10)
    
    fig.text(0.01, 0.26, '(c) Final Performance', fontsize=11, fontweight='bold')
    
    fig.suptitle('E∆-MHC-Geo: Comprehensive Experimental Analysis', 
                fontsize=14, fontweight='bold', y=0.98)
    
    plt.savefig('results/journal_fig4_comprehensive.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    print("Saved: results/journal_fig4_comprehensive.png")
    plt.close()


def create_all_figures():
    """Generate all publication figures."""
    print("=" * 60)
    print("Generating Publication-Quality Figures (Journal Standard)")
    print("=" * 60)
    print()
    
    print("Creating Figure 1: Training Dynamics...")
    create_figure_1_training_dynamics()
    
    print("\nCreating Figure 2: Stability Analysis...")
    try:
        create_figure_2_stability_analysis()
    except Exception as e:
        print(f"  Error: {e}")
    
    print("\nCreating Figure 3: Performance Comparison...")
    create_figure_3_ablation()
    
    # Figure 4 removed - redundant with Figures 1 and 3
    
    print("\n" + "=" * 60)
    print("All figures saved at 300 DPI!")
    print("=" * 60)


if __name__ == '__main__':
    create_all_figures()
