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
    'JPmHC': '#CC79A7',      # Reddish Purple
    'E∆-MHC-Geo': '#009E73', # Bluish Green
}

LINE_STYLES = {
    'GPT': '-',
    'DDL': '-',
    'mHC': '-',
    'JPmHC': '--',
    'E∆-MHC-Geo': '-',
}

# Data directories (gyroscope and stability only - correction handled separately in experiments/)
GRADNORM_DIRS = {
    'gyroscope': {
        'GPT': 'out-matched/gyroscope-gpt',
        'DDL': 'out-matched/gyroscope-ddl',
        'mHC': 'out-matched/gyroscope-mhc',
        'JPmHC': 'out-matched/gyroscope-jpmhc',
        'E∆-MHC-Geo': 'out-matched/gyroscope-proposed',
    },
    'stability': {
        'GPT': 'out-matched/stability-gpt',
        'DDL': 'out-matched/stability-ddl',
        'mHC': 'out-matched/stability-mhc',
        'JPmHC': 'out-matched/stability-jpmhc',
        'E∆-MHC-Geo': 'out-matched/stability-proposed',
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
    
    model_order = ['GPT', 'DDL', 'mHC', 'JPmHC', 'E∆-MHC-Geo']
    
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
    from src.models.jpmhc import GPT as JPmHCGPT, GPTConfig as JPmHCConfig
    from src.models.edelta_hybrid import GPT as EdeltaGPT, GPTConfig as EdeltaConfig
    
    results = {}
    model_order = ['GPT', 'DDL', 'mHC', 'JPmHC', 'E∆-MHC-Geo']
    
    for model_name, out_dir in GRADNORM_DIRS['stability'].items():
        ckpt_path = os.path.join(out_dir, 'ckpt.pt')
        if not os.path.exists(ckpt_path):
            continue
        
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        saved_config = checkpoint['config']
        model_type = saved_config['model_type']
        
        # Detect actual n_layer from state_dict (may differ from config due to parameter matching)
        layers = set()
        block_size = 128  # default
        for k, v in checkpoint['model'].items():
            if 'pos_emb' in k:
                block_size = v.shape[1]
            if '.h.' in k:
                layer_num = k.split('.h.')[1].split('.')[0]
                layers.add(int(layer_num))
        actual_n_layer = max(layers) + 1 if layers else saved_config['n_layer']
        
        model_map = {
            'gpt2': (BaselineGPT, BaselineConfig),
            'ddl': (DDLGPT, DDLConfig),
            'mhc': (mHCGPT, mHCConfig),
            'jpmhc': (JPmHCGPT, JPmHCConfig),
            'edelta': (EdeltaGPT, EdeltaConfig),
        }
        
        GPTClass, ConfigClass = model_map[model_type]
        config_args = {
            'n_layer': actual_n_layer,  # Use actual layer count from state_dict
            'n_head': saved_config['n_head'],
            'n_embd': saved_config['n_embd'],
            'dropout': saved_config['dropout'],
            'bias': False,
            'block_size': block_size,
            'vocab_size': 1
        }
        if model_type in ['mhc', 'jpmhc', 'edelta']:
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
    labels = ['Ours' if m == 'E∆-MHC-Geo' else m for m in models]
    
    x_pos = np.arange(len(models))
    bars = ax2.bar(x_pos, deviations, color=colors, edgecolor='black', linewidth=0.8, width=0.6)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(labels, fontweight='bold', fontsize=8)
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
    labels = ['Ours' if m == 'E∆-MHC-Geo' else m for m in models]
    
    x_pos = np.arange(len(models))
    bars = ax3.bar(x_pos, losses, color=colors, edgecolor='black', linewidth=0.8, width=0.6)
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(labels, fontweight='bold', fontsize=8)
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
    
    model_order = ['GPT', 'DDL', 'mHC', 'JPmHC', 'E∆-MHC-Geo']
    model_labels = ['GPT', 'DDL', 'mHC', 'JPmHC', 'Ours']
    
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
    model_order = ['GPT', 'DDL', 'mHC', 'JPmHC', 'E∆-MHC-Geo']
    
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
    width = 0.15
    
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
        
        offset = (i - 2) * width
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


def create_figure_4_reflection_aha_moment():
    """
    Figure 4: Reflection Experiment "Aha!" Moment Visualization
    Following arXiv:2601.00514v1 "Illusion of Insight" methodology.
    
    Uses the proper model implementations from train_reflection.py to ensure
    correct parameter convergence (DDL β→2, E∆-MHC-Geo γ→0).
    """
    # Import models and utilities from train_reflection.py
    from src.training.train_reflection import SimpleDDL, SimpleHybrid, train_with_trajectory
    from src.data.reflection import generate_negation_data
    
    # === Run experiment ===
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dim, n_samples = 64, 500
    
    torch.manual_seed(42)
    train_x, train_y = generate_negation_data(n_samples, dim, device)
    val_x, val_y = generate_negation_data(500, dim, device)
    
    print("  Training DDL...")
    ddl = SimpleDDL(dim).to(device)
    ddl_result = train_with_trajectory(ddl, 'DDL', train_x, train_y, val_x, val_y, 
                                        max_iters=3000, track_interval=30, verbose=False)
    ddl_traj = ddl_result['trajectory']
    
    # Note: Unbiased init (γ=0.5) gets stuck because regularization gradient is zero there!
    # Model uses symmetry-breaking init (γ≈0.38) - see train_reflection.py for detailed explanation
    print("  Training E∆-MHC-Geo (symmetry-breaking init, strong reg=1.0)...")
    hybrid = SimpleHybrid(dim, gate_reg_weight=1.0).to(device)
    hybrid_result = train_with_trajectory(hybrid, 'E∆-MHC-Geo', train_x, train_y, val_x, val_y,
                                          max_iters=5000, track_interval=50, verbose=False)
    hybrid_traj = hybrid_result['trajectory']
    
    # === Create figure ===
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    
    # (a) DDL β trajectory with uncertainty band
    ax1 = axes[0, 0]
    beta_mean = np.array(ddl_traj['beta_mean'])
    beta_std = np.array(ddl_traj['beta_std'])
    ax1.plot(ddl_traj['iter'], beta_mean, color=COLORS['DDL'], linewidth=2.5, label='β (mean)')
    ax1.fill_between(ddl_traj['iter'], beta_mean - beta_std, beta_mean + beta_std,
                     color=COLORS['DDL'], alpha=0.2, label='±1 std')
    ax1.axhline(y=2.0, color='gray', linestyle='--', linewidth=1.5, label='β = 2 (exact reflection)')
    ax1.set_xlabel('Training Iteration')
    ax1.set_ylabel('β Value', color=COLORS['DDL'])
    ax1.set_title('(a) DDL: β Trajectory (500 samples)', fontweight='bold')
    ax1.set_ylim(0.8, 2.2)
    ax1.legend(loc='lower right')
    
    # (b) E∆-MHC-Geo γ trajectory with uncertainty band
    ax2 = axes[0, 1]
    gate_mean = np.array(hybrid_traj['gate_mean'])
    gate_std = np.array(hybrid_traj['gate_std'])
    ax2.plot(hybrid_traj['iter'], gate_mean, color=COLORS['E∆-MHC-Geo'], linewidth=2.5, label='γ (mean)')
    ax2.fill_between(hybrid_traj['iter'], gate_mean - gate_std, gate_mean + gate_std,
                     color=COLORS['E∆-MHC-Geo'], alpha=0.2, label='±1 std')
    ax2.axhline(y=0.0, color='gray', linestyle='--', linewidth=1.5, label='γ = 0 (Householder)')
    ax2.set_xlabel('Training Iteration')
    ax2.set_ylabel('Gate Value (γ)', color=COLORS['E∆-MHC-Geo'])
    ax2.set_title('(b) E∆-MHC-Geo: Gate Trajectory (500 samples)', fontweight='bold')
    ax2.legend(loc='upper right')
    
    # (c) DDL: Accuracy vs β scatter - THE "AHA!" MOMENT
    ax3 = axes[1, 0]
    sc1 = ax3.scatter(ddl_traj['beta_mean'], ddl_traj['negation_accuracy'],
                     c=ddl_traj['iter'], cmap='plasma', s=60, alpha=0.85, edgecolors='white', linewidths=0.5)
    ax3.axvline(x=2.0, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)
    ax3.axhline(y=0.95, color='#27AE60', linestyle=':', linewidth=1.5, alpha=0.8)
    ax3.set_xlabel('β Value')
    ax3.set_ylabel('Negation Accuracy')
    ax3.set_title('(c) DDL: Accuracy vs β ("Aha!" Moment)', fontweight='bold')
    ax3.set_xlim(0.8, 2.15)
    plt.colorbar(sc1, ax=ax3, label='Iteration')
    
    # (d) E∆-MHC-Geo: Accuracy vs γ scatter - THE "AHA!" MOMENT
    ax4 = axes[1, 1]
    sc2 = ax4.scatter(hybrid_traj['gate_mean'], hybrid_traj['negation_accuracy'],
                     c=hybrid_traj['iter'], cmap='plasma', s=60, alpha=0.85, edgecolors='white', linewidths=0.5)
    ax4.axvline(x=0.0, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)
    ax4.axhline(y=0.95, color='#27AE60', linestyle=':', linewidth=1.5, alpha=0.8)
    ax4.set_xlabel('Gate Value (γ)')
    ax4.set_ylabel('Negation Accuracy')
    ax4.set_title('(d) E∆-MHC-Geo: Accuracy vs γ ("Aha!" Moment)', fontweight='bold')
    plt.colorbar(sc2, ax=ax4, label='Iteration')
    
    fig.suptitle('Reflection Experiment: "Aha!" Moment Visualization\n'
                '(Following arXiv:2601.00514v1 "Illusion of Insight" Methodology)',
                fontsize=14, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig('results/reflection_aha_moment.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    print("  Saved: results/reflection_aha_moment.png")
    
    # Also save to assets
    plt.savefig('assets/reflection_aha_moment.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print("  Saved: assets/reflection_aha_moment.png")
    plt.close()
    
    # Print results
    print(f"  Results: DDL β={ddl_traj['beta_mean'][-1]:.4f}, E∆ γ={hybrid_traj['gate_mean'][-1]:.4f}")


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
    
    print("\nCreating Figure 4: Reflection 'Aha!' Moment...")
    try:
        create_figure_4_reflection_aha_moment()
    except Exception as e:
        print(f"  Error: {e}")
    
    print("\n" + "=" * 60)
    print("All figures saved at 300 DPI!")
    print("=" * 60)


if __name__ == '__main__':
    create_all_figures()
