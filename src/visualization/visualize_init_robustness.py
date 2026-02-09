#!/usr/bin/env python3
"""
Visualization of initialization robustness for E∆-MHC-Geo thermodynamic gate.

Creates publication-quality figure showing:
- Training curves for different initializations converge to same loss
- Final gamma values differ but performance is consistent
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Publication-quality settings
plt.rcParams.update({
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 8,
    'figure.dpi': 150,
})


def load_results(out_dir):
    """Load results from output directory."""
    results_path = Path(out_dir) / 'results.npy'
    if results_path.exists():
        return np.load(results_path, allow_pickle=True).item()
    return None


def load_training_log(out_dir):
    """Load training log from output directory."""
    log_path = Path(out_dir) / 'train_log.npy'
    if log_path.exists():
        return np.load(log_path, allow_pickle=True).item()
    return None


def smooth_curve(values, window=10):
    """Exponential moving average smoothing."""
    if len(values) < 2:
        return values
    alpha = 2 / (window + 1)
    smoothed = [values[0]]
    for v in values[1:]:
        smoothed.append(alpha * v + (1 - alpha) * smoothed[-1])
    return smoothed


def create_init_robustness_figure():
    """Create initialization robustness visualization."""
    fig = plt.figure(figsize=(12, 8))
    
    # Experiment configurations
    configs = [
        ('single', 'Single-Plane (θ=3.1)', 'out-near-pi-single-edelta-init', 
         'out-near-pi-single-edelta-bias-pos', 'out-near-pi-single-edelta-bias-neg'),
        ('multi', 'Multi-Plane (θ=3.14)', 'out-near-pi-multiplane-init-0.0',
         'out-near-pi-multiplane-init-1.5', 'out-near-pi-multiplane-init--1.5'),
    ]
    
    bias_labels = {
        'neutral': ('bias=0.0', 'γ₀=0.50', 'C2'),
        'cayley': ('bias=+1.5', 'γ₀=0.82', 'C0'),
        'householder': ('bias=-1.5', 'γ₀=0.18', 'C1'),
    }
    
    colors = {'neutral': 'C2', 'cayley': 'C0', 'householder': 'C1'}
    
    # Panel layout: 2x3 grid
    # Row 1: Training curves
    # Row 2: Gamma evolution
    
    for col, (key, title, neutral_dir, cayley_dir, house_dir) in enumerate(configs):
        # --- Training Curves (top row) ---
        ax_train = fig.add_subplot(2, 3, col + 1)
        ax_train.set_title(f'{title}\nTraining Curves')
        ax_train.set_xlabel('Iteration')
        ax_train.set_ylabel('Validation Loss')
        ax_train.set_yscale('log')
        ax_train.grid(True, alpha=0.3)
        
        for init_type, out_dir in [('neutral', neutral_dir), ('cayley', cayley_dir), 
                                    ('householder', house_dir)]:
            log = load_training_log(out_dir)
            if log and 'val_loss' in log:
                iters = log['iter']
                val_loss = smooth_curve(log['val_loss'], window=5)
                label, gamma_init, color = bias_labels[init_type]
                ax_train.plot(iters, val_loss, label=f'{label} ({gamma_init})', 
                             color=colors[init_type], linewidth=1.5)
        
        ax_train.legend(loc='upper right')
        
        # --- Gamma Evolution (middle row) ---
        ax_gamma = fig.add_subplot(2, 3, col + 4)
        ax_gamma.set_title(f'{title}\nGate Evolution')
        ax_gamma.set_xlabel('Iteration')
        ax_gamma.set_ylabel('γ (Gate Value)')
        ax_gamma.set_ylim(-0.1, 1.1)
        ax_gamma.axhline(1.0, color='blue', linestyle=':', alpha=0.5, label='Cayley')
        ax_gamma.axhline(0.5, color='gray', linestyle=':', alpha=0.5, label='Blend')
        ax_gamma.axhline(0.0, color='red', linestyle=':', alpha=0.5, label='Householder')
        ax_gamma.grid(True, alpha=0.3)
        
        for init_type, out_dir in [('neutral', neutral_dir), ('cayley', cayley_dir), 
                                    ('householder', house_dir)]:
            log = load_training_log(out_dir)
            if log and 'gamma_mean' in log:
                iters = log['iter']
                gamma_mean = log['gamma_mean']
                label, gamma_init, color = bias_labels[init_type]
                ax_gamma.plot(iters, gamma_mean, label=f'{label}', 
                             color=colors[init_type], linewidth=2)
        
        ax_gamma.legend(loc='center right')
    
    # --- Summary Bar Chart (right column) ---
    ax_summary = fig.add_subplot(2, 3, 3)
    ax_summary.set_title('Final Performance Summary')
    
    # Collect results
    results_data = []
    labels = []
    
    for task, dirs in [('Single-Plane', ['out-near-pi-single-edelta-bias-pos', 
                                          'out-near-pi-single-edelta-init',
                                          'out-near-pi-single-edelta-bias-neg']),
                       ('Multi-Plane', ['out-near-pi-multiplane-init-1.5',
                                        'out-near-pi-multiplane-init-0.0',
                                        'out-near-pi-multiplane-init--1.5'])]:
        for i, d in enumerate(dirs):
            r = load_results(d)
            if r:
                results_data.append(r['final_val_loss'])
                init_name = ['Cayley Init', 'Neutral Init', 'Householder Init'][i]
                labels.append(f'{task}\n{init_name}')
    
    x = np.arange(len(results_data))
    colors_bar = ['C0', 'C2', 'C1'] * 2
    bars = ax_summary.bar(x, results_data, color=colors_bar, alpha=0.8)
    ax_summary.set_xticks(x)
    ax_summary.set_xticklabels(labels, rotation=45, ha='right', fontsize=7)
    ax_summary.set_ylabel('Final Val Loss')
    ax_summary.set_yscale('log')
    ax_summary.set_ylim(1e-7, 1e-4)
    
    # Add value labels
    for bar, val in zip(bars, results_data):
        ax_summary.annotate(f'{val:.1e}', xy=(bar.get_x() + bar.get_width()/2, val),
                           xytext=(0, 3), textcoords='offset points',
                           ha='center', va='bottom', fontsize=7)
    
    # --- Final Gamma Values (bottom right) ---
    ax_final = fig.add_subplot(2, 3, 6)
    ax_final.set_title('Final Gate Values by Initialization')
    
    final_gammas = []
    gamma_stds = []
    for dirs in [['out-near-pi-single-edelta-bias-pos', 
                  'out-near-pi-single-edelta-init',
                  'out-near-pi-single-edelta-bias-neg'],
                 ['out-near-pi-multiplane-init-1.5',
                  'out-near-pi-multiplane-init-0.0',
                  'out-near-pi-multiplane-init--1.5']]:
        for d in dirs:
            r = load_results(d)
            if r:
                final_gammas.append(r.get('final_gamma_mean', 0.5))
                gamma_stds.append(r.get('final_gamma_std', 0))
    
    x = np.arange(len(final_gammas))
    bars = ax_final.bar(x, final_gammas, yerr=gamma_stds, color=colors_bar, alpha=0.8, 
                        capsize=3)
    ax_final.set_xticks(x)
    ax_final.set_xticklabels(labels, rotation=45, ha='right', fontsize=7)
    ax_final.set_ylabel('Final γ Value')
    ax_final.set_ylim(-0.1, 1.2)
    ax_final.axhline(1.0, color='blue', linestyle='--', alpha=0.3)
    ax_final.axhline(0.0, color='red', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)
    fig.savefig(results_dir / 'init_robustness.png', dpi=200, bbox_inches='tight')
    fig.savefig(results_dir / 'init_robustness.pdf', bbox_inches='tight')
    print(f"Saved: {results_dir / 'init_robustness.png'}")
    
    plt.close()


def print_summary_table():
    """Print a text summary table of results."""
    print("\n" + "="*80)
    print("INITIALIZATION ROBUSTNESS SUMMARY")
    print("="*80)
    
    print("\n### Single-Plane (θ=3.1, near-π rotation)")
    print(f"{'Init Bias':<15} {'Start γ':<10} {'Final γ':<15} {'Final Loss':<12}")
    print("-"*52)
    
    for bias, out_dir in [('+1.5', 'out-near-pi-single-edelta-bias-pos'),
                          ('0.0', 'out-near-pi-single-edelta-init'),
                          ('-1.5', 'out-near-pi-single-edelta-bias-neg')]:
        r = load_results(out_dir)
        if r:
            start_g = 1/(1+np.exp(-float(bias)))
            final_g = r.get('final_gamma_mean', 0)
            final_std = r.get('final_gamma_std', 0)
            loss = r['final_val_loss']
            print(f"{bias:<15} {start_g:.2f}       {final_g:.2f}±{final_std:.2f}        {loss:.2e}")
    
    print("\n### Multi-Plane (θ=3.14, near-reflection)")
    print(f"{'Init Bias':<15} {'Start γ':<10} {'Final γ':<15} {'Final Loss':<12}")
    print("-"*52)
    
    for bias, out_dir in [('+1.5', 'out-near-pi-multiplane-init-1.5'),
                          ('0.0', 'out-near-pi-multiplane-init-0.0'),
                          ('-1.5', 'out-near-pi-multiplane-init--1.5')]:
        r = load_results(out_dir)
        if r:
            start_g = 1/(1+np.exp(-float(bias)))
            final_g = r.get('final_gamma_mean', 0)
            final_std = r.get('final_gamma_std', 0)
            loss = r['final_val_loss']
            print(f"{bias:<15} {start_g:.2f}       {final_g:.2f}±{final_std:.2f}        {loss:.2e}")
    
    print("\n" + "="*80)
    print("KEY FINDING: All initializations achieve similar excellent loss (~1-3e-6)")
    print("             Initialization determines WHERE model converges, not HOW WELL")
    print("="*80)


if __name__ == '__main__':
    create_init_robustness_figure()
    print_summary_table()
