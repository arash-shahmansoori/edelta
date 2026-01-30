#!/usr/bin/env python3
"""
Reflection Training: Direct Test of Geometric Operators

============================================================================
The PUREST test of geometric inductive bias: learning y = -x (negation)
============================================================================

This test directly validates the CORE CLAIM of geometric models:
- DDL's operator A = I - β·kk^T achieves exact reflection when β=2
- E∆-MHC-Geo's Householder component achieves reflection when gate → 0

We measure:
1. Sample efficiency (fewer samples needed for geometric models)
2. Parameter convergence (β → 2 for DDL, gate → 0 for Hybrid)
3. Final accuracy (cosine similarity with -x)

Usage (run from project root with uv):
    # Run sample efficiency test (main experiment)
    uv run src/training/train_reflection.py --mode sample_efficiency
    
    # Run single test with specific sample size
    uv run src/training/train_reflection.py --mode single --n_samples 100
    
    # Generate publication figures
    uv run src/training/train_reflection.py --mode sample_efficiency --save_figures

Author: Arash Shahmansoori (2026)
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.data.reflection import generate_negation_data, load_reflection_dataset

# Flush print for real-time output
sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None
os.environ['PYTHONUNBUFFERED'] = '1'


def flush_print(*args, **kwargs):
    print(*args, **kwargs)
    sys.stdout.flush()


# =============================================================================
# PUBLICATION-QUALITY FIGURE SETTINGS
# =============================================================================

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
    'lines.markersize': 8,
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

# Color-blind friendly palette
COLORS = {
    'GPT': '#0072B2',        # Blue
    'DDL': '#D55E00',        # Vermillion
    'mHC': '#E69F00',        # Amber
    'E∆-MHC-Geo': '#009E73', # Bluish Green
}

MARKERS = {
    'GPT': 'o',
    'DDL': 's',
    'mHC': '^',
    'E∆-MHC-Geo': 'D',
}


# =============================================================================
# MINIMAL MODELS (Direct geometric operators, no wrapper overhead)
# =============================================================================

HIDDEN_DIM = 128  # Ensures ~30-50k params per model for fair comparison


class SimpleGPT(nn.Module):
    """
    Minimal GPT-style model: y = x + MLP(x)
    
    To achieve y = -x, must learn MLP(x) = -2x (linear approximation).
    """
    
    def __init__(self, dim: int, hidden_dim: int = HIDDEN_DIM):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )
    
    def forward(self, x):
        return x + self.mlp(x)


class SimpleDDL(nn.Module):
    """
    DDL-style model: A(x) = x - β·(k^T·x)·k
    
    Reference: arXiv:2601.00417
    
    For y = -x: need β → 2 and k → x/||x||
    This is exact Householder reflection when β = 2.
    """
    
    def __init__(self, dim: int, hidden_dim: int = HIDDEN_DIM):
        super().__init__()
        self.k_net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )
        self.beta_net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )
        self.beta_scale = 2.0
        
    def forward(self, x):
        k = F.normalize(self.k_net(x), dim=-1)
        beta = self.beta_scale * self.beta_net(x)
        k_dot_x = (k * x).sum(dim=-1, keepdim=True)
        return x - beta * k_dot_x * k
    
    def get_beta(self, x):
        return self.beta_scale * self.beta_net(x)


class SimpleHybrid(nn.Module):
    """
    E∆-MHC-Geo Hybrid: G_γ(x) = γ·Q(x)·x + (1-γ)·H₂(k)·x
    
    Components:
    - Q(x): Data-dependent Cayley rotation (SO(n), det=+1)
    - H₂(k): Householder reflection (det=-1, β=2 FIXED per Theorem 7)
    - γ: Thermodynamic gate with midpoint collapse regularization
    
    For y = -x: gate should converge to γ → 0 (use Householder)
    
    Reference: RESEARCH_V3.md Section 5.3, 6.3
    """
    
    def __init__(self, dim: int, hidden_dim: int = HIDDEN_DIM, gate_reg_weight: float = 0.5):
        super().__init__()
        self.gate_reg_weight = gate_reg_weight
        
        # Cayley rotation parameters
        self.u_net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )
        self.v_net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )
        self.cayley_beta_net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Softplus(),
        )
        
        # Householder reflection (β=2 FIXED)
        self.k_net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )
        self.register_buffer('householder_beta', torch.tensor(2.0))
        
        # Thermodynamic gate
        self.gate_linear = nn.Linear(dim, 1)
        nn.init.constant_(self.gate_linear.bias, -2.0)  # Init γ ≈ 0.12
        
        self._gate_reg_loss = None
    
    def cayley_rotation(self, x, u, v, beta):
        """Exact Cayley: Q = (I + M)^{-1}(I - M), M = (β/2)(uv^T - vu^T)"""
        dim = x.shape[-1]
        A = torch.einsum('...i,...j->...ij', u, v) - torch.einsum('...i,...j->...ij', v, u)
        M = (beta / 2).unsqueeze(-1) * A
        I = torch.eye(dim, device=x.device, dtype=x.dtype)
        Q = torch.linalg.solve(I + M, I - M)
        return torch.einsum('...ij,...j->...i', Q, x)
    
    def householder_reflection(self, x, k):
        """H·x = x - 2·(k^T·x)·k (β=2 fixed)"""
        k_dot_x = (k * x).sum(dim=-1, keepdim=True)
        return x - self.householder_beta * k_dot_x * k
    
    def forward(self, x):
        # Cayley parameters
        u = F.normalize(self.u_net(x), dim=-1)
        v = F.normalize(self.v_net(x), dim=-1)
        v = v - (u * v).sum(dim=-1, keepdim=True) * u
        v = F.normalize(v, dim=-1)
        cayley_beta = self.cayley_beta_net(x)
        
        # Householder direction
        k = F.normalize(self.k_net(x), dim=-1)
        
        # Gate with midpoint collapse regularization
        gamma = torch.sigmoid(self.gate_linear(x))
        self._gate_reg_loss = self.gate_reg_weight * 4 * gamma * (1 - gamma)
        self._gate_reg_loss = self._gate_reg_loss.mean()
        
        # Blend operators
        cayley_out = self.cayley_rotation(x, u, v, cayley_beta)
        householder_out = self.householder_reflection(x, k)
        
        return gamma * cayley_out + (1 - gamma) * householder_out
    
    def get_gate(self, x):
        return torch.sigmoid(self.gate_linear(x))
    
    def get_gate_regularization_loss(self):
        return self._gate_reg_loss if self._gate_reg_loss is not None else 0.0


class SimpleMHC(nn.Module):
    """
    DeepSeek mHC: x_{l+1} = H_res·x_l + H_post^T·F(H_pre·x_l)
    
    Reference: arXiv:2512.24880
    
    Limitation: H_res is doubly stochastic (all positive entries)
    → Can only MIX streams, cannot negate → Poor at y = -x
    """
    
    def __init__(self, dim: int, hidden_dim: int = HIDDEN_DIM, n_streams: int = 4):
        super().__init__()
        self.dim = dim
        self.n_streams = n_streams
        self.d_stream = dim // n_streams
        self.n_sinkhorn_iters = 10
        
        # Data-dependent coefficient computation
        self.phi_pre = nn.Linear(dim, n_streams, bias=False)
        self.phi_post = nn.Linear(dim, n_streams, bias=False)
        self.phi_res = nn.Linear(dim, n_streams * n_streams, bias=False)
        
        self.b_pre = nn.Parameter(torch.zeros(n_streams))
        self.b_post = nn.Parameter(torch.zeros(n_streams))
        self.b_res = nn.Parameter(torch.eye(n_streams) * 2.0)
        
        self.alpha_pre = nn.Parameter(torch.tensor(0.01))
        self.alpha_post = nn.Parameter(torch.tensor(0.01))
        self.alpha_res = nn.Parameter(torch.tensor(0.01))
        
        # Layer function
        mlp_hidden = 350
        self.mlp = nn.Sequential(
            nn.Linear(self.d_stream, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, self.d_stream),
        )
    
    def sinkhorn_knopp(self, M):
        M = torch.exp(M)
        for _ in range(self.n_sinkhorn_iters):
            M = M / (M.sum(dim=-1, keepdim=True) + 1e-8)
            M = M / (M.sum(dim=-2, keepdim=True) + 1e-8)
        return M
    
    def forward(self, x):
        B = x.shape[0] if x.dim() > 1 else 1
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        x_streams = x.view(B, self.n_streams, self.d_stream)
        
        H_tilde_pre = self.alpha_pre * self.phi_pre(x) + self.b_pre
        H_tilde_post = self.alpha_post * self.phi_post(x) + self.b_post
        H_tilde_res = self.alpha_res * self.phi_res(x).view(B, self.n_streams, self.n_streams) + self.b_res
        
        H_pre = torch.sigmoid(H_tilde_pre)
        H_post = 2.0 * torch.sigmoid(H_tilde_post)
        H_res = self.sinkhorn_knopp(H_tilde_res)
        
        x_mixed = torch.einsum('bij,bjd->bid', H_res, x_streams)
        x_agg = H_pre.unsqueeze(-1) * x_streams
        x_transformed = self.mlp(x_agg)
        x_broadcast = H_post.unsqueeze(-1) * x_transformed
        x_out = x_mixed + x_broadcast
        
        out = x_out.view(B, self.dim)
        return out.squeeze(0) if B == 1 and x.dim() == 1 else out


# =============================================================================
# TRAINING AND ANALYSIS
# =============================================================================

def train_and_analyze(
    model: nn.Module,
    model_name: str,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    val_x: torch.Tensor,
    val_y: torch.Tensor,
    max_iters: int = 2000,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    eval_interval: int = 200,
    verbose: bool = True,
):
    """Train model and analyze learned parameters."""
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    n_train = len(train_x)
    n_params = sum(p.numel() for p in model.parameters())
    
    if verbose:
        flush_print(f"\n{'='*60}")
        flush_print(f"Model: {model_name} ({n_params:,} params)")
    
    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': [], 'val_cos': [], 'iter': []}
    
    for iter_num in range(max_iters):
        model.train()
        
        idx = torch.randint(0, n_train, (min(batch_size, n_train),))
        xb, yb = train_x[idx], train_y[idx]
        
        pred = model(xb)
        loss = F.mse_loss(pred, yb)
        
        if hasattr(model, 'get_gate_regularization_loss'):
            gate_reg = model.get_gate_regularization_loss()
            if gate_reg is not None and gate_reg > 0:
                loss = loss + gate_reg
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (iter_num + 1) % eval_interval == 0 or iter_num == 0:
            model.eval()
            with torch.no_grad():
                val_pred = model(val_x)
                val_loss = F.mse_loss(val_pred, val_y)
                val_cos = F.cosine_similarity(val_pred, val_y, dim=-1).mean().item()
                
                history['train_loss'].append(loss.item())
                history['val_loss'].append(val_loss.item())
                history['val_cos'].append(val_cos)
                history['iter'].append(iter_num + 1)
                
                if val_loss.item() < best_val_loss:
                    best_val_loss = val_loss.item()
                
                if verbose:
                    flush_print(f"  Step {iter_num+1:4d}: loss={loss.item():.6f}, "
                               f"val_loss={val_loss.item():.6f}, cos_sim={val_cos:.4f}")
    
    # Final analysis
    model.eval()
    with torch.no_grad():
        final_pred = model(val_x)
        final_loss = F.mse_loss(final_pred, val_y).item()
        final_cos = F.cosine_similarity(final_pred, val_y, dim=-1).mean().item()
        negation_accuracy = F.cosine_similarity(final_pred, -val_x, dim=-1).mean().item()
    
    # Analyze learned parameters
    param_info = {}
    if hasattr(model, 'get_beta'):
        with torch.no_grad():
            betas = model.get_beta(val_x[:100])
            param_info['beta_mean'] = betas.mean().item()
            param_info['beta_std'] = betas.std().item()
            if verbose:
                flush_print(f"  β: mean={param_info['beta_mean']:.4f}, std={param_info['beta_std']:.4f}")
                if param_info['beta_mean'] > 1.8:
                    flush_print(f"  ✓ β → 2 (learned full reflection!)")
    
    if hasattr(model, 'get_gate'):
        with torch.no_grad():
            gates = model.get_gate(val_x[:100])
            param_info['gate_mean'] = gates.mean().item()
            param_info['gate_std'] = gates.std().item()
            if verbose:
                flush_print(f"  Gate: mean={param_info['gate_mean']:.4f}, std={param_info['gate_std']:.4f}")
                if param_info['gate_mean'] < 0.2:
                    flush_print(f"  ✓ Gate → 0 (learned to use Householder!)")
    
    return {
        'model_name': model_name,
        'n_params': n_params,
        'final_loss': final_loss,
        'final_cos': final_cos,
        'negation_accuracy': negation_accuracy,
        'best_val_loss': best_val_loss,
        'param_info': param_info,
        'history': history,
    }


# =============================================================================
# SAMPLE EFFICIENCY TEST
# =============================================================================

def run_sample_efficiency_test(
    dim: int,
    device: str,
    max_iters: int = 1000,
    sample_sizes: list = None,
    save_figures: bool = False,
    output_dir: str = 'results',
):
    """
    Test sample efficiency: how many samples needed to learn y = -x?
    
    Geometric models should need FEWER samples due to inductive bias.
    """
    
    if sample_sizes is None:
        sample_sizes = [10, 25, 50, 100, 200]
    
    flush_print("\n" + "="*80)
    flush_print("SAMPLE EFFICIENCY TEST: Learning y = -x")
    flush_print("="*80)
    flush_print("Geometric models should need fewer samples.\n")
    
    # Fixed validation set
    val_x, val_y = generate_negation_data(500, dim, device, seed=999)
    
    results = {size: {} for size in sample_sizes}
    model_names = ['GPT', 'DDL', 'mHC', 'E∆-MHC-Geo']
    
    for n_samples in sample_sizes:
        flush_print(f"\n{'='*60}")
        flush_print(f"Training with {n_samples} samples")
        flush_print("="*60)
        
        train_x, train_y = generate_negation_data(n_samples, dim, device, seed=42)
        
        models = {
            'GPT': SimpleGPT(dim).to(device),
            'DDL': SimpleDDL(dim).to(device),
            'mHC': SimpleMHC(dim).to(device),
            'E∆-MHC-Geo': SimpleHybrid(dim).to(device),
        }
        
        for name in model_names:
            model = models[name]
            result = train_and_analyze(
                model=model,
                model_name=name,
                train_x=train_x,
                train_y=train_y,
                val_x=val_x,
                val_y=val_y,
                max_iters=max_iters,
                learning_rate=1e-3,
                eval_interval=max_iters // 5,
            )
            results[n_samples][name] = result
    
    # Print summary table
    flush_print("\n" + "="*80)
    flush_print("SAMPLE EFFICIENCY RESULTS (Negation Accuracy)")
    flush_print("="*80)
    
    flush_print(f"\n{'Samples':<10}", end='')
    for name in model_names:
        flush_print(f"{name:<16}", end='')
    flush_print()
    flush_print("-" * (10 + 16 * len(model_names)))
    
    for n_samples in sample_sizes:
        flush_print(f"{n_samples:<10}", end='')
        for name in model_names:
            cos = results[n_samples][name]['negation_accuracy']
            flush_print(f"{cos:>8.4f}        ", end='')
        flush_print()
    
    # Find minimum samples for 95% accuracy
    flush_print("\n" + "="*80)
    flush_print("SAMPLES NEEDED FOR 95% NEGATION ACCURACY")
    flush_print("="*80)
    
    min_samples = {}
    for name in model_names:
        for n_samples in sample_sizes:
            if results[n_samples][name]['negation_accuracy'] >= 0.95:
                min_samples[name] = n_samples
                flush_print(f"{name:<16}: {n_samples} samples")
                break
        else:
            min_samples[name] = '>200'
            flush_print(f"{name:<16}: >200 samples needed")
    
    if save_figures:
        create_reflection_figures(results, sample_sizes, model_names, output_dir)
    
    return results, min_samples


# =============================================================================
# PUBLICATION-QUALITY FIGURES
# =============================================================================

def create_reflection_figures(results, sample_sizes, model_names, output_dir='results'):
    """Generate publication-quality figures for reflection experiment."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Figure 1: Sample Efficiency Curve
    fig, ax = plt.subplots(figsize=(8, 5))
    
    for name in model_names:
        accuracies = [results[n][name]['negation_accuracy'] for n in sample_sizes]
        ax.plot(sample_sizes, accuracies, 
               marker=MARKERS[name], color=COLORS[name], 
               label=name, linewidth=2, markersize=8)
    
    ax.axhline(y=0.95, color='gray', linestyle='--', linewidth=1, alpha=0.7, label='95% threshold')
    ax.set_xlabel('Number of Training Samples')
    ax.set_ylabel('Negation Accuracy (Cosine Similarity)')
    ax.set_title('Sample Efficiency: Learning y = -x', fontweight='bold')
    ax.set_ylim(-0.1, 1.05)
    ax.legend(loc='lower right', framealpha=0.95)
    ax.set_xscale('log')
    ax.set_xticks(sample_sizes)
    ax.set_xticklabels(sample_sizes)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/reflection_sample_efficiency.png', dpi=300, 
                bbox_inches='tight', facecolor='white')
    flush_print(f"\nSaved: {output_dir}/reflection_sample_efficiency.png")
    plt.close()
    
    # Figure 2: Parameter Analysis (β for DDL, Gate for Hybrid)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    
    # DDL β values
    ax1 = axes[0]
    beta_means = [results[n]['DDL']['param_info'].get('beta_mean', 0) for n in sample_sizes]
    beta_stds = [results[n]['DDL']['param_info'].get('beta_std', 0) for n in sample_sizes]
    ax1.errorbar(sample_sizes, beta_means, yerr=beta_stds, 
                marker='s', color=COLORS['DDL'], capsize=4, linewidth=2, markersize=8)
    ax1.axhline(y=2.0, color='gray', linestyle='--', linewidth=1, alpha=0.7, label='β = 2 (exact reflection)')
    ax1.set_xlabel('Number of Training Samples')
    ax1.set_ylabel('Learned β Value')
    ax1.set_title('(a) DDL: β Convergence', fontweight='bold')
    ax1.set_ylim(0, 2.2)
    ax1.legend(loc='lower right')
    ax1.set_xscale('log')
    ax1.set_xticks(sample_sizes)
    ax1.set_xticklabels(sample_sizes)
    
    # Hybrid gate values
    ax2 = axes[1]
    gate_means = [results[n]['E∆-MHC-Geo']['param_info'].get('gate_mean', 1) for n in sample_sizes]
    gate_stds = [results[n]['E∆-MHC-Geo']['param_info'].get('gate_std', 0) for n in sample_sizes]
    ax2.errorbar(sample_sizes, gate_means, yerr=gate_stds,
                marker='D', color=COLORS['E∆-MHC-Geo'], capsize=4, linewidth=2, markersize=8)
    ax2.axhline(y=0.0, color='gray', linestyle='--', linewidth=1, alpha=0.7, label='γ = 0 (Householder)')
    ax2.set_xlabel('Number of Training Samples')
    ax2.set_ylabel('Learned Gate Value (γ)')
    ax2.set_title('(b) E∆-MHC-Geo: Gate Convergence', fontweight='bold')
    ax2.set_ylim(-0.05, 0.5)
    ax2.legend(loc='upper right')
    ax2.set_xscale('log')
    ax2.set_xticks(sample_sizes)
    ax2.set_xticklabels(sample_sizes)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/reflection_parameter_analysis.png', dpi=300,
                bbox_inches='tight', facecolor='white')
    flush_print(f"Saved: {output_dir}/reflection_parameter_analysis.png")
    plt.close()
    
    # Figure 3: Comprehensive Summary
    fig = plt.figure(figsize=(12, 8))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.25)
    
    # (a) Sample efficiency
    ax1 = fig.add_subplot(gs[0, 0])
    for name in model_names:
        accuracies = [results[n][name]['negation_accuracy'] for n in sample_sizes]
        ax1.plot(sample_sizes, accuracies, 
               marker=MARKERS[name], color=COLORS[name], 
               label=name, linewidth=2, markersize=7)
    ax1.axhline(y=0.95, color='gray', linestyle='--', linewidth=1, alpha=0.7)
    ax1.set_xlabel('Training Samples')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('(a) Sample Efficiency', fontweight='bold')
    ax1.set_ylim(-0.1, 1.05)
    ax1.legend(loc='lower right', fontsize=9)
    ax1.set_xscale('log')
    ax1.set_xticks(sample_sizes)
    ax1.set_xticklabels(sample_sizes)
    
    # (b) Final loss comparison (bar chart)
    ax2 = fig.add_subplot(gs[0, 1])
    final_losses = [results[200][name]['final_loss'] for name in model_names]
    x_pos = np.arange(len(model_names))
    colors = [COLORS[name] for name in model_names]
    bars = ax2.bar(x_pos, final_losses, color=colors, edgecolor='black', linewidth=0.8)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(['GPT', 'DDL', 'mHC', 'Ours'], fontweight='bold')
    ax2.set_ylabel('Final MSE Loss')
    ax2.set_title('(b) Final Performance (200 samples)', fontweight='bold')
    ax2.set_yscale('log')
    for bar, loss in zip(bars, final_losses):
        ax2.annotate(f'{loss:.1e}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
    
    # (c) β convergence
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.errorbar(sample_sizes, beta_means, yerr=beta_stds,
                marker='s', color=COLORS['DDL'], capsize=4, linewidth=2, markersize=7)
    ax3.axhline(y=2.0, color='gray', linestyle='--', linewidth=1, alpha=0.7)
    ax3.set_xlabel('Training Samples')
    ax3.set_ylabel('β Value')
    ax3.set_title('(c) DDL β → 2', fontweight='bold')
    ax3.set_ylim(0, 2.2)
    ax3.set_xscale('log')
    ax3.set_xticks(sample_sizes)
    ax3.set_xticklabels(sample_sizes)
    
    # (d) Gate convergence
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.errorbar(sample_sizes, gate_means, yerr=gate_stds,
                marker='D', color=COLORS['E∆-MHC-Geo'], capsize=4, linewidth=2, markersize=7)
    ax4.axhline(y=0.0, color='gray', linestyle='--', linewidth=1, alpha=0.7)
    ax4.set_xlabel('Training Samples')
    ax4.set_ylabel('Gate Value (γ)')
    ax4.set_title('(d) E∆-MHC-Geo γ → 0', fontweight='bold')
    ax4.set_ylim(-0.05, 0.5)
    ax4.set_xscale('log')
    ax4.set_xticks(sample_sizes)
    ax4.set_xticklabels(sample_sizes)
    
    fig.suptitle('Reflection Task: Direct Test of Geometric Operators', 
                fontsize=14, fontweight='bold', y=1.0)
    
    plt.savefig(f'{output_dir}/reflection_comprehensive.png', dpi=300,
                bbox_inches='tight', facecolor='white')
    flush_print(f"Saved: {output_dir}/reflection_comprehensive.png")
    plt.close()
    
    flush_print("\nAll reflection figures generated successfully!")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Reflection Task Training')
    parser.add_argument('--dim', type=int, default=64, help='Vector dimension')
    parser.add_argument('--max_iters', type=int, default=1000, help='Training iterations')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')
    parser.add_argument('--mode', type=str, default='sample_efficiency',
                       choices=['sample_efficiency', 'single'])
    parser.add_argument('--n_samples', type=int, default=100, help='Samples for single mode')
    parser.add_argument('--save_figures', action='store_true', help='Save publication figures')
    parser.add_argument('--output_dir', type=str, default='results', help='Output directory')
    args = parser.parse_args()
    
    flush_print("="*80)
    flush_print("REFLECTION EXPERIMENT: Direct Test of Geometric Operators")
    flush_print("Task: Learn y = -x (pure negation)")
    flush_print("="*80)
    
    if args.mode == 'sample_efficiency':
        results, min_samples = run_sample_efficiency_test(
            dim=args.dim,
            device=args.device,
            max_iters=args.max_iters,
            save_figures=args.save_figures,
            output_dir=args.output_dir,
        )
    else:
        train_x, train_y = generate_negation_data(args.n_samples, args.dim, args.device)
        val_x, val_y = generate_negation_data(200, args.dim, args.device)
        
        models = {
            'GPT': SimpleGPT(args.dim).to(args.device),
            'DDL': SimpleDDL(args.dim).to(args.device),
            'mHC': SimpleMHC(args.dim).to(args.device),
            'E∆-MHC-Geo': SimpleHybrid(args.dim).to(args.device),
        }
        
        for name, model in models.items():
            train_and_analyze(
                model=model,
                model_name=name,
                train_x=train_x,
                train_y=train_y,
                val_x=val_x,
                val_y=val_y,
                max_iters=args.max_iters,
            )
    
    flush_print("\n" + "="*80)
    flush_print("Reflection experiment complete!")
    flush_print("="*80)


if __name__ == '__main__':
    main()
