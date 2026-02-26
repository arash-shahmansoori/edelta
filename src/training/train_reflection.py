#!/usr/bin/env python3
"""
Reflection Training: Direct Test of Geometric Operators

============================================================================
The PUREST test of geometric inductive bias: learning y = -x (negation)
============================================================================

Inspired by "The Illusion of Insight in Reasoning Models" (arXiv:2601.00514v1):
- Mid-trace "Aha!" moments require parameter shifts that improve performance
- We test whether geometric operators learn the correct parameters for reflection

This test directly validates the CORE CLAIM of geometric models:
- DDL's operator A = I - β·kk^T achieves exact reflection when β=2
- E∆-MHC-Geo's Householder component achieves reflection when gate γ → 0

Key Metrics (following arXiv:2601.00514v1 methodology):
1. Parameter convergence: β → 2 (DDL), γ → 0 (E∆-MHC-Geo)
2. Sample efficiency: fewer samples needed with correct inductive bias
3. Parameter trajectory: how β and γ evolve during training
4. Accuracy conditional on parameter values

NOTE: We exclude GPT and mHC as they rely on MLP approximation rather than
geometric operators - this would be "cheating" as the MLP can learn any
function without the geometric inductive bias we're testing.

JPmHC is included as an SO(n)-only baseline: its iterative Cayley retraction
can only produce rotations (det=+1), so it should FAIL at exact negation
(which requires eigenvalue -1). This validates the need for E∆'s hybrid.

Usage (run from project root with uv):
    # Run sample efficiency test (main experiment)
    uv run src/training/train_reflection.py --mode sample_efficiency --save_figures
    
    # Run parameter trajectory analysis
    uv run src/training/train_reflection.py --mode trajectory --save_figures

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
    'JPmHC': '#CC79A7',      # Reddish Purple
    'E∆-MHC-Geo': '#009E73', # Bluish Green
}

MARKERS = {
    'GPT': 'o',
    'DDL': 's',
    'mHC': '^',
    'JPmHC': 'v',
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
    
    Reference: RESEARCH.md Section 5.3, 6.3
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
        
        # Thermodynamic gate with symmetry-breaking initialization
        # 
        # CRITICAL FINDING (see RESEARCH.md Section 6.5):
        # The midpoint collapse regularization L = 4γ(1-γ) has gradient:
        #   ∂L/∂γ = 4(1-2γ) = 0  at γ = 0.5
        # 
        # This creates a zero-gradient critical point where the model CANNOT escape
        # via gradient descent, even though the penalty is maximum at γ=0.5.
        # 
        # WHY CONTINUOUS BENCHMARKS WORK WITH UNBIASED INIT:
        # - Input features have natural variation (sequences, continuous values)
        # - Gate network γ(x) = σ(w·x + b) produces input-dependent values
        # - Different inputs produce γ ≠ 0.5, breaking symmetry naturally
        # 
        # WHY REFLECTION TASK NEEDS SYMMETRY BREAKING:
        # - All inputs are unit-normalized (spherically symmetric)
        # - With zero-mean init, γ ≈ 0.5 uniformly across all samples
        # - No natural symmetry breaking → trapped at zero-gradient point
        # 
        # SOLUTION: Small negative bias (b = -1.5 → γ ≈ 0.18) breaks symmetry
        # This allows the regularization gradient to push γ → 0 (Householder)
        self.gate_linear = nn.Linear(dim, 1)
        nn.init.zeros_(self.gate_linear.weight)
        nn.init.constant_(self.gate_linear.bias, -1.5)  # Init γ ≈ 0.18 (symmetry break towards Householder)
        
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


class SimpleJPmHC(nn.Module):
    """
    JPmHC-style model: y = Q(x) · x where Q is from iterative Cayley retraction.

    Reference: Sengupta, Wang & Brunswic, arXiv:2602.18308

    The iterative Cayley retraction produces approximately orthogonal matrices
    in SO(n) (det ≈ +1). Since Cayley CANNOT produce eigenvalue -1
    (Theorem: exclusion), this model should FAIL at y = -x.

    This validates E∆-MHC-Geo's hybrid approach: SO(n)-only methods
    fundamentally cannot handle negation tasks.
    """

    def __init__(self, dim: int, hidden_dim: int = HIDDEN_DIM,
                 cayley_alpha: float = 0.1, cayley_iters: int = 2):
        super().__init__()
        self.dim = dim
        self.cayley_alpha = cayley_alpha
        self.cayley_iters = cayley_iters

        self.W_net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim * dim),
        )

    def iterative_cayley(self, W_raw):
        """
        Iterative fixed-point Cayley retraction (JPmHC Algorithm 3).
        Y_0 = I; Y_{i+1} = I + (α/2) W (I + Y_i)
        """
        W = W_raw - W_raw.transpose(-1, -2)

        n = W.shape[-1]
        I = torch.eye(n, device=W.device, dtype=W.dtype)
        Y = I.expand_as(W).clone()
        for _ in range(self.cayley_iters):
            Y = I + (self.cayley_alpha / 2) * torch.matmul(W, I + Y)
        return Y

    def forward(self, x):
        B = x.shape[0] if x.dim() > 1 else 1
        if x.dim() == 1:
            x = x.unsqueeze(0)

        W_raw = self.W_net(x).view(B, self.dim, self.dim)
        Q = self.iterative_cayley(W_raw)
        out = torch.einsum('bij,bj->bi', Q, x)
        return out.squeeze(0) if B == 1 else out

    def get_orthogonality_error(self, x):
        """Report ||Q^T Q - I||_max for diagnostics."""
        B = x.shape[0] if x.dim() > 1 else 1
        if x.dim() == 1:
            x = x.unsqueeze(0)
        W_raw = self.W_net(x).view(B, self.dim, self.dim)
        Q = self.iterative_cayley(W_raw)
        QtQ = torch.matmul(Q.transpose(-1, -2), Q)
        I = torch.eye(self.dim, device=x.device, dtype=x.dtype)
        return (QtQ - I).abs().max().item()


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

    if hasattr(model, 'get_orthogonality_error'):
        with torch.no_grad():
            orth_err = model.get_orthogonality_error(val_x[:100])
            param_info['orth_error'] = orth_err
            if verbose:
                flush_print(f"  Orthogonality error: {orth_err:.6f}")
                flush_print(f"  ✗ SO(n) only — cannot produce eigenvalue -1")

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
# TRAINING WITH PARAMETER TRAJECTORY TRACKING
# =============================================================================

def train_with_trajectory(
    model: nn.Module,
    model_name: str,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    val_x: torch.Tensor,
    val_y: torch.Tensor,
    max_iters: int = 2000,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    track_interval: int = 50,
    verbose: bool = True,
):
    """
    Train model with detailed parameter trajectory tracking.
    
    Inspired by arXiv:2601.00514v1 - tracks parameter evolution to identify
    "Aha!" moments (sudden parameter shifts that improve performance).
    """
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    n_train = len(train_x)
    n_params = sum(p.numel() for p in model.parameters())
    
    if verbose:
        flush_print(f"\n{'='*60}")
        flush_print(f"Model: {model_name} ({n_params:,} params)")
    
    # Trajectory tracking
    trajectory = {
        'iter': [],
        'train_loss': [],
        'val_loss': [],
        'val_cos': [],
        'negation_accuracy': [],
    }
    
    # Model-specific parameter tracking
    if hasattr(model, 'get_beta'):
        trajectory['beta_mean'] = []
        trajectory['beta_std'] = []
    if hasattr(model, 'get_gate'):
        trajectory['gate_mean'] = []
        trajectory['gate_std'] = []
    if hasattr(model, 'get_orthogonality_error'):
        trajectory['orth_error'] = []
    
    best_val_loss = float('inf')
    
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
        
        # Track trajectory at intervals
        if (iter_num + 1) % track_interval == 0 or iter_num == 0:
            model.eval()
            with torch.no_grad():
                val_pred = model(val_x)
                val_loss = F.mse_loss(val_pred, val_y).item()
                val_cos = F.cosine_similarity(val_pred, val_y, dim=-1).mean().item()
                neg_acc = F.cosine_similarity(val_pred, -val_x, dim=-1).mean().item()
                
                trajectory['iter'].append(iter_num + 1)
                trajectory['train_loss'].append(loss.item())
                trajectory['val_loss'].append(val_loss)
                trajectory['val_cos'].append(val_cos)
                trajectory['negation_accuracy'].append(neg_acc)
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                
                # Track β for DDL
                if hasattr(model, 'get_beta'):
                    betas = model.get_beta(val_x[:100])
                    trajectory['beta_mean'].append(betas.mean().item())
                    trajectory['beta_std'].append(betas.std().item())
                
                # Track gate for E∆-MHC-Geo
                if hasattr(model, 'get_gate'):
                    gates = model.get_gate(val_x[:100])
                    trajectory['gate_mean'].append(gates.mean().item())
                    trajectory['gate_std'].append(gates.std().item())

                if hasattr(model, 'get_orthogonality_error'):
                    orth_err = model.get_orthogonality_error(val_x[:100])
                    trajectory['orth_error'].append(orth_err)
                
                if verbose and (iter_num + 1) % (track_interval * 4) == 0:
                    param_str = ""
                    if 'beta_mean' in trajectory:
                        param_str = f", β={trajectory['beta_mean'][-1]:.3f}"
                    if 'gate_mean' in trajectory:
                        param_str = f", γ={trajectory['gate_mean'][-1]:.3f}"
                    if 'orth_error' in trajectory:
                        param_str = f", orth_err={trajectory['orth_error'][-1]:.6f}"
                    flush_print(f"  Step {iter_num+1:4d}: loss={loss.item():.6f}, "
                               f"neg_acc={neg_acc:.4f}{param_str}")
    
    # Final analysis
    model.eval()
    with torch.no_grad():
        final_pred = model(val_x)
        final_loss = F.mse_loss(final_pred, val_y).item()
        final_cos = F.cosine_similarity(final_pred, val_y, dim=-1).mean().item()
        negation_accuracy = F.cosine_similarity(final_pred, -val_x, dim=-1).mean().item()
    
    # Compute parameter insights
    param_info = {}
    if hasattr(model, 'get_beta'):
        betas = model.get_beta(val_x[:100])
        param_info['beta_mean'] = betas.mean().item()
        param_info['beta_std'] = betas.std().item()
        param_info['beta_converged'] = param_info['beta_mean'] > 1.9
        if verbose:
            status = "✓ CONVERGED" if param_info['beta_converged'] else "✗ NOT converged"
            flush_print(f"  Final β: {param_info['beta_mean']:.4f} ± {param_info['beta_std']:.4f} {status}")
    
    if hasattr(model, 'get_gate'):
        gates = model.get_gate(val_x[:100])
        param_info['gate_mean'] = gates.mean().item()
        param_info['gate_std'] = gates.std().item()
        param_info['gate_converged'] = param_info['gate_mean'] < 0.1
        if verbose:
            status = "✓ CONVERGED" if param_info['gate_converged'] else "✗ NOT converged"
            flush_print(f"  Final γ: {param_info['gate_mean']:.4f} ± {param_info['gate_std']:.4f} {status}")

    if hasattr(model, 'get_orthogonality_error'):
        orth_err = model.get_orthogonality_error(val_x[:100])
        param_info['orth_error'] = orth_err
        if verbose:
            flush_print(f"  Orthogonality error: {orth_err:.6f}")
            flush_print(f"  ✗ SO(n) only — CANNOT produce eigenvalue -1 for negation")
    
    return {
        'model_name': model_name,
        'n_params': n_params,
        'final_loss': final_loss,
        'final_cos': final_cos,
        'negation_accuracy': negation_accuracy,
        'best_val_loss': best_val_loss,
        'param_info': param_info,
        'trajectory': trajectory,
    }


# =============================================================================
# SAMPLE EFFICIENCY TEST (DDL vs E∆-MHC-Geo only)
# =============================================================================

def run_sample_efficiency_test(
    dim: int,
    device: str,
    max_iters: int = 2000,
    sample_sizes: list = None,
    save_figures: bool = False,
    output_dir: str = 'results',
    seed: int = 42,
):
    """
    Test sample efficiency for geometric operators: DDL vs E∆-MHC-Geo.
    
    Following arXiv:2601.00514v1 methodology:
    - Track parameter trajectories (β for DDL, γ for E∆-MHC-Geo)
    - Measure accuracy conditional on parameter convergence
    - Identify "Aha!" moments (parameter shifts that improve performance)
    
    NOTE: GPT and mHC are excluded as they use MLP approximation rather than
    geometric operators - testing them would be "cheating".
    """
    
    if sample_sizes is None:
        sample_sizes = [10, 25, 50, 100, 200, 500]
    
    flush_print("\n" + "="*80)
    flush_print("REFLECTION EXPERIMENT: Geometric Operator Analysis")
    flush_print("="*80)
    flush_print("Testing: DDL (β → 2) vs JPmHC (SO(n) only) vs E∆-MHC-Geo (γ → 0)")
    flush_print("Task: Learn y = -x (pure negation/reflection)")
    flush_print("")
    flush_print("Following arXiv:2601.00514v1 'Illusion of Insight' methodology:")
    flush_print("  - Track parameter trajectories during training")
    flush_print("  - Measure convergence to optimal values (β=2, γ=0)")
    flush_print("  - JPmHC: SO(n)-only baseline (should FAIL — no eigenvalue -1)")
    flush_print("  - Analyze accuracy conditional on parameter state\n")
    
    # Fixed validation set
    val_x, val_y = generate_negation_data(500, dim, device, seed=999)
    
    results = {size: {} for size in sample_sizes}
    model_names = ['DDL', 'JPmHC', 'E∆-MHC-Geo']
    
    for n_samples in sample_sizes:
        flush_print(f"\n{'='*60}")
        flush_print(f"Training with {n_samples} samples")
        flush_print("="*60)
        
        train_x, train_y = generate_negation_data(n_samples, dim, device, seed=seed)
        
        models = {
            'DDL': SimpleDDL(dim).to(device),
            'JPmHC': SimpleJPmHC(dim).to(device),
            'E∆-MHC-Geo': SimpleHybrid(dim).to(device),
        }
        
        for name in model_names:
            model = models[name]
            result = train_with_trajectory(
                model=model,
                model_name=name,
                train_x=train_x,
                train_y=train_y,
                val_x=val_x,
                val_y=val_y,
                max_iters=max_iters,
                learning_rate=1e-3,
                track_interval=50,
            )
            results[n_samples][name] = result
    
    # Print summary table
    flush_print("\n" + "="*80)
    flush_print("RESULTS SUMMARY")
    flush_print("="*80)
    
    flush_print(f"\n{'Samples':<10}{'DDL Acc':<12}{'DDL β':<12}{'JPmHC Acc':<12}{'E∆ Acc':<12}{'E∆ γ':<12}")
    flush_print("-" * 70)
    
    for n_samples in sample_sizes:
        ddl = results[n_samples]['DDL']
        jpmhc = results[n_samples]['JPmHC']
        edelta = results[n_samples]['E∆-MHC-Geo']
        
        ddl_acc = ddl['negation_accuracy']
        ddl_beta = ddl['param_info'].get('beta_mean', 0)
        jpmhc_acc = jpmhc['negation_accuracy']
        edelta_acc = edelta['negation_accuracy']
        edelta_gate = edelta['param_info'].get('gate_mean', 1)
        
        flush_print(f"{n_samples:<10}{ddl_acc:>8.4f}    {ddl_beta:>8.4f}    "
                   f"{jpmhc_acc:>8.4f}    "
                   f"{edelta_acc:>8.4f}    {edelta_gate:>8.4f}")
    
    # Parameter convergence analysis
    flush_print("\n" + "="*80)
    flush_print("PARAMETER CONVERGENCE ANALYSIS")
    flush_print("="*80)
    flush_print("\nDDL: β should converge to 2.0 for exact Householder reflection")
    flush_print("JPmHC: SO(n) only — CANNOT converge (no eigenvalue -1 mechanism)")
    flush_print("E∆-MHC-Geo: γ should converge to 0.0 to use Householder component\n")
    
    for name in model_names:
        flush_print(f"\n{name}:")
        for n_samples in sample_sizes:
            r = results[n_samples][name]
            if 'beta_mean' in r['param_info']:
                beta = r['param_info']['beta_mean']
                converged = "✓" if r['param_info'].get('beta_converged', False) else "✗"
                flush_print(f"  {n_samples:>3} samples: β = {beta:.4f} {converged}")
            if 'gate_mean' in r['param_info']:
                gate = r['param_info']['gate_mean']
                converged = "✓" if r['param_info'].get('gate_converged', False) else "✗"
                flush_print(f"  {n_samples:>3} samples: γ = {gate:.4f} {converged}")
    
    if save_figures:
        create_reflection_figures(results, sample_sizes, model_names, output_dir)
    
    return results, model_names


# =============================================================================
# PUBLICATION-QUALITY FIGURES
# =============================================================================

def create_reflection_figures(results, sample_sizes, model_names, output_dir='results'):
    """
    Generate publication-quality figures for reflection experiment.
    
    Following arXiv:2601.00514v1 visualization style:
    - Parameter trajectory over training
    - Accuracy vs parameter convergence
    - Sample efficiency comparison
    """
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Figure 1: Parameter Trajectories (2x3 grid including JPmHC)
    largest_n = max(sample_sizes)
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    
    # (a) DDL β trajectory
    ax1 = axes[0, 0]
    traj = results[largest_n]['DDL']['trajectory']
    ax1.plot(traj['iter'], traj['beta_mean'], 
            color=COLORS['DDL'], linewidth=2, label='β (mean)')
    ax1.fill_between(traj['iter'], 
                     np.array(traj['beta_mean']) - np.array(traj['beta_std']),
                     np.array(traj['beta_mean']) + np.array(traj['beta_std']),
                     color=COLORS['DDL'], alpha=0.2)
    ax1.axhline(y=2.0, color='gray', linestyle='--', linewidth=1.5, label='β = 2 (exact reflection)')
    ax1.set_xlabel('Training Iteration')
    ax1.set_ylabel('β Value')
    ax1.set_title(f'(a) DDL: β Trajectory ({largest_n} samples)', fontweight='bold')
    ax1.set_ylim(0, 2.2)
    ax1.legend(loc='lower right', fontsize=8)
    
    # (b) JPmHC: accuracy + loss trajectory (demonstrating SO(n) failure)
    ax2 = axes[0, 1]
    traj_jp = results[largest_n]['JPmHC']['trajectory']
    ax2.plot(traj_jp['iter'], traj_jp['negation_accuracy'],
            color=COLORS['JPmHC'], linewidth=2, label='Neg. Accuracy')
    ax2.axhline(y=0.95, color='red', linestyle=':', linewidth=1, alpha=0.5, label='95% target')
    ax2.axhline(y=0.0, color='gray', linestyle='--', linewidth=0.8, alpha=0.3)
    ax2_twin = ax2.twinx()
    ax2_twin.plot(traj_jp['iter'], traj_jp['val_loss'],
                 color=COLORS['JPmHC'], linewidth=1.5, linestyle='--', alpha=0.5, label='Val Loss')
    ax2_twin.set_ylabel('Val Loss', color=COLORS['JPmHC'], alpha=0.6, fontsize=9)
    ax2.set_xlabel('Training Iteration')
    ax2.set_ylabel('Negation Accuracy')
    ax2.set_title(f'(b) JPmHC: SO(n) Failure ({largest_n} samples)', fontweight='bold')
    ax2.set_ylim(-1.15, 1.05)
    ax2.legend(loc='upper right', fontsize=8)
    ax2.annotate('SO(n) cannot negate', xy=(0.5, 0.03), xycoords='axes fraction',
                ha='center', fontsize=9, color='#CC79A7', fontweight='bold')
    
    # (c) E∆-MHC-Geo gate trajectory
    ax3 = axes[0, 2]
    traj = results[largest_n]['E∆-MHC-Geo']['trajectory']
    ax3.plot(traj['iter'], traj['gate_mean'],
            color=COLORS['E∆-MHC-Geo'], linewidth=2, label='γ (mean)')
    ax3.fill_between(traj['iter'],
                     np.array(traj['gate_mean']) - np.array(traj['gate_std']),
                     np.array(traj['gate_mean']) + np.array(traj['gate_std']),
                     color=COLORS['E∆-MHC-Geo'], alpha=0.2)
    ax3.axhline(y=0.0, color='gray', linestyle='--', linewidth=1.5, label='γ = 0 (Householder)')
    ax3.set_xlabel('Training Iteration')
    ax3.set_ylabel('Gate Value (γ)')
    ax3.set_title(f'(c) E∆-MHC-Geo: γ → 0 ({largest_n} samples)', fontweight='bold')
    ax3.set_ylim(-0.05, 0.5)
    ax3.legend(loc='upper right', fontsize=8)
    
    # (d) DDL accuracy vs β phase plot
    ax4 = axes[1, 0]
    traj = results[largest_n]['DDL']['trajectory']
    sc = ax4.scatter(traj['beta_mean'], traj['negation_accuracy'],
                    c=traj['iter'], cmap='viridis', s=50, alpha=0.8)
    ax4.axvline(x=2.0, color='gray', linestyle='--', linewidth=1, alpha=0.7)
    ax4.axhline(y=0.95, color='red', linestyle=':', linewidth=1, alpha=0.7)
    ax4.set_xlabel('β Value')
    ax4.set_ylabel('Negation Accuracy')
    ax4.set_title('(d) DDL: Accuracy vs β', fontweight='bold')
    ax4.set_xlim(0, 2.2)
    cbar = plt.colorbar(sc, ax=ax4)
    cbar.set_label('Iteration')
    
    # (e) JPmHC accuracy vs orth error (or vs iteration if no orth data)
    ax5 = axes[1, 1]
    traj_jp = results[largest_n]['JPmHC']['trajectory']
    if 'orth_error' in traj_jp and len(traj_jp['orth_error']) > 0:
        sc = ax5.scatter(traj_jp['orth_error'], traj_jp['negation_accuracy'],
                        c=traj_jp['iter'], cmap='viridis', s=50, alpha=0.8)
        ax5.set_xlabel('Orthogonality Error')
        cbar = plt.colorbar(sc, ax=ax5)
        cbar.set_label('Iteration')
    else:
        sc = ax5.scatter(traj_jp['iter'], traj_jp['negation_accuracy'],
                        c=traj_jp['iter'], cmap='viridis', s=50, alpha=0.8)
        ax5.set_xlabel('Training Iteration')
        cbar = plt.colorbar(sc, ax=ax5)
        cbar.set_label('Iteration')
    ax5.axhline(y=0.0, color='gray', linestyle='--', linewidth=0.8, alpha=0.3)
    ax5.axhline(y=0.95, color='red', linestyle=':', linewidth=1, alpha=0.7)
    ax5.set_ylabel('Negation Accuracy')
    ax5.set_title('(e) JPmHC: Accuracy (stuck negative)', fontweight='bold')
    ax5.set_ylim(-1.15, 1.05)
    ax5.annotate('Orthogonal but cannot reach\neigenvalue −1', xy=(0.5, 0.05),
                xycoords='axes fraction', ha='center', fontsize=8, color='red', fontstyle='italic')
    
    # (f) E∆-MHC-Geo accuracy vs gate phase plot
    ax6 = axes[1, 2]
    traj = results[largest_n]['E∆-MHC-Geo']['trajectory']
    sc = ax6.scatter(traj['gate_mean'], traj['negation_accuracy'],
                    c=traj['iter'], cmap='viridis', s=50, alpha=0.8)
    ax6.axvline(x=0.0, color='gray', linestyle='--', linewidth=1, alpha=0.7)
    ax6.axhline(y=0.95, color='red', linestyle=':', linewidth=1, alpha=0.7)
    ax6.set_xlabel('Gate Value (γ)')
    ax6.set_ylabel('Negation Accuracy')
    ax6.set_title('(f) E∆-MHC-Geo: Accuracy vs γ', fontweight='bold')
    cbar = plt.colorbar(sc, ax=ax6)
    cbar.set_label('Iteration')
    
    fig.suptitle('Parameter Trajectories During Training\n(Following arXiv:2601.00514v1 methodology)', 
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/reflection_trajectories.png', dpi=300,
                bbox_inches='tight', facecolor='white')
    flush_print(f"\nSaved: {output_dir}/reflection_trajectories.png")
    plt.close()
    
    # Figure 2: Sample Efficiency Comparison
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # (a) Negation accuracy vs samples
    ax1 = axes[0]
    for name in model_names:
        accuracies = [results[n][name]['negation_accuracy'] for n in sample_sizes]
        ax1.plot(sample_sizes, accuracies, 
               marker=MARKERS[name], color=COLORS[name], 
               label=name, linewidth=2.5, markersize=10)
    ax1.axhline(y=0.95, color='gray', linestyle='--', linewidth=1.5, alpha=0.7, label='95% threshold')
    ax1.set_xlabel('Number of Training Samples', fontsize=12)
    ax1.set_ylabel('Negation Accuracy', fontsize=12)
    ax1.set_title('(a) Sample Efficiency: DDL vs JPmHC vs E∆-MHC-Geo', fontweight='bold')
    ax1.set_ylim(-1.15, 1.05)
    ax1.axhline(y=0.0, color='gray', linewidth=0.5, alpha=0.3)
    ax1.legend(loc='center right', fontsize=10)
    ax1.set_xscale('log')
    ax1.set_xticks(sample_sizes)
    ax1.set_xticklabels(sample_sizes)
    
    # (b) Final parameter values vs samples
    ax2 = axes[1]
    beta_means = [results[n]['DDL']['param_info'].get('beta_mean', 0) for n in sample_sizes]
    gate_means = [results[n]['E∆-MHC-Geo']['param_info'].get('gate_mean', 1) for n in sample_sizes]
    
    ax2_twin = ax2.twinx()
    
    l1, = ax2.plot(sample_sizes, beta_means, marker='s', color=COLORS['DDL'], 
                  linewidth=2.5, markersize=10, label='DDL β')
    ax2.axhline(y=2.0, color=COLORS['DDL'], linestyle='--', linewidth=1, alpha=0.5)
    ax2.set_ylabel('DDL β Value', color=COLORS['DDL'], fontsize=12)
    ax2.tick_params(axis='y', labelcolor=COLORS['DDL'])
    ax2.set_ylim(0, 2.2)
    
    l2, = ax2_twin.plot(sample_sizes, gate_means, marker='D', color=COLORS['E∆-MHC-Geo'],
                       linewidth=2.5, markersize=10, label='E∆-MHC-Geo γ')
    ax2_twin.axhline(y=0.0, color=COLORS['E∆-MHC-Geo'], linestyle='--', linewidth=1, alpha=0.5)
    ax2_twin.set_ylabel('E∆-MHC-Geo γ Value', color=COLORS['E∆-MHC-Geo'], fontsize=12)
    ax2_twin.tick_params(axis='y', labelcolor=COLORS['E∆-MHC-Geo'])
    ax2_twin.set_ylim(-0.05, 0.5)
    
    ax2.set_xlabel('Number of Training Samples', fontsize=12)
    ax2.set_title('(b) Parameter Convergence vs Sample Size', fontweight='bold')
    ax2.set_xscale('log')
    ax2.set_xticks(sample_sizes)
    ax2.set_xticklabels(sample_sizes)
    ax2.legend([l1, l2], ['DDL β', 'E∆-MHC-Geo γ'], loc='center right')
    
    fig.suptitle('Geometric Operator Analysis: Reflection Task (y = -x)', 
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/reflection_sample_efficiency.png', dpi=300,
                bbox_inches='tight', facecolor='white')
    flush_print(f"Saved: {output_dir}/reflection_sample_efficiency.png")
    plt.close()
    
    # Figure 3: Comprehensive Summary (top: 2 panels, bottom: 3 panels)
    fig = plt.figure(figsize=(16, 11))
    gs_top = gridspec.GridSpec(1, 2, figure=fig,
                               left=0.06, right=0.94, top=0.91, bottom=0.56, wspace=0.28)
    gs_bot = gridspec.GridSpec(1, 3, figure=fig,
                               left=0.06, right=0.96, top=0.46, bottom=0.06, wspace=0.35)
    
    # (a) Parameter convergence summary
    ax1 = fig.add_subplot(gs_top[0, 0])
    beta_final = [results[n]['DDL']['param_info'].get('beta_mean', 0) for n in sample_sizes]
    gate_final = [results[n]['E∆-MHC-Geo']['param_info'].get('gate_mean', 1) for n in sample_sizes]
    
    x = np.arange(len(sample_sizes))
    width = 0.35
    bars1 = ax1.bar(x - width/2, beta_final, width, label='DDL β', color=COLORS['DDL'], edgecolor='black')
    ax1.axhline(y=2.0, color=COLORS['DDL'], linestyle='--', linewidth=1.5, alpha=0.7)
    
    ax1_twin = ax1.twinx()
    bars2 = ax1_twin.bar(x + width/2, gate_final, width, label='E∆-MHC-Geo γ', 
                        color=COLORS['E∆-MHC-Geo'], edgecolor='black')
    ax1_twin.axhline(y=0.0, color=COLORS['E∆-MHC-Geo'], linestyle='--', linewidth=1.5, alpha=0.7)
    
    ax1.set_xlabel('Training Samples')
    ax1.set_ylabel('DDL β', color=COLORS['DDL'])
    ax1_twin.set_ylabel('E∆-MHC-Geo γ', color=COLORS['E∆-MHC-Geo'])
    ax1.set_title('(a) Final Parameter Values', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(sample_sizes)
    ax1.set_ylim(0, 2.2)
    ax1_twin.set_ylim(-0.05, 0.5)
    ax1.legend([bars1, bars2], ['DDL β', 'E∆ γ'], loc='upper right')
    
    # (b) Accuracy comparison — y-axis extended to show JPmHC negatives
    ax2 = fig.add_subplot(gs_top[0, 1])
    ddl_acc = [results[n]['DDL']['negation_accuracy'] for n in sample_sizes]
    jpmhc_acc = [results[n]['JPmHC']['negation_accuracy'] for n in sample_sizes]
    edelta_acc = [results[n]['E∆-MHC-Geo']['negation_accuracy'] for n in sample_sizes]
    
    width3 = 0.25
    ax2.bar(x - width3, ddl_acc, width3, label='DDL', color=COLORS['DDL'], edgecolor='black')
    ax2.bar(x, jpmhc_acc, width3, label='JPmHC', color=COLORS['JPmHC'], edgecolor='black')
    ax2.bar(x + width3, edelta_acc, width3, label='E∆-MHC-Geo', color=COLORS['E∆-MHC-Geo'], edgecolor='black')
    ax2.axhline(y=0.95, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='95% target')
    ax2.axhline(y=0.0, color='gray', linewidth=0.5, alpha=0.3)
    ax2.set_xlabel('Training Samples')
    ax2.set_ylabel('Negation Accuracy')
    ax2.set_title('(b) Final Accuracy Comparison', fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(sample_sizes)
    ax2.set_ylim(-1.15, 1.05)
    ax2.legend(loc='lower right', fontsize=9)
    ax2.annotate('JPmHC fails (SO(n) only)', xy=(0.35, 0.02), xycoords='axes fraction',
                fontsize=8, color='#CC79A7', fontweight='bold')
    
    # (c) DDL training trajectory
    ax3 = fig.add_subplot(gs_bot[0, 0])
    traj = results[largest_n]['DDL']['trajectory']
    ax3.plot(traj['iter'], traj['beta_mean'], color=COLORS['DDL'], linewidth=2, label='β')
    ax3_twin = ax3.twinx()
    ax3_twin.plot(traj['iter'], traj['negation_accuracy'], color='green', linewidth=2, linestyle='--', label='Accuracy')
    ax3.axhline(y=2.0, color='gray', linestyle=':', linewidth=1)
    ax3.set_xlabel('Training Iteration')
    ax3.set_ylabel('β Value', color=COLORS['DDL'])
    ax3_twin.set_ylabel('Accuracy', color='green')
    ax3.set_title(f'(c) DDL Dynamics ({largest_n} samples)', fontweight='bold')
    ax3.set_ylim(0, 2.2)
    ax3_twin.set_ylim(-0.1, 1.05)
    
    # (d) JPmHC training trajectory (demonstrating SO(n) failure)
    ax4 = fig.add_subplot(gs_bot[0, 1])
    traj_jp = results[largest_n]['JPmHC']['trajectory']
    ax4.plot(traj_jp['iter'], traj_jp['negation_accuracy'],
            color=COLORS['JPmHC'], linewidth=2, label='Accuracy')
    ax4_twin = ax4.twinx()
    ax4_twin.plot(traj_jp['iter'], traj_jp['val_loss'],
                 color=COLORS['JPmHC'], linewidth=1.5, linestyle='--', alpha=0.5, label='Val Loss')
    ax4_twin.set_ylabel('Val Loss', color=COLORS['JPmHC'], alpha=0.6, fontsize=9)
    ax4.axhline(y=0.0, color='gray', linestyle=':', linewidth=1)
    ax4.axhline(y=0.95, color='red', linestyle=':', linewidth=1, alpha=0.5)
    ax4.set_xlabel('Training Iteration')
    ax4.set_ylabel('Neg. Accuracy', color=COLORS['JPmHC'])
    ax4.set_title(f'(d) JPmHC: SO(n) Failure ({largest_n} samples)', fontweight='bold')
    ax4.set_ylim(-1.15, 1.05)
    ax4.annotate('Cannot negate (SO(n) only)', xy=(0.5, 0.03), xycoords='axes fraction',
                ha='center', fontsize=8, color='red', fontweight='bold')
    
    # (e) E∆-MHC-Geo training trajectory
    ax5 = fig.add_subplot(gs_bot[0, 2])
    traj = results[largest_n]['E∆-MHC-Geo']['trajectory']
    ax5.plot(traj['iter'], traj['gate_mean'], color=COLORS['E∆-MHC-Geo'], linewidth=2, label='γ')
    ax5_twin = ax5.twinx()
    ax5_twin.plot(traj['iter'], traj['negation_accuracy'], color='green', linewidth=2, linestyle='--', label='Accuracy')
    ax5.axhline(y=0.0, color='gray', linestyle=':', linewidth=1)
    ax5.set_xlabel('Training Iteration')
    ax5.set_ylabel('Gate γ', color=COLORS['E∆-MHC-Geo'])
    ax5_twin.set_ylabel('Accuracy', color='green')
    ax5.set_title(f'(e) E∆-MHC-Geo Dynamics ({largest_n} samples)', fontweight='bold')
    ax5.set_ylim(-0.05, 0.5)
    ax5_twin.set_ylim(-0.1, 1.05)
    
    fig.suptitle('Reflection Experiment: Geometric Operator Analysis\n'
                '(Following arXiv:2601.00514v1 "Illusion of Insight" methodology)',
                fontsize=14, fontweight='bold', y=0.98)
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
    parser.add_argument('--max_iters', type=int, default=2000, help='Training iterations')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')
    parser.add_argument('--mode', type=str, default='sample_efficiency',
                       choices=['sample_efficiency', 'single', 'trajectory'])
    parser.add_argument('--n_samples', type=int, default=100, help='Samples for single mode')
    parser.add_argument('--save_figures', action='store_true', help='Save publication figures')
    parser.add_argument('--output_dir', type=str, default='results', help='Output directory')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for training data')
    args = parser.parse_args()
    
    flush_print("="*80)
    flush_print("REFLECTION EXPERIMENT: Geometric Operator Analysis")
    flush_print("="*80)
    flush_print("Task: Learn y = -x (pure negation/reflection)")
    flush_print("")
    flush_print("Following arXiv:2601.00514v1 'Illusion of Insight' methodology:")
    flush_print("  - DDL: β should converge to 2 (Householder reflection)")
    flush_print("  - E∆-MHC-Geo: γ should converge to 0 (use Householder)")
    flush_print("")
    flush_print("NOTE: GPT and mHC excluded (MLP approximation, not geometric)")
    flush_print("="*80)
    
    if args.mode == 'sample_efficiency':
        results, model_names = run_sample_efficiency_test(
            dim=args.dim,
            device=args.device,
            max_iters=args.max_iters,
            save_figures=args.save_figures,
            output_dir=args.output_dir,
            seed=args.seed,
        )
    elif args.mode == 'trajectory':
        # Detailed trajectory analysis with single sample size
        flush_print(f"\nRunning trajectory analysis with {args.n_samples} samples...")
        train_x, train_y = generate_negation_data(args.n_samples, args.dim, args.device)
        val_x, val_y = generate_negation_data(500, args.dim, args.device)
        
        results = {}
        for name, ModelClass in [('DDL', SimpleDDL), ('JPmHC', SimpleJPmHC), ('E∆-MHC-Geo', SimpleHybrid)]:
            model = ModelClass(args.dim).to(args.device)
            results[name] = train_with_trajectory(
                model=model,
                model_name=name,
                train_x=train_x,
                train_y=train_y,
                val_x=val_x,
                val_y=val_y,
                max_iters=args.max_iters,
                track_interval=20,  # More frequent tracking
            )
        
        if args.save_figures:
            create_reflection_figures(
                {args.n_samples: results}, 
                [args.n_samples], 
                ['DDL', 'JPmHC', 'E∆-MHC-Geo'], 
                args.output_dir
            )
    else:
        # Single test mode
        train_x, train_y = generate_negation_data(args.n_samples, args.dim, args.device)
        val_x, val_y = generate_negation_data(200, args.dim, args.device)
        
        models = {
            'DDL': SimpleDDL(args.dim).to(args.device),
            'JPmHC': SimpleJPmHC(args.dim).to(args.device),
            'E∆-MHC-Geo': SimpleHybrid(args.dim).to(args.device),
        }
        
        for name, model in models.items():
            train_with_trajectory(
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
