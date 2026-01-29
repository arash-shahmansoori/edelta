#!/usr/bin/env python3
"""
Direct Reflection Test: Testing Geometric Operators Without Wrappers

The previous tests used ContinuousWrapper which adds projection layers,
potentially bypassing the geometric benefits of DDL/Proposed.

This test directly tests the CORE CLAIM of geometric models:
- DDL's operator A = I - β·kk^T can perform exact reflections when β=2
- Proposed's Householder component can perform reflections when gate → 0

We test this by:
1. Creating a simple task where the model must learn to NEGATE inputs
2. Testing if models converge to β≈2 (DDL) or gate≈0 (Proposed)
3. Comparing sample efficiency (fewer samples needed for geometric models)

This is the PUREST test of the geometric inductive bias.
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None
os.environ['PYTHONUNBUFFERED'] = '1'

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def flush_print(*args, **kwargs):
    print(*args, **kwargs)
    sys.stdout.flush()


# =============================================================================
# SIMPLE REFLECTION TASK: Learn y = -x
# =============================================================================

def generate_negation_data(n_samples: int, dim: int, device: str = 'cuda'):
    """
    Simple task: Given x, output -x.
    
    This is the PUREST test of reflection capability.
    - Householder with β=2: H·x = x - 2x = -x (perfect reflection)
    - Standard networks: Must learn f(x) ≈ -2x, then add residual x + f(x) = -x
    """
    x = torch.randn(n_samples, dim, device=device)
    y = -x  # Target is negation
    return x, y


# =============================================================================
# MINIMAL MODELS (No wrapper overhead)
# =============================================================================

# Global hidden_dim for fair parameter comparison
# All models use this to ensure similar capacity
HIDDEN_DIM = 128  # Gives ~30-50k params per model


class SimpleGPT(nn.Module):
    """
    Minimal GPT-style model: x + MLP(x)
    
    This is the baseline - standard residual network.
    To achieve y = -x, must learn f(x) = -2x which requires
    approximating a linear function with the MLP.
    """
    
    def __init__(self, dim: int, hidden_dim: int = HIDDEN_DIM):
        super().__init__()
        # 2-layer MLP (simpler than 3-layer for fair comparison)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )
    
    def forward(self, x):
        # Standard residual: y = x + f(x)
        # To get y = -x, must learn f(x) = -2x
        return x + self.mlp(x)


class SimpleDDL(nn.Module):
    """
    DDL-style model following proposed_model_ddl.py:
    A(X) = I - β·kk^T where k is normalized
    
    Key: k and β computed from input (data-dependent)
    Reference: arXiv:2601.00417
    
    For y = -x task:
    - Need β → 2 (full reflection)
    - Need k → x/||x|| (direction to negate along)
    """
    
    def __init__(self, dim: int, hidden_dim: int = HIDDEN_DIM):
        super().__init__()
        # DDL operator parameters
        self.k_net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )
        self.beta_net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),  # β ∈ [0, 1], scaled to [0, 2]
        )
        self.beta_scale = 2.0
        
    def forward(self, x):
        # Compute direction k (normalized) - data-dependent
        k = self.k_net(x)
        k = F.normalize(k, dim=-1)  # Unit norm
        
        # Compute β ∈ [0, 2]
        beta = self.beta_scale * self.beta_net(x)
        
        # DDL operator: A·x = x - β·(k^T·x)·k
        # This is exact Householder reflection when β=2
        k_dot_x = (k * x).sum(dim=-1, keepdim=True)
        return x - beta * k_dot_x * k
    
    def get_beta(self, x):
        """Return current β value for analysis."""
        return self.beta_scale * self.beta_net(x)


class SimpleHybrid(nn.Module):
    """
    Hybrid model following RESEARCH_V3.md and proposed_model_hybrid.py:
    
    G_γ(X) = γ·Q(X)·X + (1-γ)·H₂(k(X))·X
    
    Where (from RESEARCH_V3.md Definition 5.2):
    - Q(X) ∈ SO(n): Data-Dependent Cayley rotation (det=+1)
    - H₂(k) = I - 2·kk^T: Householder reflection (det=-1, β=2 FIXED per Theorem 7)
    - γ(X) = σ(W_γ·X̄ + b_γ): Learned gate
    
    CRITICAL from RESEARCH_V3.md Section 6:
    - Midpoint Collapse Regularization: L_gate = λ · 4γ(1-γ)
    - Forces "jump, don't swim" - gate should be binary (0 or 1)
    - Prevents non-orthogonal interpolation at γ = 0.5
    
    Reference: RESEARCH_V3.md Section 5.3, 6.3, proposed_model_hybrid.py
    """
    
    def __init__(self, dim: int, hidden_dim: int = HIDDEN_DIM, gate_reg_weight: float = 0.5):
        super().__init__()
        self.gate_reg_weight = gate_reg_weight  # Increased from 0.1 for stronger push to binary
        
        # === CAYLEY ROTATION (RESEARCH_V3.md Definition 2.1-2.3) ===
        # u(x) and v(x) for skew-symmetric matrix A = uv^T - vu^T
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
        # β for Cayley rotation magnitude (Softplus ensures positive)
        self.cayley_beta_net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Softplus(),
        )
        
        # === HOUSEHOLDER REFLECTION (RESEARCH_V3.md Definition 5.1) ===
        # β = 2 is FIXED (not learnable!) per Theorem 7
        self.k_net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )
        self.register_buffer('householder_beta', torch.tensor(2.0))
        
        # === THERMODYNAMIC GATE (RESEARCH_V3.md Definition 5.2) ===
        # γ(X) = σ(W_γ·X̄ + b_γ) ∈ (0, 1)
        self.gate_linear = nn.Linear(dim, 1)
        
        # Initialize gate bias to -2.0 so initial γ ≈ 0.12 (prefer Householder)
        # This helps for tasks where reflection is needed (like negation y=-x)
        nn.init.constant_(self.gate_linear.bias, -2.0)
        
        # Storage for regularization loss
        self._gate_reg_loss = None
    
    def cayley_rotation(self, x, u, v, beta):
        """
        EXACT Cayley transform: Q = (I + (β/2)A)^{-1}(I - (β/2)A)
        where A = uv^T - vu^T is skew-symmetric (RESEARCH_V3.md Definition 2.2-2.3)
        
        Following proposed_model_hybrid.py exactly:
        1. Build skew-symmetric A = uv^T - vu^T
        2. Compute M = (β/2) * A  
        3. Solve Q = (I + M)^{-1}(I - M) using torch.linalg.solve
        4. Apply Q @ x
        """
        dim = x.shape[-1]
        
        # Build skew-symmetric matrix A = uv^T - vu^T (Definition 2.2)
        # u, v: (..., dim), A: (..., dim, dim)
        A = torch.einsum('...i,...j->...ij', u, v) - torch.einsum('...i,...j->...ij', v, u)
        
        # Scale: M = (β/2) * A
        M = (beta / 2).unsqueeze(-1) * A  # (..., dim, dim)
        
        # Identity matrix
        I = torch.eye(dim, device=x.device, dtype=x.dtype)
        
        # Cayley transform: Q = (I + M)^{-1}(I - M)
        # Using solve for numerical stability (RESEARCH_V3.md, proposed_model_hybrid.py)
        Q = torch.linalg.solve(I + M, I - M)  # (..., dim, dim)
        
        # Apply rotation: x' = Q @ x
        # x: (..., dim), Q: (..., dim, dim)
        x_rotated = torch.einsum('...ij,...j->...i', Q, x)
        
        return x_rotated
    
    def householder_reflection(self, x, k):
        """
        Householder: H·x = x - 2·(k^T·x)·k
        β = 2 is FIXED (Theorem 7: only β=2 gives both orthogonality AND negation)
        """
        k_dot_x = (k * x).sum(dim=-1, keepdim=True)
        return x - self.householder_beta * k_dot_x * k
    
    def forward(self, x):
        # Compute Cayley parameters (data-dependent)
        u = F.normalize(self.u_net(x), dim=-1)
        v = F.normalize(self.v_net(x), dim=-1)
        # Orthogonalize v w.r.t. u for cleaner rotation plane
        v = v - (u * v).sum(dim=-1, keepdim=True) * u
        v = F.normalize(v, dim=-1)
        cayley_beta = self.cayley_beta_net(x)
        
        # Compute Householder direction (data-dependent)
        k = F.normalize(self.k_net(x), dim=-1)
        
        # Compute gate: γ = σ(W_γ·x + b_γ)
        gamma = torch.sigmoid(self.gate_linear(x))
        
        # === MIDPOINT COLLAPSE REGULARIZATION (RESEARCH_V3.md Section 6.2-6.3) ===
        # L_gate = λ · 4γ(1-γ)
        # Maximum at γ=0.5, zero at γ∈{0,1}
        # Forces "jump, don't swim" strategy
        self._gate_reg_loss = self.gate_reg_weight * 4 * gamma * (1 - gamma)
        self._gate_reg_loss = self._gate_reg_loss.mean()
        
        # Apply both operators
        cayley_out = self.cayley_rotation(x, u, v, cayley_beta)
        householder_out = self.householder_reflection(x, k)
        
        # Blend: γ·Cayley + (1-γ)·Householder
        return gamma * cayley_out + (1 - gamma) * householder_out
    
    def get_gate(self, x):
        """Return gate value for analysis."""
        return torch.sigmoid(self.gate_linear(x))
    
    def get_gate_regularization_loss(self):
        """Return midpoint collapse regularization loss for training."""
        return self._gate_reg_loss if self._gate_reg_loss is not None else 0.0


class SimpleMHC(nn.Module):
    """
    Simplified mHC following proposed_model_mhc_real.py (DeepSeek mHC).
    
    Full mHC equation (Paper Eq. 3):
        x_{l+1} = H_res · x_l + H_post^T · F(H_pre · x_l)
    
    Where:
    - H_res: Doubly stochastic mixing (Sinkhorn-Knopp) - preserves energy
    - H_pre: Aggregation weights for layer function
    - H_post: Broadcast weights back to streams
    - F: Transformation function (MLP)
    
    For negation task:
    - H_res can only MIX streams (all positive entries)
    - The MLP F must learn the negation
    - But H_res mixing dilutes the effect
    
    Reference: arXiv:2512.24880
    """
    
    def __init__(self, dim: int, hidden_dim: int = HIDDEN_DIM, n_streams: int = 4, n_sinkhorn_iters: int = 10):
        super().__init__()
        self.dim = dim
        self.n_streams = n_streams
        self.d_stream = dim // n_streams
        self.n_sinkhorn_iters = n_sinkhorn_iters
        
        # === Data-dependent coefficient computation (Paper Eq. 7) ===
        # φ projections: compute H_pre, H_post, H_res from input
        self.phi_pre = nn.Linear(dim, n_streams, bias=False)
        self.phi_post = nn.Linear(dim, n_streams, bias=False)
        self.phi_res = nn.Linear(dim, n_streams * n_streams, bias=False)
        
        # Static biases
        self.b_pre = nn.Parameter(torch.zeros(n_streams))
        self.b_post = nn.Parameter(torch.zeros(n_streams))
        self.b_res = nn.Parameter(torch.eye(n_streams) * 2.0)  # Init near identity
        
        # Learnable gating factors (α), init small per paper
        self.alpha_pre = nn.Parameter(torch.tensor(0.01))
        self.alpha_post = nn.Parameter(torch.tensor(0.01))
        self.alpha_res = nn.Parameter(torch.tensor(0.01))
        
        # === Layer function F (MLP) ===
        # MLP operates per-stream (d_stream), scale hidden_dim to match DDL params (~25k)
        # MLP params = 2 * d_stream * mlp_hidden + 2 * mlp_hidden ≈ 23k
        # mlp_hidden ≈ 23k / (2 * d_stream + 2) = 23k / 34 ≈ 680
        mlp_hidden = 350  # Gives ~25k total params to match DDL
        self.mlp = nn.Sequential(
            nn.Linear(self.d_stream, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, self.d_stream),
        )
    
    def sinkhorn_knopp(self, M, n_iters=None):
        """Project onto doubly stochastic manifold (Paper Eq. 9)."""
        if n_iters is None:
            n_iters = self.n_sinkhorn_iters
        
        M = torch.exp(M)
        for _ in range(n_iters):
            M = M / (M.sum(dim=-1, keepdim=True) + 1e-8)
            M = M / (M.sum(dim=-2, keepdim=True) + 1e-8)
        return M
    
    def forward(self, x):
        B = x.shape[0] if x.dim() > 1 else 1
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        # Reshape to streams: (B, n_streams, d_stream)
        x_streams = x.view(B, self.n_streams, self.d_stream)
        
        # === Compute data-dependent mappings (Paper Eq. 7-8) ===
        H_tilde_pre = self.alpha_pre * self.phi_pre(x) + self.b_pre  # (B, n)
        H_tilde_post = self.alpha_post * self.phi_post(x) + self.b_post  # (B, n)
        H_tilde_res = self.alpha_res * self.phi_res(x).view(B, self.n_streams, self.n_streams) + self.b_res
        
        # Apply constraints (Paper Eq. 8)
        H_pre = torch.sigmoid(H_tilde_pre)  # (B, n)
        H_post = 2.0 * torch.sigmoid(H_tilde_post)  # (B, n)
        H_res = self.sinkhorn_knopp(H_tilde_res)  # (B, n, n) - doubly stochastic
        
        # === Full mHC equation (Paper Eq. 3) ===
        # x_{l+1} = H_res · x_l + H_post^T · F(H_pre · x_l)
        
        # 1. H_res · x_l (doubly stochastic mixing)
        x_mixed = torch.einsum('bij,bjd->bid', H_res, x_streams)  # (B, n, d)
        
        # 2. H_pre · x_l (weighted aggregation per stream)
        # H_pre: (B, n), x_streams: (B, n, d)
        x_agg = H_pre.unsqueeze(-1) * x_streams  # (B, n, d) - weighted
        
        # 3. F(H_pre · x_l) - apply MLP to each stream
        x_transformed = self.mlp(x_agg)  # (B, n, d)
        
        # 4. H_post^T · F(...) (broadcast back)
        x_broadcast = H_post.unsqueeze(-1) * x_transformed  # (B, n, d)
        
        # 5. Combine: residual + transformed
        x_out = x_mixed + x_broadcast  # (B, n, d)
        
        # Reshape back to (B, dim)
        out = x_out.view(B, self.dim)
        
        return out.squeeze(0) if B == 1 and x.dim() == 1 else out


# =============================================================================
# TRAINING
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
):
    """Train model and analyze learned parameters."""
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    n_train = len(train_x)
    
    n_params = sum(p.numel() for p in model.parameters())
    flush_print(f"\n{'='*60}")
    flush_print(f"Model: {model_name} ({n_params:,} params)")
    
    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': [], 'val_cos': []}
    
    for iter_num in range(max_iters):
        model.train()
        
        idx = torch.randint(0, n_train, (min(batch_size, n_train),))
        xb, yb = train_x[idx], train_y[idx]
        
        pred = model(xb)
        loss = F.mse_loss(pred, yb)
        
        # Add gate regularization for Hybrid model (RESEARCH_V3.md Section 6.3)
        if hasattr(model, 'get_gate_regularization_loss'):
            gate_reg = model.get_gate_regularization_loss()
            if gate_reg is not None and gate_reg > 0:
                loss = loss + gate_reg
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        history['train_loss'].append(loss.item())
        
        if (iter_num + 1) % eval_interval == 0 or iter_num == 0:
            model.eval()
            with torch.no_grad():
                val_pred = model(val_x)
                val_loss = F.mse_loss(val_pred, val_y)
                val_cos = F.cosine_similarity(val_pred, val_y, dim=-1).mean().item()
                
                history['val_loss'].append(val_loss.item())
                history['val_cos'].append(val_cos)
                
                if val_loss.item() < best_val_loss:
                    best_val_loss = val_loss.item()
                
                flush_print(f"  Step {iter_num+1:4d}: loss={loss.item():.6f}, "
                           f"val_loss={val_loss.item():.6f}, cos_sim={val_cos:.4f}")
    
    # Final analysis
    model.eval()
    with torch.no_grad():
        final_pred = model(val_x)
        final_loss = F.mse_loss(final_pred, val_y).item()
        final_cos = F.cosine_similarity(final_pred, val_y, dim=-1).mean().item()
        
        # Check if output ≈ -input
        negation_accuracy = F.cosine_similarity(final_pred, -val_x, dim=-1).mean().item()
    
    # Analyze learned parameters
    param_info = {}
    if hasattr(model, 'get_beta'):
        with torch.no_grad():
            betas = model.get_beta(val_x[:100])
            param_info['beta_mean'] = betas.mean().item()
            param_info['beta_std'] = betas.std().item()
            flush_print(f"  β analysis: mean={param_info['beta_mean']:.4f}, std={param_info['beta_std']:.4f}")
            if param_info['beta_mean'] > 1.8:
                flush_print(f"  ✓ β → 2 (learned to use full reflection!)")
    
    if hasattr(model, 'get_gate'):
        with torch.no_grad():
            gates = model.get_gate(val_x[:100])
            param_info['gate_mean'] = gates.mean().item()
            param_info['gate_std'] = gates.std().item()
            flush_print(f"  Gate analysis: mean={param_info['gate_mean']:.4f}, std={param_info['gate_std']:.4f}")
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

def run_sample_efficiency_test(dim: int, device: str, max_iters: int = 1000):
    """
    Test how many samples each model needs to learn negation.
    
    Geometric models should need FEWER samples because they have
    the right inductive bias built in.
    """
    
    flush_print("\n" + "="*80)
    flush_print("SAMPLE EFFICIENCY TEST")
    flush_print("="*80)
    flush_print("Testing: How many samples to learn y = -x?")
    flush_print("Geometric models should need fewer samples.\n")
    
    sample_sizes = [10, 25, 50, 100, 200]
    
    # Fixed validation set
    val_x, val_y = generate_negation_data(500, dim, device)
    
    results = {size: {} for size in sample_sizes}
    
    for n_samples in sample_sizes:
        flush_print(f"\n{'='*60}")
        flush_print(f"Training with {n_samples} samples")
        flush_print("="*60)
        
        train_x, train_y = generate_negation_data(n_samples, dim, device)
        
        models = {
            'GPT (x + MLP)': SimpleGPT(dim).to(device),
            'DDL (I - βkk^T)': SimpleDDL(dim).to(device),
            'Hybrid (Cayley+House)': SimpleHybrid(dim).to(device),
            'mHC (Sinkhorn)': SimpleMHC(dim).to(device),
        }
        
        for name, model in models.items():
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
    
    # Summary table
    flush_print("\n" + "="*80)
    flush_print("SAMPLE EFFICIENCY RESULTS")
    flush_print("="*80)
    
    flush_print(f"\n{'Samples':<10}", end='')
    model_names = list(results[sample_sizes[0]].keys())
    for name in model_names:
        flush_print(f"{name[:15]:<18}", end='')
    flush_print()
    
    flush_print("-" * (10 + 18 * len(model_names)))
    
    for n_samples in sample_sizes:
        flush_print(f"{n_samples:<10}", end='')
        for name in model_names:
            cos = results[n_samples][name]['negation_accuracy']
            flush_print(f"{cos:>8.4f}          ", end='')
        flush_print()
    
    # Find minimum samples for each model to reach 0.95 accuracy
    flush_print("\n" + "="*80)
    flush_print("SAMPLES NEEDED FOR 95% NEGATION ACCURACY")
    flush_print("="*80)
    
    for name in model_names:
        for n_samples in sample_sizes:
            if results[n_samples][name]['negation_accuracy'] >= 0.95:
                flush_print(f"{name:<25}: {n_samples} samples")
                break
        else:
            flush_print(f"{name:<25}: >200 samples needed")
    
    return results


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dim', type=int, default=64)
    parser.add_argument('--max_iters', type=int, default=1000)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--mode', type=str, default='sample_efficiency',
                       choices=['sample_efficiency', 'single'])
    parser.add_argument('--n_samples', type=int, default=100)
    args = parser.parse_args()
    
    flush_print("="*80)
    flush_print("DIRECT REFLECTION TEST")
    flush_print("Testing the CORE geometric capability: learning y = -x")
    flush_print("="*80)
    
    if args.mode == 'sample_efficiency':
        results = run_sample_efficiency_test(args.dim, args.device, args.max_iters)
    else:
        # Single test
        train_x, train_y = generate_negation_data(args.n_samples, args.dim, args.device)
        val_x, val_y = generate_negation_data(200, args.dim, args.device)
        
        models = {
            'GPT': SimpleGPT(args.dim).to(args.device),
            'DDL': SimpleDDL(args.dim).to(args.device),
            'Hybrid': SimpleHybrid(args.dim).to(args.device),
            'mHC': SimpleMHC(args.dim).to(args.device),
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
    flush_print("Test complete!")
    flush_print("="*80)


if __name__ == '__main__':
    main()
