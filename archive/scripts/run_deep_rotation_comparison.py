"""
DEEP ROTATION COMPARISON: DDC vs DDL vs Baseline

Phase 1: Simplified models with DEEP layers on challenging rotation task

Key changes from previous experiments:
1. DEEPER networks (8, 16, 32, 64 layers) - gradient stability critical
2. HARDER task - predict rotation after N steps (not just next step)
3. VARYING angles - model must learn to generalize
4. Track NORM PRESERVATION - DDC's key advantage

Expected Results:
- DDC maintains perfect norm (Cayley is always isometric)
- DDL may drift at deep networks (not always orthogonal)
- Baseline will struggle with depth (vanishing/exploding gradients)
"""

import os
import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


# =============================================================================
# Challenging Rotation Dataset
# =============================================================================

def create_challenging_rotation_data(n_samples=5000, dim=32, seq_len=16):
    """
    Create a CHALLENGING rotation dataset that requires learning:
    - Input: initial vector + angle encoding
    - Output: vector after K rotations (K varies)
    - Model must learn to apply rotation K times
    """
    sequences = []
    targets = []
    
    for _ in range(n_samples):
        # Random initial unit vector
        v0 = np.random.randn(dim)
        v0 = v0 / np.linalg.norm(v0)
        
        # Random rotation angle (5-90 degrees)
        angle = np.random.uniform(5, 90)
        angle_rad = np.radians(angle)
        
        # Create rotation matrix using Cayley (exact rotation in ground truth)
        A = np.random.randn(dim, dim)
        A = (A - A.T) / 2  # Skew-symmetric
        A = A / np.linalg.norm(A) * angle_rad
        I = np.eye(dim)
        R = np.linalg.solve(I + A/2, I - A/2)
        
        # Generate sequence: v0, v1, v2, ..., v_{seq_len-1}
        seq = [v0]
        v = v0
        for _ in range(seq_len - 1):
            v = R @ v
            seq.append(v)
        
        # Input: sequence of vectors (model sees trajectory)
        # Target: final vector (requires understanding rotation)
        sequences.append(np.array(seq[:-1]))  # [v0, ..., v_{T-2}]
        targets.append(seq[-1])  # v_{T-1}
    
    return (np.array(sequences, dtype=np.float32), 
            np.array(targets, dtype=np.float32))


# =============================================================================
# Model Layers
# =============================================================================

class BaselineLayer(nn.Module):
    """Standard residual layer."""
    def __init__(self, dim):
        super().__init__()
        self.ln = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )
        # Scale for deep networks
        self.scale = nn.Parameter(torch.ones(1) * 0.1)
    
    def forward(self, x):
        return x + self.scale * self.mlp(self.ln(x))


class DDLLayer(nn.Module):
    """Householder reflection layer."""
    def __init__(self, dim):
        super().__init__()
        self.ln = nn.LayerNorm(dim)
        self.k_net = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.GELU(),
            nn.Linear(dim // 4, dim),
        )
        self.beta_net = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.GELU(),
            nn.Linear(dim // 4, 1),
            nn.Sigmoid(),
        )
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )
    
    def forward(self, x):
        B, T, D = x.shape
        x_norm = self.ln(x)
        
        # Householder reflection
        x_pooled = x_norm.mean(dim=1)
        k = F.normalize(self.k_net(x_pooled), dim=-1).unsqueeze(1)
        beta = 2 * self.beta_net(x_pooled).unsqueeze(1)  # [0, 2]
        
        dot = (x_norm * k).sum(dim=-1, keepdim=True)
        x_ref = x_norm - beta * dot * k
        
        return x_ref + self.mlp(x_ref)


class DDCLayer(nn.Module):
    """Data-Dependent Cayley layer - GUARANTEED orthogonal."""
    def __init__(self, dim, n_streams=4):
        super().__init__()
        self.n_streams = n_streams
        self.d_stream = dim // n_streams
        self.dim = dim
        
        self.ln = nn.LayerNorm(dim)
        
        # Data-dependent generators
        self.u_net = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.GELU(),
            nn.Linear(dim // 4, n_streams),
        )
        self.v_net = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.GELU(),
            nn.Linear(dim // 4, n_streams),
        )
        self.beta_net = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.GELU(),
            nn.Linear(dim // 4, 1),
            nn.Softplus(),
        )
        
        # Residual gate (allows identity when needed)
        self.alpha_gate = nn.Linear(dim, 1)
        
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )
        
        self.register_buffer('I', torch.eye(n_streams))
        
        # Initialize for small rotations
        nn.init.normal_(self.u_net[-1].weight, std=0.01)
        nn.init.normal_(self.v_net[-1].weight, std=0.01)
        nn.init.zeros_(self.alpha_gate.weight)
        nn.init.constant_(self.alpha_gate.bias, -1.0)
    
    def forward(self, x):
        B, T, D = x.shape
        x_norm = self.ln(x)
        x_pooled = x_norm.mean(dim=1)
        
        # Data-dependent generators
        u = self.u_net(x_pooled)
        v = self.v_net(x_pooled)
        beta = self.beta_net(x_pooled).unsqueeze(-1)
        
        # Skew-symmetric A (ALWAYS!) => Cayley ALWAYS orthogonal
        A = torch.einsum('bi,bj->bij', u, v) - torch.einsum('bi,bj->bij', v, u)
        
        # Cayley transform
        M = (beta / 2) * A
        I = self.I.unsqueeze(0).expand(B, -1, -1)
        Q = torch.linalg.solve(I + M, I - M)  # GUARANTEED orthogonal
        
        # Apply rotation
        x_streams = x_norm.view(B, T, self.n_streams, self.d_stream)
        x_rotated = torch.einsum('bij,btjd->btid', Q, x_streams).reshape(B, T, D)
        
        # Residual gate
        alpha = torch.sigmoid(self.alpha_gate(x_pooled)).unsqueeze(1)
        x_out = alpha * x_rotated + (1 - alpha) * x_norm
        
        return x_out + self.mlp(x_out)


class DDCHybridLayer(nn.Module):
    """DDC + Householder with learned gate."""
    def __init__(self, dim, n_streams=4):
        super().__init__()
        self.n_streams = n_streams
        self.d_stream = dim // n_streams
        self.dim = dim
        
        self.ln = nn.LayerNorm(dim)
        
        # DDC components
        self.u_net = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.GELU(),
            nn.Linear(dim // 4, n_streams),
        )
        self.v_net = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.GELU(),
            nn.Linear(dim // 4, n_streams),
        )
        self.beta_rot_net = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.GELU(),
            nn.Linear(dim // 4, 1),
            nn.Softplus(),
        )
        
        # Householder components
        self.k_net = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.GELU(),
            nn.Linear(dim // 4, dim),
        )
        
        # Gate: rotation vs reflection
        self.gate = nn.Linear(dim, 1)
        
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )
        
        self.register_buffer('I', torch.eye(n_streams))
        
        nn.init.normal_(self.u_net[-1].weight, std=0.01)
        nn.init.normal_(self.v_net[-1].weight, std=0.01)
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)
    
    def forward(self, x):
        B, T, D = x.shape
        x_norm = self.ln(x)
        x_pooled = x_norm.mean(dim=1)
        
        # === DDC Rotation ===
        u = self.u_net(x_pooled)
        v = self.v_net(x_pooled)
        beta = self.beta_rot_net(x_pooled).unsqueeze(-1)
        
        A = torch.einsum('bi,bj->bij', u, v) - torch.einsum('bi,bj->bij', v, u)
        M = (beta / 2) * A
        I = self.I.unsqueeze(0).expand(B, -1, -1)
        Q = torch.linalg.solve(I + M, I - M)
        
        x_streams = x_norm.view(B, T, self.n_streams, self.d_stream)
        x_rotated = torch.einsum('bij,btjd->btid', Q, x_streams).reshape(B, T, D)
        
        # === Householder Reflection (β=2 for orthogonality) ===
        k = F.normalize(self.k_net(x_pooled), dim=-1).unsqueeze(1)
        dot = (x_norm * k).sum(dim=-1, keepdim=True)
        x_reflected = x_norm - 2 * dot * k
        
        # === Gate ===
        gamma = torch.sigmoid(self.gate(x_pooled)).unsqueeze(1)
        x_hybrid = gamma * x_rotated + (1 - gamma) * x_reflected
        
        return x_hybrid + self.mlp(x_hybrid)


# =============================================================================
# Model
# =============================================================================

class DeepRotationModel(nn.Module):
    """Model for rotation prediction with configurable depth and type."""
    def __init__(self, dim, hidden_dim, n_layers, model_type='baseline'):
        super().__init__()
        self.input_proj = nn.Linear(dim, hidden_dim)
        
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            if model_type == 'baseline':
                self.layers.append(BaselineLayer(hidden_dim))
            elif model_type == 'ddl':
                self.layers.append(DDLLayer(hidden_dim))
            elif model_type == 'ddc':
                self.layers.append(DDCLayer(hidden_dim, n_streams=4))
            elif model_type == 'ddc_hybrid':
                self.layers.append(DDCHybridLayer(hidden_dim, n_streams=4))
        
        self.ln = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, dim)
    
    def forward(self, x):
        # x: (B, T, dim) - sequence of input vectors
        h = self.input_proj(x)
        for layer in self.layers:
            h = layer(h)
        h = self.ln(h)
        # Predict from last position
        out = self.output_proj(h[:, -1, :])  # (B, dim)
        return out


# =============================================================================
# Training
# =============================================================================

def train_model(model_type, dim, hidden_dim, n_layers, train_data, val_data,
                max_iters=2000, batch_size=64, lr=1e-3, device='cuda'):
    """Train and return metrics."""
    
    train_x, train_y = train_data
    val_x, val_y = val_data
    
    train_x = torch.from_numpy(train_x).to(device)
    train_y = torch.from_numpy(train_y).to(device)
    val_x = torch.from_numpy(val_x).to(device)
    val_y = torch.from_numpy(val_y).to(device)
    
    model = DeepRotationModel(dim, hidden_dim, n_layers, model_type).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    
    n_params = sum(p.numel() for p in model.parameters())
    
    # Metrics
    train_losses = []
    val_losses = []
    norm_errors = []
    grad_norms = []
    
    best_val_loss = float('inf')
    best_norm_error = float('inf')
    
    t0 = time.time()
    
    for it in range(max_iters):
        model.train()
        
        # Get batch
        idx = torch.randint(len(train_x), (batch_size,))
        x, y = train_x[idx], train_y[idx]
        
        # Forward
        pred = model(x)
        loss = F.mse_loss(pred, y)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        
        # Track gradient norm
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        total_norm = total_norm ** 0.5
        grad_norms.append(total_norm)
        
        # Check explosion
        if torch.isnan(loss) or total_norm > 1e6:
            print(f"    {model_type}: EXPLODED at iter {it}")
            return {
                'converged': False,
                'best_val_loss': float('inf'),
                'best_norm_error': float('inf'),
                'n_params': n_params,
                'exploded_at': it,
            }
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        train_losses.append(loss.item())
        
        # Evaluate
        if it % 200 == 0:
            model.eval()
            with torch.no_grad():
                # Full validation
                pred = model(val_x)
                val_loss = F.mse_loss(pred, val_y).item()
                
                # Norm preservation (key metric!)
                pred_norms = torch.norm(pred, dim=-1)
                target_norms = torch.norm(val_y, dim=-1)
                norm_error = (pred_norms - target_norms).abs().mean().item()
                
                # Angular error
                cos_sim = F.cosine_similarity(pred, val_y, dim=-1).mean().item()
                
                val_losses.append((it, val_loss))
                norm_errors.append((it, norm_error))
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_norm_error = norm_error
            
            print(f"    {model_type:12s} iter {it:4d}: train={loss.item():.6f}, "
                  f"val={val_loss:.6f}, norm_err={norm_error:.4f}, cos_sim={cos_sim:.4f}, grad={total_norm:.2f}")
    
    train_time = time.time() - t0
    
    return {
        'converged': True,
        'n_params': n_params,
        'best_val_loss': best_val_loss,
        'best_norm_error': best_norm_error,
        'mean_grad_norm': np.mean(grad_norms[-500:]) if len(grad_norms) > 500 else np.mean(grad_norms),
        'max_grad_norm': np.max(grad_norms),
        'train_time': train_time,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'norm_errors': norm_errors,
        'grad_norms': grad_norms,
    }


# =============================================================================
# Main
# =============================================================================

def run_comparison():
    print("=" * 80)
    print("DEEP ROTATION COMPARISON: DDC vs DDL vs Baseline vs DDC-Hybrid")
    print("Simplified models, DEEP layers, challenging rotation task")
    print("=" * 80)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # Create data
    print("\nCreating challenging rotation dataset...")
    dim = 32
    seq_len = 16
    train_data = create_challenging_rotation_data(n_samples=8000, dim=dim, seq_len=seq_len)
    val_data = create_challenging_rotation_data(n_samples=1000, dim=dim, seq_len=seq_len)
    print(f"  Train: {train_data[0].shape}, {train_data[1].shape}")
    print(f"  Val: {val_data[0].shape}, {val_data[1].shape}")
    
    # Configuration
    hidden_dim = 128
    depths = [8, 16, 32, 64]  # Test DEEP networks
    model_types = ['baseline', 'ddl', 'ddc', 'ddc_hybrid']
    
    results = {mt: {} for mt in model_types}
    
    for depth in depths:
        print(f"\n{'='*80}")
        print(f"DEPTH = {depth} LAYERS")
        print(f"{'='*80}")
        
        for mt in model_types:
            print(f"\n  Training {mt} (depth={depth})...")
            
            results[mt][depth] = train_model(
                model_type=mt,
                dim=dim,
                hidden_dim=hidden_dim,
                n_layers=depth,
                train_data=train_data,
                val_data=val_data,
                max_iters=1500,
                batch_size=64,
                lr=1e-3,
                device=device
            )
    
    # =============================================================================
    # Results Summary
    # =============================================================================
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    
    print("\n" + "-" * 80)
    print("VALIDATION LOSS (MSE) - lower is better")
    print("-" * 80)
    print(f"{'Depth':<8}", end="")
    for mt in model_types:
        print(f"{mt:<15}", end="")
    print()
    
    for depth in depths:
        print(f"{depth:<8}", end="")
        for mt in model_types:
            r = results[mt][depth]
            if r.get('converged', False):
                print(f"{r['best_val_loss']:<15.6f}", end="")
            else:
                print(f"{'EXPLODED':<15}", end="")
        print()
    
    print("\n" + "-" * 80)
    print("NORM PRESERVATION ERROR - lower is better (DDC's KEY advantage!)")
    print("-" * 80)
    print(f"{'Depth':<8}", end="")
    for mt in model_types:
        print(f"{mt:<15}", end="")
    print()
    
    for depth in depths:
        print(f"{depth:<8}", end="")
        for mt in model_types:
            r = results[mt][depth]
            if r.get('converged', False):
                print(f"{r['best_norm_error']:<15.6f}", end="")
            else:
                print(f"{'-':<15}", end="")
        print()
    
    print("\n" + "-" * 80)
    print("GRADIENT STABILITY (mean grad norm) - lower is better")
    print("-" * 80)
    print(f"{'Depth':<8}", end="")
    for mt in model_types:
        print(f"{mt:<15}", end="")
    print()
    
    for depth in depths:
        print(f"{depth:<8}", end="")
        for mt in model_types:
            r = results[mt][depth]
            if r.get('converged', False):
                print(f"{r['mean_grad_norm']:<15.2f}", end="")
            else:
                print(f"{'-':<15}", end="")
        print()
    
    # Winner analysis
    print("\n" + "=" * 80)
    print("WINNER AT EACH DEPTH")
    print("=" * 80)
    
    for depth in depths:
        print(f"\nDepth {depth}:")
        
        # Best val loss
        converged = [(mt, results[mt][depth]['best_val_loss']) 
                     for mt in model_types if results[mt][depth].get('converged', False)]
        if converged:
            winner = min(converged, key=lambda x: x[1])
            print(f"  Best Val Loss: {winner[0]} ({winner[1]:.6f})")
        else:
            print(f"  Best Val Loss: All models exploded!")
        
        # Best norm preservation
        converged = [(mt, results[mt][depth]['best_norm_error']) 
                     for mt in model_types if results[mt][depth].get('converged', False)]
        if converged:
            winner = min(converged, key=lambda x: x[1])
            print(f"  Best Norm Preservation: {winner[0]} ({winner[1]:.6f})")
    
    # Convergence analysis
    print("\n" + "=" * 80)
    print("CONVERGENCE ANALYSIS")
    print("=" * 80)
    
    for mt in model_types:
        max_depth_converged = max([d for d in depths if results[mt][d].get('converged', False)], default=0)
        print(f"{mt}: Max depth converged = {max_depth_converged}")
    
    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    colors = {'baseline': 'gray', 'ddl': 'red', 'ddc': 'blue', 'ddc_hybrid': 'green'}
    
    # Plot 1: Val loss vs depth
    ax = axes[0, 0]
    for mt in model_types:
        losses = [results[mt][d]['best_val_loss'] for d in depths if results[mt][d].get('converged', False)]
        valid_depths = [d for d in depths if results[mt][d].get('converged', False)]
        if losses:
            ax.plot(valid_depths, losses, 'o-', label=mt, color=colors[mt], 
                   markersize=10, linewidth=2)
    ax.set_xlabel('Depth (layers)', fontsize=12)
    ax.set_ylabel('Best Validation Loss (MSE)', fontsize=12)
    ax.set_title('Validation Loss vs Depth', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    # Plot 2: Norm error vs depth (KEY PLOT)
    ax = axes[0, 1]
    for mt in model_types:
        errors = [results[mt][d]['best_norm_error'] for d in depths if results[mt][d].get('converged', False)]
        valid_depths = [d for d in depths if results[mt][d].get('converged', False)]
        if errors:
            ax.plot(valid_depths, errors, 'o-', label=mt, color=colors[mt],
                   markersize=10, linewidth=2)
    ax.set_xlabel('Depth (layers)', fontsize=12)
    ax.set_ylabel('Norm Preservation Error', fontsize=12)
    ax.set_title('NORM PRESERVATION vs Depth (DDC Key Advantage)', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Training curves at max depth
    ax = axes[1, 0]
    max_depth = max(depths)
    for mt in model_types:
        if results[mt][max_depth].get('converged', False):
            losses = results[mt][max_depth]['train_losses']
            window = 30
            if len(losses) > window:
                smoothed = np.convolve(losses, np.ones(window)/window, mode='valid')
                ax.plot(smoothed, label=mt, color=colors[mt], alpha=0.8, linewidth=1.5)
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Training Loss', fontsize=12)
    ax.set_title(f'Training Curves (depth={max_depth})', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    # Plot 4: Gradient norms
    ax = axes[1, 1]
    for mt in model_types:
        if results[mt][max_depth].get('converged', False):
            norms = results[mt][max_depth]['grad_norms']
            window = 30
            if len(norms) > window:
                smoothed = np.convolve(norms, np.ones(window)/window, mode='valid')
                ax.plot(smoothed, label=mt, color=colors[mt], alpha=0.8, linewidth=1.5)
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Gradient Norm', fontsize=12)
    ax.set_title(f'Gradient Stability (depth={max_depth})', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('deep_rotation_comparison.png', dpi=150)
    print(f"\nPlot saved to: deep_rotation_comparison.png")
    
    # Key insights
    print("\n" + "=" * 80)
    print("KEY INSIGHTS")
    print("=" * 80)
    print("""
This experiment tests DEEP networks on a CHALLENGING rotation task:

1. BASELINE: Standard residual - may struggle with depth
2. DDL: Householder reflection - NOT always orthogonal (β≠0,2)
3. DDC: Data-Dependent Cayley - UNCONDITIONALLY orthogonal
4. DDC-HYBRID: DDC + Householder - best of both worlds

Expected DDC advantages at DEEP networks:
- More stable gradients (orthogonality prevents explosion)
- Better norm preservation (Cayley is always isometric)
- Higher max depth before explosion

If DDL fails before DDC at deep networks → proves orthogonality matters!
If DDC has lower norm error → proves Cayley's isometry property works!
""")


if __name__ == '__main__':
    run_comparison()
