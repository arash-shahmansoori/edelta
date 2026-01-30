"""
COMPREHENSIVE CONTINUOUS ROTATION COMPARISON

Compares:
1. Baseline - Standard Transformer residual
2. DDL - Deep Delta Learning (Householder reflection)
3. DDC - Data-Dependent Cayley (unconditional orthogonality)
4. DDC-Hybrid - DDC + Householder with learned gate

Metrics:
- Validation Loss (MSE)
- Norm Preservation Error (critical for rotation tasks!)
- Gradient Stability
- Training Stability at Different Depths

This task directly tests geometric rotation capabilities:
- DDL's linear approximation may cause norm drift
- DDC's Cayley transform maintains ||Qx|| = ||x|| exactly
"""

import os
import sys
import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# First, prepare the dataset if not exists
DATA_DIR = 'data/continuous_rotation'


def prepare_dataset():
    """Create the continuous rotation dataset if it doesn't exist."""
    if os.path.exists(os.path.join(DATA_DIR, 'train.npy')):
        print("Dataset already exists, skipping preparation.")
        return
    
    print("Preparing continuous rotation dataset...")
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # Configuration
    DIM = 16
    SEQ_LEN = 32
    N_TRAIN = 10000
    N_VAL = 1000
    ANGLE_RANGE = (5, 60)
    
    def random_rotation_matrix(dim, angle_deg):
        A = np.random.randn(dim, dim)
        A = (A - A.T) / 2
        A = A / np.linalg.norm(A) * np.radians(angle_deg)
        I = np.eye(dim)
        R = np.linalg.solve(I + A/2, I - A/2)
        return R
    
    def generate_sequence(dim, seq_len, angle_deg):
        v0 = np.random.randn(dim)
        v0 = v0 / np.linalg.norm(v0)
        R = random_rotation_matrix(dim, angle_deg)
        sequence = [v0]
        v = v0
        for _ in range(seq_len - 1):
            v = R @ v
            sequence.append(v)
        return np.array(sequence)
    
    def create_dataset(n_samples, dim, seq_len, angle_range):
        sequences = []
        for _ in range(n_samples):
            angle = np.random.uniform(*angle_range)
            seq = generate_sequence(dim, seq_len, angle)
            sequences.append(seq)
        return np.array(sequences)
    
    train_data = create_dataset(N_TRAIN, DIM, SEQ_LEN, ANGLE_RANGE)
    val_data = create_dataset(N_VAL, DIM, SEQ_LEN, ANGLE_RANGE)
    
    np.save(os.path.join(DATA_DIR, 'train.npy'), train_data.astype(np.float32))
    np.save(os.path.join(DATA_DIR, 'val.npy'), val_data.astype(np.float32))
    
    print(f"  Train: {train_data.shape}")
    print(f"  Val: {val_data.shape}")
    print("Dataset prepared!")


# =============================================================================
# Model Definitions (from train_continuous_rotation.py)
# =============================================================================

class BaselineLayer(nn.Module):
    """Standard MLP with residual connection."""
    def __init__(self, hidden_dim):
        super().__init__()
        self.ln = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
    
    def forward(self, x):
        return x + self.mlp(self.ln(x))


class DDLLayer(nn.Module):
    """Deep Delta Learning layer with Householder reflection."""
    def __init__(self, hidden_dim):
        super().__init__()
        self.ln = nn.LayerNorm(hidden_dim)
        self.k_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, hidden_dim),
        )
        self.beta_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid(),
        )
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
    
    def forward(self, x):
        x_norm = self.ln(x)
        k = F.normalize(self.k_net(x_norm.mean(dim=1)), dim=-1)
        beta = 2 * self.beta_net(x_norm.mean(dim=1))
        k = k.unsqueeze(1)
        dot = (x_norm * k).sum(dim=-1, keepdim=True)
        x_reflected = x_norm - beta.unsqueeze(1) * dot * k
        return x_reflected + self.mlp(x_reflected)


class DDCLayer(nn.Module):
    """Data-Dependent Cayley layer with GUARANTEED orthogonality."""
    def __init__(self, hidden_dim, n_streams=4):
        super().__init__()
        self.n_streams = n_streams
        self.d_stream = hidden_dim // n_streams
        self.hidden_dim = hidden_dim
        
        self.ln = nn.LayerNorm(hidden_dim)
        
        self.u_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, n_streams),
        )
        self.v_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, n_streams),
        )
        self.beta_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Softplus(),
        )
        
        self.alpha_gate = nn.Linear(hidden_dim, 1)
        
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        
        self.register_buffer('I', torch.eye(n_streams))
        
        nn.init.normal_(self.u_net[-1].weight, std=0.01)
        nn.init.normal_(self.v_net[-1].weight, std=0.01)
        nn.init.zeros_(self.alpha_gate.weight)
        nn.init.constant_(self.alpha_gate.bias, -1.0)
    
    def forward(self, x):
        B, T, D = x.shape
        x_norm = self.ln(x)
        x_pooled = x_norm.mean(dim=1)
        
        u = self.u_net(x_pooled)
        v = self.v_net(x_pooled)
        beta = self.beta_net(x_pooled).unsqueeze(-1)
        
        # Skew-symmetric (ALWAYS!) => Cayley is ALWAYS orthogonal
        A = torch.einsum('bi,bj->bij', u, v) - torch.einsum('bi,bj->bij', v, u)
        
        M = (beta / 2) * A
        I = self.I.unsqueeze(0).expand(B, -1, -1)
        Q = torch.linalg.solve(I + M, I - M)  # GUARANTEED orthogonal
        
        x_streams = x_norm.view(B, T, self.n_streams, self.d_stream)
        x_rotated = torch.einsum('bij,btjd->btid', Q, x_streams)
        x_rotated = x_rotated.reshape(B, T, D)
        
        alpha = torch.sigmoid(self.alpha_gate(x_pooled)).unsqueeze(1)
        x_out = alpha * x_rotated + (1 - alpha) * x_norm
        
        return x_out + self.mlp(x_out)


class DDCHybridLayer(nn.Module):
    """DDC + Householder hybrid with learned gate."""
    def __init__(self, hidden_dim, n_streams=4):
        super().__init__()
        self.n_streams = n_streams
        self.d_stream = hidden_dim // n_streams
        self.hidden_dim = hidden_dim
        
        self.ln = nn.LayerNorm(hidden_dim)
        
        # DDC components
        self.u_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, n_streams),
        )
        self.v_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, n_streams),
        )
        self.beta_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Softplus(),
        )
        
        # Householder components
        self.k_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, hidden_dim),
        )
        
        # Gate: rotation (DDC) vs reflection (Householder)
        self.gate = nn.Linear(hidden_dim, 1)
        
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
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
        beta = self.beta_net(x_pooled).unsqueeze(-1)
        
        A = torch.einsum('bi,bj->bij', u, v) - torch.einsum('bi,bj->bij', v, u)
        M = (beta / 2) * A
        I = self.I.unsqueeze(0).expand(B, -1, -1)
        Q = torch.linalg.solve(I + M, I - M)
        
        x_streams = x_norm.view(B, T, self.n_streams, self.d_stream)
        x_rotated = torch.einsum('bij,btjd->btid', Q, x_streams)
        x_rotated = x_rotated.reshape(B, T, D)
        
        # === Householder Reflection (β=2 for orthogonality) ===
        k = F.normalize(self.k_net(x_pooled), dim=-1).unsqueeze(1)
        dot = (x_norm * k).sum(dim=-1, keepdim=True)
        x_reflected = x_norm - 2 * dot * k
        
        # === Learned gate ===
        gamma = torch.sigmoid(self.gate(x_pooled)).unsqueeze(1)
        x_hybrid = gamma * x_rotated + (1 - gamma) * x_reflected
        
        return x_hybrid + self.mlp(x_hybrid)


class ContinuousRotationModel(nn.Module):
    """Model for continuous rotation prediction."""
    def __init__(self, dim, hidden_dim, n_layers, operator_type='baseline'):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.operator_type = operator_type
        
        self.input_proj = nn.Linear(dim, hidden_dim)
        
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            if operator_type == 'baseline':
                self.layers.append(BaselineLayer(hidden_dim))
            elif operator_type == 'ddl':
                self.layers.append(DDLLayer(hidden_dim))
            elif operator_type == 'ddc':
                self.layers.append(DDCLayer(hidden_dim))
            elif operator_type == 'ddc_hybrid':
                self.layers.append(DDCHybridLayer(hidden_dim))
        
        self.output_proj = nn.Linear(hidden_dim, dim)
        self.ln = nn.LayerNorm(hidden_dim)
    
    def forward(self, x):
        h = self.input_proj(x)
        for layer in self.layers:
            h = layer(h)
        h = self.ln(h)
        return self.output_proj(h)


# =============================================================================
# Training Function
# =============================================================================

def train_model(model_type, dim, hidden_dim, n_layers, train_data, val_data,
                max_iters=3000, batch_size=64, lr=1e-3, device='cuda'):
    """Train a model and return comprehensive metrics."""
    
    model = ContinuousRotationModel(dim, hidden_dim, n_layers, model_type).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    
    n_params = sum(p.numel() for p in model.parameters())
    
    def get_batch(data):
        idx = torch.randint(len(data), (batch_size,))
        batch = data[idx].to(device)
        x = batch[:, :-1, :]
        y = batch[:, 1:, :]
        return x, y
    
    # Metrics tracking
    train_losses = []
    val_losses = []
    norm_errors = []
    grad_norms = []
    
    best_val_loss = float('inf')
    best_norm_error = float('inf')
    
    t0 = time.time()
    
    for it in range(max_iters):
        model.train()
        x, y = get_batch(train_data)
        
        pred = model(x)
        loss = F.mse_loss(pred, y)
        
        optimizer.zero_grad()
        loss.backward()
        
        # Track gradient norm
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        total_norm = total_norm ** 0.5
        grad_norms.append(total_norm)
        
        # Check for explosion
        if torch.isnan(loss) or total_norm > 1e6:
            print(f"  {model_type}: EXPLODED at iter {it}")
            return {
                'converged': False,
                'best_val_loss': float('inf'),
                'best_norm_error': float('inf'),
                'n_params': n_params,
            }
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        train_losses.append(loss.item())
        
        # Evaluation
        if it % 200 == 0:
            model.eval()
            with torch.no_grad():
                val_loss_sum = 0
                norm_error_sum = 0
                for _ in range(20):
                    x, y = get_batch(val_data)
                    pred = model(x)
                    val_loss_sum += F.mse_loss(pred, y).item()
                    
                    pred_norms = torch.norm(pred, dim=-1)
                    target_norms = torch.norm(y, dim=-1)
                    norm_error_sum += (pred_norms - target_norms).abs().mean().item()
                
                val_loss = val_loss_sum / 20
                norm_error = norm_error_sum / 20
                
                val_losses.append((it, val_loss))
                norm_errors.append((it, norm_error))
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_norm_error = norm_error
            
            print(f"    {model_type:12s} iter {it:4d}: train={loss.item():.6f}, "
                  f"val={val_loss:.6f}, norm_err={norm_error:.6f}, grad={total_norm:.2f}")
    
    train_time = time.time() - t0
    
    return {
        'converged': True,
        'n_params': n_params,
        'best_val_loss': best_val_loss,
        'best_norm_error': best_norm_error,
        'final_train_loss': train_losses[-1],
        'mean_grad_norm': np.mean(grad_norms[-500:]) if len(grad_norms) > 500 else np.mean(grad_norms),
        'max_grad_norm': np.max(grad_norms),
        'train_time': train_time,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'norm_errors': norm_errors,
        'grad_norms': grad_norms,
    }


# =============================================================================
# Main Comparison
# =============================================================================

def run_comparison():
    print("=" * 80)
    print("CONTINUOUS ROTATION TASK: DDC vs DDL vs Baseline vs DDC-Hybrid")
    print("=" * 80)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # Prepare dataset
    prepare_dataset()
    
    # Load data
    train_data = torch.from_numpy(np.load(os.path.join(DATA_DIR, 'train.npy'))).float()
    val_data = torch.from_numpy(np.load(os.path.join(DATA_DIR, 'val.npy'))).float()
    
    dim = train_data.shape[-1]
    print(f"\nDataset loaded:")
    print(f"  Train: {train_data.shape}")
    print(f"  Val: {val_data.shape}")
    print(f"  Vector dim: {dim}")
    
    # Configuration
    hidden_dim = 128
    depths = [4, 8, 16]  # Test at different depths
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
                max_iters=2000,
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
            if r['converged']:
                print(f"{r['best_val_loss']:<15.6f}", end="")
            else:
                print(f"{'FAILED':<15}", end="")
        print()
    
    print("\n" + "-" * 80)
    print("NORM PRESERVATION ERROR - lower is better (critical metric!)")
    print("-" * 80)
    print(f"{'Depth':<8}", end="")
    for mt in model_types:
        print(f"{mt:<15}", end="")
    print()
    
    for depth in depths:
        print(f"{depth:<8}", end="")
        for mt in model_types:
            r = results[mt][depth]
            if r['converged']:
                print(f"{r['best_norm_error']:<15.6f}", end="")
            else:
                print(f"{'FAILED':<15}", end="")
        print()
    
    print("\n" + "-" * 80)
    print("GRADIENT STABILITY - lower is better")
    print("-" * 80)
    print(f"{'Depth':<8}", end="")
    for mt in model_types:
        print(f"{mt:<15}", end="")
    print()
    
    for depth in depths:
        print(f"{depth:<8}", end="")
        for mt in model_types:
            r = results[mt][depth]
            if r['converged']:
                print(f"{r['mean_grad_norm']:<15.2f}", end="")
            else:
                print(f"{'FAILED':<15}", end="")
        print()
    
    # Winner analysis
    print("\n" + "=" * 80)
    print("WINNERS AT EACH DEPTH")
    print("=" * 80)
    
    for depth in depths:
        print(f"\nDepth {depth}:")
        
        # Best val loss
        converged = [(mt, results[mt][depth]['best_val_loss']) 
                     for mt in model_types if results[mt][depth]['converged']]
        if converged:
            winner = min(converged, key=lambda x: x[1])
            print(f"  Best Val Loss: {winner[0]} ({winner[1]:.6f})")
        
        # Best norm preservation
        converged = [(mt, results[mt][depth]['best_norm_error']) 
                     for mt in model_types if results[mt][depth]['converged']]
        if converged:
            winner = min(converged, key=lambda x: x[1])
            print(f"  Best Norm Preservation: {winner[0]} ({winner[1]:.6f})")
    
    # Plot results
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    colors = {'baseline': 'gray', 'ddl': 'red', 'ddc': 'blue', 'ddc_hybrid': 'green'}
    
    # Plot 1: Val loss vs depth
    ax = axes[0, 0]
    for mt in model_types:
        losses = [results[mt][d]['best_val_loss'] for d in depths if results[mt][d]['converged']]
        valid_depths = [d for d in depths if results[mt][d]['converged']]
        if losses:
            ax.plot(valid_depths, losses, 'o-', label=mt, color=colors[mt], 
                   markersize=10, linewidth=2)
    ax.set_xlabel('Depth (layers)', fontsize=12)
    ax.set_ylabel('Best Validation Loss (MSE)', fontsize=12)
    ax.set_title('Validation Loss vs Depth', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Norm error vs depth
    ax = axes[0, 1]
    for mt in model_types:
        errors = [results[mt][d]['best_norm_error'] for d in depths if results[mt][d]['converged']]
        valid_depths = [d for d in depths if results[mt][d]['converged']]
        if errors:
            ax.plot(valid_depths, errors, 'o-', label=mt, color=colors[mt],
                   markersize=10, linewidth=2)
    ax.set_xlabel('Depth (layers)', fontsize=12)
    ax.set_ylabel('Norm Preservation Error', fontsize=12)
    ax.set_title('Norm Preservation vs Depth (CRITICAL)', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Training curves at max depth
    ax = axes[1, 0]
    max_depth = max(depths)
    for mt in model_types:
        if results[mt][max_depth]['converged']:
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
    
    # Plot 4: Gradient norms
    ax = axes[1, 1]
    for mt in model_types:
        if results[mt][max_depth]['converged']:
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
    plt.savefig('continuous_rotation_comparison.png', dpi=150)
    print(f"\nPlot saved to: continuous_rotation_comparison.png")
    
    # Key insights
    print("\n" + "=" * 80)
    print("KEY INSIGHTS")
    print("=" * 80)
    print("""
This experiment tests TRUE geometric rotation capability:

1. BASELINE: Standard residual - no geometric awareness
2. DDL: Householder reflection - can reflect but NOT always orthogonal
3. DDC: Data-Dependent Cayley - UNCONDITIONALLY orthogonal (||Qx|| = ||x||)
4. DDC-HYBRID: DDC + Householder - rotation + reflection capabilities

Critical metrics:
- NORM PRESERVATION: DDC should excel (Cayley is always isometric)
- GRADIENT STABILITY: DDC's orthogonality prevents explosion
- VALIDATION LOSS: Shows overall task performance

Expected DDC advantages:
- Perfect norm preservation (critical for geometric tasks)
- More stable training at deep networks
- No singularity issues (unlike DDL at β=1)
""")


if __name__ == '__main__':
    run_comparison()
