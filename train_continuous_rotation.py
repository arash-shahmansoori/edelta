"""
Training script for Continuous Vector Rotation task.

This is NOT a language modeling task - it's a direct vector-to-vector prediction:
- Input: sequence of vectors [v0, v1, ..., v_{t-1}]
- Output: predict v_t
- Loss: MSE (not cross-entropy)

This tests true geometric capabilities:
- Norm preservation (DDC should excel)
- Rotation accuracy
- Long-term stability
"""

import os
import sys
import time
import math
import pickle
import argparse
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# =============================================================================
# Continuous Rotation Model (simplified, non-autoregressive)
# =============================================================================

class ContinuousRotationModel(nn.Module):
    """
    Base class for continuous rotation prediction.
    Takes a sequence of vectors and predicts the next vector.
    """
    def __init__(self, dim, hidden_dim, n_layers, operator_type='baseline'):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.operator_type = operator_type
        
        # Input projection
        self.input_proj = nn.Linear(dim, hidden_dim)
        
        # Processing layers
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(self._make_layer(hidden_dim, operator_type))
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, dim)
        
        # Layer norm
        self.ln = nn.LayerNorm(hidden_dim)
    
    def _make_layer(self, hidden_dim, operator_type):
        if operator_type == 'baseline':
            return BaselineLayer(hidden_dim)
        elif operator_type == 'ddl':
            return DDLLayer(hidden_dim)
        elif operator_type == 'ddc':
            return DDCLayer(hidden_dim)
        elif operator_type == 'ddc_hybrid':
            return DDCHybridLayer(hidden_dim)
        else:
            raise ValueError(f"Unknown operator type: {operator_type}")
    
    def forward(self, x):
        """
        x: (B, T, dim) - sequence of vectors
        Returns: (B, T, dim) - predicted next vectors
        """
        # Project to hidden dimension
        h = self.input_proj(x)  # (B, T, hidden_dim)
        
        # Apply layers
        for layer in self.layers:
            h = layer(h)
        
        # Project back to vector dimension
        h = self.ln(h)
        out = self.output_proj(h)  # (B, T, dim)
        
        return out


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
        
        # Householder reflection
        k = F.normalize(self.k_net(x_norm.mean(dim=1)), dim=-1)  # (B, D)
        beta = 2 * self.beta_net(x_norm.mean(dim=1))  # (B, 1) in [0, 2]
        
        k = k.unsqueeze(1)  # (B, 1, D)
        dot = (x_norm * k).sum(dim=-1, keepdim=True)
        x_reflected = x_norm - beta.unsqueeze(1) * dot * k
        
        return x_reflected + self.mlp(x_reflected)


class DDCLayer(nn.Module):
    """Data-Dependent Cayley layer with guaranteed orthogonality."""
    def __init__(self, hidden_dim, n_streams=4):
        super().__init__()
        self.n_streams = n_streams
        self.d_stream = hidden_dim // n_streams
        self.hidden_dim = hidden_dim
        
        self.ln = nn.LayerNorm(hidden_dim)
        
        # Generator networks for u(x), v(x)
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
        
        # Residual gate
        self.alpha_gate = nn.Linear(hidden_dim, 1)
        
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
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
        x_pooled = x_norm.mean(dim=1)  # (B, D)
        
        # Compute data-dependent generators
        u = self.u_net(x_pooled)  # (B, n_streams)
        v = self.v_net(x_pooled)  # (B, n_streams)
        beta = self.beta_net(x_pooled).unsqueeze(-1)  # (B, 1, 1)
        
        # Skew-symmetric A = uv^T - vu^T (ALWAYS skew-symmetric)
        A = torch.einsum('bi,bj->bij', u, v) - torch.einsum('bi,bj->bij', v, u)
        
        # Cayley transform: Q = (I + βA/2)^{-1} (I - βA/2)
        M = (beta / 2) * A
        I = self.I.unsqueeze(0).expand(B, -1, -1)
        Q = torch.linalg.solve(I + M, I - M)  # (B, n, n) - GUARANTEED orthogonal!
        
        # Apply rotation to streams
        x_streams = x_norm.view(B, T, self.n_streams, self.d_stream)
        x_rotated = torch.einsum('bij,btjd->btid', Q, x_streams)
        x_rotated = x_rotated.reshape(B, T, D)
        
        # Residual gate (allows learning "no rotation" = classical residual)
        alpha = torch.sigmoid(self.alpha_gate(x_pooled)).unsqueeze(1)  # (B, 1, 1)
        x_out = alpha * x_rotated + (1 - alpha) * x_norm
        
        return x_out + self.mlp(x_out)


class DDCHybridLayer(nn.Module):
    """DDC + Householder hybrid with learned gate."""
    def __init__(self, hidden_dim, n_streams=4):
        super().__init__()
        self.ddc = DDCLayer(hidden_dim, n_streams)
        self.ddc.mlp = nn.Identity()  # Remove MLP from DDC, add it after hybrid
        
        self.ln = nn.LayerNorm(hidden_dim)
        
        # Householder components
        self.k_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, hidden_dim),
        )
        
        # Gate: rotation vs reflection
        self.gate = nn.Linear(hidden_dim, 1)
        
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)
    
    def forward(self, x):
        B, T, D = x.shape
        x_norm = self.ln(x)
        x_pooled = x_norm.mean(dim=1)
        
        # DDC rotation
        x_rotated = self.ddc(x_norm)
        
        # Householder reflection (β=2 for orthogonality)
        k = F.normalize(self.k_net(x_pooled), dim=-1).unsqueeze(1)
        dot = (x_norm * k).sum(dim=-1, keepdim=True)
        x_reflected = x_norm - 2 * dot * k
        
        # Learned gate
        gamma = torch.sigmoid(self.gate(x_pooled)).unsqueeze(1)
        x_hybrid = gamma * x_rotated + (1 - gamma) * x_reflected
        
        return x_hybrid + self.mlp(x_hybrid)


# =============================================================================
# Training Loop
# =============================================================================

def train(args):
    print(f"Training {args.model_type} on continuous rotation task")
    print(f"  Vector dim: {args.dim}")
    print(f"  Hidden dim: {args.hidden_dim}")
    print(f"  Layers: {args.n_layers}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Max iters: {args.max_iters}")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"  Device: {device}")
    
    # Load data
    data_dir = 'data/continuous_rotation'
    train_data = np.load(os.path.join(data_dir, 'train.npy'))
    val_data = np.load(os.path.join(data_dir, 'val.npy'))
    
    train_data = torch.from_numpy(train_data).float()
    val_data = torch.from_numpy(val_data).float()
    
    print(f"  Train data: {train_data.shape}")
    print(f"  Val data: {val_data.shape}")
    
    # Create model
    model = ContinuousRotationModel(
        dim=args.dim,
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
        operator_type=args.model_type
    ).to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    
    # Training loop
    def get_batch(data, batch_size):
        idx = torch.randint(len(data), (batch_size,))
        batch = data[idx].to(device)
        # Input: all but last, Target: all but first
        x = batch[:, :-1, :]
        y = batch[:, 1:, :]
        return x, y
    
    @torch.no_grad()
    def evaluate():
        model.eval()
        losses = []
        norm_errors = []
        for _ in range(args.eval_iters):
            x, y = get_batch(val_data, args.batch_size)
            pred = model(x)
            loss = F.mse_loss(pred, y)
            losses.append(loss.item())
            
            # Compute norm preservation error
            pred_norms = torch.norm(pred, dim=-1)
            target_norms = torch.norm(y, dim=-1)
            norm_error = (pred_norms - target_norms).abs().mean()
            norm_errors.append(norm_error.item())
        
        model.train()
        return np.mean(losses), np.mean(norm_errors)
    
    # Training
    best_val_loss = float('inf')
    out_dir = f'out-continuous-{args.model_type}'
    os.makedirs(out_dir, exist_ok=True)
    
    t0 = time.time()
    for iter_num in range(args.max_iters):
        # Get batch
        x, y = get_batch(train_data, args.batch_size)
        
        # Forward
        pred = model(x)
        loss = F.mse_loss(pred, y)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        # Logging
        if iter_num % args.log_interval == 0:
            t1 = time.time()
            dt = t1 - t0
            t0 = t1
            print(f"iter {iter_num}: loss {loss.item():.6f}, time {dt*1000:.1f}ms")
        
        # Evaluation
        if iter_num % args.eval_interval == 0:
            val_loss, norm_error = evaluate()
            print(f"  -> val_loss: {val_loss:.6f}, norm_error: {norm_error:.6f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                checkpoint = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'norm_error': norm_error,
                    'args': vars(args),
                }
                torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
                print(f"  -> Saved checkpoint (best_val_loss: {best_val_loss:.6f})")
    
    print(f"\nTraining complete!")
    print(f"Best val loss: {best_val_loss:.6f}")
    return best_val_loss


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, default='baseline', 
                        choices=['baseline', 'ddl', 'ddc', 'ddc_hybrid'])
    parser.add_argument('--dim', type=int, default=16)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--n_layers', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--max_iters', type=int, default=5000)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--eval_interval', type=int, default=500)
    parser.add_argument('--eval_iters', type=int, default=50)
    parser.add_argument('--log_interval', type=int, default=100)
    args = parser.parse_args()
    
    train(args)
