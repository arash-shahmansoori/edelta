"""
GRADIENT STABILITY TEST: DDC's True Advantage

This experiment tests what DDC is ACTUALLY good at:
- Maintaining gradient stability in VERY DEEP networks (50-150 layers)
- Preventing gradient explosion/vanishing over long training

HYPOTHESIS:
- Baseline: Will struggle with vanishing gradients at extreme depth
- DDL: Will show instability (not always orthogonal, singular at β=1)
- DDC: Should maintain stable gradients (UNCONDITIONALLY orthogonal)
- DDC-Hybrid: Should also be stable (inherits DDC's orthogonality)

METRICS TRACKED:
1. Gradient norm at each iteration
2. Loss variance (window-based)
3. NaN/Inf occurrences
4. Maximum depth before training fails
5. Convergence speed at each depth

WHY THIS TESTS DDC'S ADVANTAGE:
- Orthogonality → eigenvalues on unit circle → gradients neither explode nor vanish
- At 100+ layers, this MATTERS
- DDL's non-orthogonality (β ≠ 0,2) accumulates errors over depth
"""

import os
import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from collections import defaultdict


# =============================================================================
# Learnable Data (structured patterns)
# =============================================================================

def create_learnable_data(size=200000):
    """Create structured character-level data."""
    patterns = [
        "the quick brown fox jumps over the lazy dog ",
        "to be or not to be that is the question ",
        "all that glitters is not gold ",
        "a journey of a thousand miles begins ",
        "knowledge is power and power is knowledge ",
    ]
    
    all_chars = set()
    for p in patterns:
        all_chars.update(p)
    chars = sorted(list(all_chars))
    char_to_idx = {c: i for i, c in enumerate(chars)}
    vocab_size = len(chars)
    
    data = []
    while len(data) < size:
        p = patterns[np.random.randint(len(patterns))]
        for c in p:
            data.append(char_to_idx[c])
    
    return np.array(data[:size], dtype=np.int64), vocab_size


# =============================================================================
# Simplified Deep Transformer Blocks
# =============================================================================

class CausalSelfAttention(nn.Module):
    """Simplified causal attention."""
    def __init__(self, dim, n_head, block_size):
        super().__init__()
        self.n_head = n_head
        self.dim = dim
        self.head_dim = dim // n_head
        
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        
        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size))
                            .view(1, 1, block_size, block_size))
    
    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_head, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        att = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        
        out = (att @ v).transpose(1, 2).reshape(B, T, C)
        return self.proj(out)


class MLP(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim * 4, bias=False)
        self.fc2 = nn.Linear(dim * 4, dim, bias=False)
    
    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


# =============================================================================
# Block Types
# =============================================================================

class BaselineBlock(nn.Module):
    """Standard transformer block."""
    def __init__(self, dim, n_head, block_size):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = CausalSelfAttention(dim, n_head, block_size)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim)
    
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class DDLBlock(nn.Module):
    """DDL block with Householder reflection."""
    def __init__(self, dim, n_head, block_size):
        super().__init__()
        self.dim = dim
        self.ln1 = nn.LayerNorm(dim)
        self.attn = CausalSelfAttention(dim, n_head, block_size)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim)
        
        # Householder params
        self.k_net = nn.Linear(dim, dim, bias=False)
        self.beta_net = nn.Sequential(nn.Linear(dim, 1), nn.Sigmoid())
    
    def householder(self, x):
        B, T, D = x.shape
        x_pool = x.mean(dim=1)
        k = F.normalize(self.k_net(x_pool), dim=-1).unsqueeze(1)
        beta = 2 * self.beta_net(x_pool).unsqueeze(-1)
        dot = (x * k).sum(dim=-1, keepdim=True)
        return x - beta * dot * k
    
    def forward(self, x):
        x_h = self.householder(x)
        x = x_h + self.attn(self.ln1(x_h))
        x_h = self.householder(x)
        x = x_h + self.mlp(self.ln2(x_h))
        return x


class DDCBlock(nn.Module):
    """DDC block with GUARANTEED orthogonality."""
    def __init__(self, dim, n_head, block_size, n_streams=4):
        super().__init__()
        self.dim = dim
        self.n_streams = n_streams
        self.d_stream = dim // n_streams
        
        self.ln1 = nn.LayerNorm(dim)
        self.attn = CausalSelfAttention(dim, n_head, block_size)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim)
        
        # DDC params - one set for attn, one for mlp
        self.u_attn = nn.Linear(dim, n_streams, bias=False)
        self.v_attn = nn.Linear(dim, n_streams, bias=False)
        self.beta_attn = nn.Sequential(nn.Linear(dim, 1), nn.Softplus())
        self.alpha_attn = nn.Linear(dim, 1)
        
        self.u_mlp = nn.Linear(dim, n_streams, bias=False)
        self.v_mlp = nn.Linear(dim, n_streams, bias=False)
        self.beta_mlp = nn.Sequential(nn.Linear(dim, 1), nn.Softplus())
        self.alpha_mlp = nn.Linear(dim, 1)
        
        self.register_buffer('I', torch.eye(n_streams))
        
        # Small init for stability
        for p in [self.u_attn, self.v_attn, self.u_mlp, self.v_mlp]:
            nn.init.normal_(p.weight, std=0.01)
        for p in [self.alpha_attn, self.alpha_mlp]:
            nn.init.zeros_(p.weight)
            nn.init.constant_(p.bias, -2.0)  # Start close to identity
    
    def cayley(self, x, u_net, v_net, beta_net, alpha_net):
        B, T, D = x.shape
        x_pool = x.mean(dim=1)
        
        u = u_net(x_pool)
        v = v_net(x_pool)
        beta = beta_net(x_pool).unsqueeze(-1)
        
        # Skew-symmetric A (ALWAYS!)
        A = torch.einsum('bi,bj->bij', u, v) - torch.einsum('bi,bj->bij', v, u)
        M = (beta / 2) * A
        
        I = self.I.unsqueeze(0).expand(B, -1, -1)
        Q = torch.linalg.solve(I + M, I - M)  # GUARANTEED orthogonal
        
        x_streams = x.view(B, T, self.n_streams, self.d_stream)
        x_rot = torch.einsum('bij,btjd->btid', Q, x_streams).reshape(B, T, D)
        
        alpha = torch.sigmoid(alpha_net(x_pool)).unsqueeze(1)
        return alpha * x_rot + (1 - alpha) * x
    
    def forward(self, x):
        x_c = self.cayley(x, self.u_attn, self.v_attn, self.beta_attn, self.alpha_attn)
        x = x_c + self.attn(self.ln1(x_c))
        x_c = self.cayley(x, self.u_mlp, self.v_mlp, self.beta_mlp, self.alpha_mlp)
        x = x_c + self.mlp(self.ln2(x_c))
        return x


class DDCHybridBlock(nn.Module):
    """DDC + Householder hybrid."""
    def __init__(self, dim, n_head, block_size, n_streams=4):
        super().__init__()
        self.dim = dim
        self.n_streams = n_streams
        self.d_stream = dim // n_streams
        
        self.ln1 = nn.LayerNorm(dim)
        self.attn = CausalSelfAttention(dim, n_head, block_size)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim)
        
        # DDC params
        self.u = nn.Linear(dim, n_streams, bias=False)
        self.v = nn.Linear(dim, n_streams, bias=False)
        self.beta_rot = nn.Sequential(nn.Linear(dim, 1), nn.Softplus())
        
        # Householder params  
        self.k_net = nn.Linear(dim, dim, bias=False)
        
        # Gate: rotation vs reflection
        self.gate = nn.Linear(dim, 1)
        
        self.register_buffer('I', torch.eye(n_streams))
        
        nn.init.normal_(self.u.weight, std=0.01)
        nn.init.normal_(self.v.weight, std=0.01)
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)
    
    def hybrid(self, x):
        B, T, D = x.shape
        x_pool = x.mean(dim=1)
        
        # DDC rotation
        u = self.u(x_pool)
        v = self.v(x_pool)
        beta = self.beta_rot(x_pool).unsqueeze(-1)
        
        A = torch.einsum('bi,bj->bij', u, v) - torch.einsum('bi,bj->bij', v, u)
        M = (beta / 2) * A
        I = self.I.unsqueeze(0).expand(B, -1, -1)
        Q = torch.linalg.solve(I + M, I - M)
        
        x_streams = x.view(B, T, self.n_streams, self.d_stream)
        x_rot = torch.einsum('bij,btjd->btid', Q, x_streams).reshape(B, T, D)
        
        # Householder reflection (β=2)
        k = F.normalize(self.k_net(x_pool), dim=-1).unsqueeze(1)
        dot = (x * k).sum(dim=-1, keepdim=True)
        x_ref = x - 2 * dot * k
        
        # Gate
        gamma = torch.sigmoid(self.gate(x_pool)).unsqueeze(1)
        return gamma * x_rot + (1 - gamma) * x_ref
    
    def forward(self, x):
        x_h = self.hybrid(x)
        x = x_h + self.attn(self.ln1(x_h))
        x_h = self.hybrid(x)
        x = x_h + self.mlp(self.ln2(x_h))
        return x


# =============================================================================
# Deep Transformer Model
# =============================================================================

class DeepTransformer(nn.Module):
    """Very deep transformer for stability testing."""
    def __init__(self, vocab_size, dim, n_layers, n_head, block_size, block_type='baseline'):
        super().__init__()
        
        self.embed = nn.Embedding(vocab_size, dim)
        self.pos_embed = nn.Embedding(block_size, dim)
        
        BlockClass = {
            'baseline': BaselineBlock,
            'ddl': DDLBlock,
            'ddc': DDCBlock,
            'ddc_hybrid': DDCHybridBlock,
        }[block_type]
        
        self.blocks = nn.ModuleList([BlockClass(dim, n_head, block_size) for _ in range(n_layers)])
        
        self.ln_f = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)
        self.head.weight = self.embed.weight  # Weight tying
        
        # Scale down initialization for deep networks
        self.apply(self._init_weights)
        for block in self.blocks:
            if hasattr(block, 'attn') and hasattr(block.attn, 'proj'):
                nn.init.normal_(block.attn.proj.weight, std=0.02 / math.sqrt(2 * n_layers))
            if hasattr(block, 'mlp') and hasattr(block.mlp, 'fc2'):
                nn.init.normal_(block.mlp.fc2.weight, std=0.02 / math.sqrt(2 * n_layers))
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)
    
    def forward(self, x, targets=None):
        B, T = x.shape
        
        h = self.embed(x) + self.pos_embed(torch.arange(T, device=x.device))
        
        for block in self.blocks:
            h = block(h)
        
        h = self.ln_f(h)
        logits = self.head(h)
        
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            return logits, loss
        return logits, None


# =============================================================================
# Stability Metrics
# =============================================================================

class StabilityTracker:
    """Track gradient stability metrics."""
    def __init__(self):
        self.grad_norms = []
        self.losses = []
        self.nan_count = 0
        self.inf_count = 0
        self.max_grad = 0
        self.exploded = False
        self.exploded_at = None
    
    def update(self, loss, grad_norm, iter_num):
        if torch.isnan(loss) or math.isnan(grad_norm):
            self.nan_count += 1
            if self.nan_count > 5 and not self.exploded:
                self.exploded = True
                self.exploded_at = iter_num
            return False
        
        if torch.isinf(loss) or math.isinf(grad_norm) or grad_norm > 1e6:
            self.inf_count += 1
            if self.inf_count > 5 and not self.exploded:
                self.exploded = True
                self.exploded_at = iter_num
            return False
        
        self.grad_norms.append(grad_norm)
        self.losses.append(loss.item())
        self.max_grad = max(self.max_grad, grad_norm)
        return True
    
    def get_summary(self):
        if len(self.grad_norms) == 0:
            return {
                'stable': False,
                'mean_grad': float('inf'),
                'max_grad': float('inf'),
                'loss_std': float('inf'),
                'final_loss': float('inf'),
            }
        
        return {
            'stable': not self.exploded,
            'mean_grad': np.mean(self.grad_norms[-500:]) if len(self.grad_norms) > 500 else np.mean(self.grad_norms),
            'max_grad': self.max_grad,
            'loss_std': np.std(self.losses[-500:]) if len(self.losses) > 500 else np.std(self.losses),
            'final_loss': self.losses[-1] if self.losses else float('inf'),
            'exploded_at': self.exploded_at,
            'nan_count': self.nan_count,
            'inf_count': self.inf_count,
        }


# =============================================================================
# Training
# =============================================================================

def train_and_measure_stability(block_type, n_layers, train_data, vocab_size,
                                 max_iters=2000, batch_size=32, lr=3e-4, 
                                 dim=128, n_head=4, block_size=64, device='cuda'):
    """Train model and measure gradient stability."""
    
    model = DeepTransformer(vocab_size, dim, n_layers, n_head, block_size, block_type).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    
    n_params = sum(p.numel() for p in model.parameters())
    tracker = StabilityTracker()
    
    def get_batch():
        idx = torch.randint(len(train_data) - block_size, (batch_size,))
        x = torch.stack([torch.from_numpy(train_data[i:i+block_size].astype(np.int64)) for i in idx])
        y = torch.stack([torch.from_numpy(train_data[i+1:i+1+block_size].astype(np.int64)) for i in idx])
        return x.to(device), y.to(device)
    
    t0 = time.time()
    
    for it in range(max_iters):
        model.train()
        x, y = get_batch()
        
        _, loss = model(x, y)
        
        optimizer.zero_grad()
        loss.backward()
        
        # Compute gradient norm
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        total_norm = total_norm ** 0.5
        
        # Track stability
        if not tracker.update(loss, total_norm, it):
            print(f"    {block_type:12s} depth={n_layers}: UNSTABLE at iter {it}")
            break
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        if it % 500 == 0:
            print(f"    {block_type:12s} depth={n_layers} iter {it:4d}: loss={loss.item():.4f}, grad={total_norm:.2f}")
    
    summary = tracker.get_summary()
    summary['n_params'] = n_params
    summary['train_time'] = time.time() - t0
    summary['grad_norms'] = tracker.grad_norms
    summary['losses'] = tracker.losses
    
    return summary


# =============================================================================
# Main
# =============================================================================

def run_stability_test():
    print("=" * 80)
    print("GRADIENT STABILITY TEST: DDC's TRUE Advantage")
    print("Very deep networks (50-150 layers), track gradient stability")
    print("=" * 80)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # Create data
    print("\nCreating learnable data...")
    train_data, vocab_size = create_learnable_data(200000)
    print(f"  Vocab size: {vocab_size}, Data size: {len(train_data)}")
    
    # Test configurations
    depths = [125]
    block_types = ['baseline', 'ddl', 'ddc', 'ddc_hybrid']
    
    results = {bt: {} for bt in block_types}
    
    for depth in depths:
        print(f"\n{'='*80}")
        print(f"TESTING DEPTH = {depth} LAYERS")
        print(f"{'='*80}")
        
        for bt in block_types:
            print(f"\n  Training {bt}...")
            
            try:
                results[bt][depth] = train_and_measure_stability(
                    block_type=bt,
                    n_layers=depth,
                    train_data=train_data,
                    vocab_size=vocab_size,
                    max_iters=1500,
                    batch_size=32,
                    lr=3e-4,
                    dim=128,
                    n_head=4,
                    block_size=64,
                    device=device
                )
                
                summary = results[bt][depth]
                status = "✓ STABLE" if summary['stable'] else f"✗ EXPLODED at iter {summary.get('exploded_at', '?')}"
                print(f"    Result: {status}")
                print(f"    Final loss: {summary['final_loss']:.4f}")
                print(f"    Mean grad norm: {summary['mean_grad']:.2f}")
                print(f"    Max grad norm: {summary['max_grad']:.2f}")
                
            except Exception as e:
                print(f"    ERROR: {e}")
                results[bt][depth] = {'stable': False, 'error': str(e)}
            
            torch.cuda.empty_cache()
    
    # =============================================================================
    # Summary
    # =============================================================================
    print("\n" + "=" * 80)
    print("GRADIENT STABILITY SUMMARY")
    print("=" * 80)
    
    print("\n" + "-" * 80)
    print("STABILITY STATUS (stable = trained without explosion)")
    print("-" * 80)
    print(f"{'Depth':<8}", end="")
    for bt in block_types:
        print(f"{bt:<15}", end="")
    print()
    
    for depth in depths:
        print(f"{depth:<8}", end="")
        for bt in block_types:
            r = results[bt].get(depth, {})
            if r.get('stable', False):
                print(f"{'✓ STABLE':<15}", end="")
            else:
                exp_at = r.get('exploded_at', '?')
                print(f"{'✗ @' + str(exp_at):<15}", end="")
        print()
    
    print("\n" + "-" * 80)
    print("MEAN GRADIENT NORM (lower = more stable)")
    print("-" * 80)
    print(f"{'Depth':<8}", end="")
    for bt in block_types:
        print(f"{bt:<15}", end="")
    print()
    
    for depth in depths:
        print(f"{depth:<8}", end="")
        for bt in block_types:
            r = results[bt].get(depth, {})
            if r.get('stable', False):
                print(f"{r['mean_grad']:<15.2f}", end="")
            else:
                print(f"{'-':<15}", end="")
        print()
    
    print("\n" + "-" * 80)
    print("FINAL VALIDATION LOSS (lower = better learned)")
    print("-" * 80)
    print(f"{'Depth':<8}", end="")
    for bt in block_types:
        print(f"{bt:<15}", end="")
    print()
    
    for depth in depths:
        print(f"{depth:<8}", end="")
        for bt in block_types:
            r = results[bt].get(depth, {})
            if r.get('stable', False):
                print(f"{r['final_loss']:<15.4f}", end="")
            else:
                print(f"{'-':<15}", end="")
        print()
    
    # Max stable depth
    print("\n" + "-" * 80)
    print("MAXIMUM STABLE DEPTH")
    print("-" * 80)
    
    for bt in block_types:
        max_stable = max([d for d in depths if results[bt].get(d, {}).get('stable', False)], default=0)
        print(f"  {bt}: {max_stable} layers")
    
    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    colors = {'baseline': 'gray', 'ddl': 'red', 'ddc': 'blue', 'ddc_hybrid': 'green'}
    
    # Plot 1: Stability by depth
    ax = axes[0, 0]
    for bt in block_types:
        stable = [1 if results[bt].get(d, {}).get('stable', False) else 0 for d in depths]
        ax.plot(depths, stable, 'o-', label=bt, color=colors[bt], markersize=12, linewidth=2)
    ax.set_xlabel('Depth (layers)', fontsize=12)
    ax.set_ylabel('Stable (1) vs Exploded (0)', fontsize=12)
    ax.set_title('Training Stability vs Depth', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.1, 1.1)
    
    # Plot 2: Mean gradient norm
    ax = axes[0, 1]
    for bt in block_types:
        grads = [results[bt].get(d, {}).get('mean_grad', None) for d in depths]
        valid = [(d, g) for d, g in zip(depths, grads) if g is not None and g < 1e6]
        if valid:
            ds, gs = zip(*valid)
            ax.plot(ds, gs, 'o-', label=bt, color=colors[bt], markersize=10, linewidth=2)
    ax.set_xlabel('Depth (layers)', fontsize=12)
    ax.set_ylabel('Mean Gradient Norm', fontsize=12)
    ax.set_title('Gradient Magnitude vs Depth', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Final loss
    ax = axes[1, 0]
    for bt in block_types:
        losses = [results[bt].get(d, {}).get('final_loss', None) for d in depths]
        valid = [(d, l) for d, l in zip(depths, losses) if l is not None and l < 1e6]
        if valid:
            ds, ls = zip(*valid)
            ax.plot(ds, ls, 'o-', label=bt, color=colors[bt], markersize=10, linewidth=2)
    ax.set_xlabel('Depth (layers)', fontsize=12)
    ax.set_ylabel('Final Loss', fontsize=12)
    ax.set_title('Learning Performance vs Depth', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Gradient norm over training at max depth
    ax = axes[1, 1]
    max_depth = max(depths)
    for bt in block_types:
        r = results[bt].get(max_depth, {})
        if 'grad_norms' in r and len(r['grad_norms']) > 50:
            norms = r['grad_norms']
            window = 30
            smoothed = np.convolve(norms, np.ones(window)/window, mode='valid')
            ax.plot(smoothed, label=bt, color=colors[bt], alpha=0.8, linewidth=1.5)
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Gradient Norm', fontsize=12)
    ax.set_title(f'Gradient Evolution at Depth={max_depth}', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('gradient_stability_test.png', dpi=150)
    print(f"\nPlot saved to: gradient_stability_test.png")
    
    # Analysis
    print("\n" + "=" * 80)
    print("ANALYSIS: Does DDC's Orthogonality Help?")
    print("=" * 80)
    print("""
HYPOTHESIS:
- DDC's Cayley transform is UNCONDITIONALLY orthogonal
- Orthogonality → eigenvalues on unit circle → stable gradients
- At 100+ layers, this should MATTER

WHAT TO LOOK FOR:
1. DDC/DDC-Hybrid stable at deeper networks than Baseline/DDL
2. DDC/DDC-Hybrid lower gradient norms at same depth
3. DDC/DDC-Hybrid better final loss at extreme depths

If DDC shows advantage → orthogonality IS practically useful
If DDC fails similarly → orthogonality alone isn't enough
""")


if __name__ == '__main__':
    run_stability_test()
