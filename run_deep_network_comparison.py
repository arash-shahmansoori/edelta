"""
DEEP NETWORK COMPARISON: DDC vs DDL vs Baseline

Tests:
1. Very deep networks (32, 64, 128 layers)
2. Real learnable task (character-level language modeling on structured data)
3. Comprehensive metrics: loss, gradient norms, training stability

Expected: DDC's unconditional orthogonality should show clear advantage at 100+ layers
"""

import os
import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass
import matplotlib.pyplot as plt


# =============================================================================
# Create Structured Learnable Data (Shakespeare-like)
# =============================================================================

def create_learnable_data(size=500000):
    """
    Create structured data that has learnable patterns.
    Uses repeating patterns and character-level sequences.
    """
    # Simple patterns that can be learned
    patterns = [
        "the quick brown fox jumps over the lazy dog ",
        "to be or not to be that is the question ",
        "all that glitters is not gold ",
        "a journey of a thousand miles begins with a single step ",
        "actions speak louder than words ",
        "knowledge is power ",
        "time flies when you are having fun ",
        "practice makes perfect ",
    ]
    
    # Build vocabulary
    all_chars = set()
    for p in patterns:
        all_chars.update(p)
    all_chars = sorted(list(all_chars))
    char_to_idx = {c: i for i, c in enumerate(all_chars)}
    vocab_size = len(all_chars)
    
    # Generate data by randomly sampling patterns
    data = []
    while len(data) < size:
        pattern = patterns[np.random.randint(len(patterns))]
        for c in pattern:
            data.append(char_to_idx[c])
    
    data = np.array(data[:size], dtype=np.uint8)
    return data, vocab_size, char_to_idx


# =============================================================================
# Model Components
# =============================================================================

@dataclass
class Config:
    dim: int = 128
    n_layers: int = 4
    vocab_size: int = 32
    block_size: int = 64
    n_head: int = 4


class CausalSelfAttention(nn.Module):
    """Simple causal self-attention."""
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.dim = config.dim
        self.head_dim = config.dim // config.n_head
        
        self.qkv = nn.Linear(config.dim, 3 * config.dim)
        self.proj = nn.Linear(config.dim, config.dim)
        
        # Causal mask
        self.register_buffer("mask", torch.tril(torch.ones(config.block_size, config.block_size))
                            .view(1, 1, config.block_size, config.block_size))
    
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
        self.fc1 = nn.Linear(dim, dim * 4)
        self.fc2 = nn.Linear(dim * 4, dim)
    
    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


class BaselineBlock(nn.Module):
    """Standard transformer block."""
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.dim)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.dim)
        self.mlp = MLP(config.dim)
    
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class DDLBlock(nn.Module):
    """DDL block with Householder reflection replacing residual."""
    def __init__(self, config):
        super().__init__()
        self.dim = config.dim
        self.ln1 = nn.LayerNorm(config.dim)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.dim)
        self.mlp = MLP(config.dim)
        
        # Householder for attention residual
        self.k_attn = nn.Linear(config.dim, config.dim)
        self.beta_attn = nn.Linear(config.dim, 1)
        
        # Householder for MLP residual
        self.k_mlp = nn.Linear(config.dim, config.dim)
        self.beta_mlp = nn.Linear(config.dim, 1)
    
    def householder(self, x, k_proj, beta_proj):
        B, T, D = x.shape
        x_pool = x.mean(dim=1)
        k = F.normalize(k_proj(x_pool), dim=-1).unsqueeze(1)  # (B, 1, D)
        beta = 2 * torch.sigmoid(beta_proj(x_pool)).unsqueeze(-1)  # (B, 1, 1)
        dot = (x * k).sum(dim=-1, keepdim=True)
        return x - beta * dot * k
    
    def forward(self, x):
        # Attention with Householder residual
        x_h = self.householder(x, self.k_attn, self.beta_attn)
        x = x_h + self.attn(self.ln1(x_h))
        
        # MLP with Householder residual
        x_h = self.householder(x, self.k_mlp, self.beta_mlp)
        x = x_h + self.mlp(self.ln2(x_h))
        return x


class DDCBlock(nn.Module):
    """DDC block with Cayley rotation (unconditional orthogonality)."""
    def __init__(self, config, n_streams=4):
        super().__init__()
        self.dim = config.dim
        self.n_streams = n_streams
        self.d_stream = config.dim // n_streams
        
        self.ln1 = nn.LayerNorm(config.dim)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.dim)
        self.mlp = MLP(config.dim)
        
        # Cayley for attention residual
        self.u_attn = nn.Linear(config.dim, n_streams)
        self.v_attn = nn.Linear(config.dim, n_streams)
        self.beta_attn = nn.Linear(config.dim, 1)
        self.alpha_attn = nn.Linear(config.dim, 1)
        
        # Cayley for MLP residual
        self.u_mlp = nn.Linear(config.dim, n_streams)
        self.v_mlp = nn.Linear(config.dim, n_streams)
        self.beta_mlp = nn.Linear(config.dim, 1)
        self.alpha_mlp = nn.Linear(config.dim, 1)
        
        self.register_buffer('I', torch.eye(n_streams))
        
        # Initialize for small rotations
        for proj in [self.u_attn, self.v_attn, self.u_mlp, self.v_mlp]:
            nn.init.normal_(proj.weight, std=0.01)
        for proj in [self.alpha_attn, self.alpha_mlp]:
            nn.init.zeros_(proj.weight)
            nn.init.constant_(proj.bias, -1.0)
    
    def cayley_rotate(self, x, u_proj, v_proj, beta_proj, alpha_proj):
        B, T, D = x.shape
        x_pool = x.mean(dim=1)
        
        u = u_proj(x_pool)
        v = v_proj(x_pool)
        beta = F.softplus(beta_proj(x_pool)).unsqueeze(-1)
        
        # Skew-symmetric A = uv^T - vu^T
        A = torch.einsum('bi,bj->bij', u, v) - torch.einsum('bi,bj->bij', v, u)
        
        # Cayley transform
        M = (beta / 2) * A
        I = self.I.unsqueeze(0).expand(B, -1, -1)
        Q = torch.linalg.solve(I + M, I - M)  # GUARANTEED orthogonal
        
        # Apply rotation
        x_streams = x.view(B, T, self.n_streams, self.d_stream)
        x_rotated = torch.einsum('bij,btjd->btid', Q, x_streams).reshape(B, T, D)
        
        # Residual gate
        alpha = torch.sigmoid(alpha_proj(x_pool)).unsqueeze(-1)
        return alpha * x_rotated + (1 - alpha) * x
    
    def forward(self, x):
        # Attention with Cayley residual
        x_c = self.cayley_rotate(x, self.u_attn, self.v_attn, self.beta_attn, self.alpha_attn)
        x = x_c + self.attn(self.ln1(x_c))
        
        # MLP with Cayley residual
        x_c = self.cayley_rotate(x, self.u_mlp, self.v_mlp, self.beta_mlp, self.alpha_mlp)
        x = x_c + self.mlp(self.ln2(x_c))
        return x


class DeepTransformer(nn.Module):
    """Scalable transformer for depth testing."""
    def __init__(self, config, block_type='baseline'):
        super().__init__()
        self.config = config
        
        self.embed = nn.Embedding(config.vocab_size, config.dim)
        self.pos_embed = nn.Embedding(config.block_size, config.dim)
        
        if block_type == 'baseline':
            self.blocks = nn.ModuleList([BaselineBlock(config) for _ in range(config.n_layers)])
        elif block_type == 'ddl':
            self.blocks = nn.ModuleList([DDLBlock(config) for _ in range(config.n_layers)])
        elif block_type == 'ddc':
            self.blocks = nn.ModuleList([DDCBlock(config) for _ in range(config.n_layers)])
        
        self.ln_f = nn.LayerNorm(config.dim)
        self.head = nn.Linear(config.dim, config.vocab_size, bias=False)
        self.head.weight = self.embed.weight
        
        self.apply(self._init_weights)
        
        # Scale down residual projections for stability
        for block in self.blocks:
            if hasattr(block, 'attn') and hasattr(block.attn, 'proj'):
                nn.init.normal_(block.attn.proj.weight, std=0.02 / math.sqrt(2 * config.n_layers))
            if hasattr(block, 'mlp') and hasattr(block.mlp, 'fc2'):
                nn.init.normal_(block.mlp.fc2.weight, std=0.02 / math.sqrt(2 * config.n_layers))
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)
    
    def forward(self, x, targets=None):
        B, T = x.shape
        
        tok_emb = self.embed(x)
        pos_emb = self.pos_embed(torch.arange(T, device=x.device))
        h = tok_emb + pos_emb
        
        for block in self.blocks:
            h = block(h)
        
        h = self.ln_f(h)
        logits = self.head(h)
        
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            return logits, loss
        return logits, None


# =============================================================================
# Training
# =============================================================================

def train_and_evaluate(config, block_type, train_data, val_data, 
                       max_iters=3000, batch_size=32, lr=1e-3, device='cuda'):
    """Train model and return comprehensive metrics."""
    
    model = DeepTransformer(config, block_type).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    
    # Warmup + cosine decay
    def get_lr(it):
        warmup = 200
        if it < warmup:
            return lr * it / warmup
        decay_ratio = (it - warmup) / (max_iters - warmup)
        return lr * 0.1 + 0.9 * lr * (1 + math.cos(math.pi * decay_ratio)) / 2
    
    n_params = sum(p.numel() for p in model.parameters())
    block_size = config.block_size
    
    def get_batch(data):
        ix = torch.randint(len(data) - block_size, (batch_size,))
        x = torch.stack([torch.from_numpy(data[i:i+block_size].astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy(data[i+1:i+1+block_size].astype(np.int64)) for i in ix])
        return x.to(device), y.to(device)
    
    # Metrics
    train_losses = []
    val_losses = []
    grad_norms = []
    learning_rates = []
    
    best_val_loss = float('inf')
    converged = True
    failed_at = None
    
    t0 = time.time()
    
    for it in range(max_iters):
        # Update learning rate
        current_lr = get_lr(it)
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr
        learning_rates.append(current_lr)
        
        # Training step
        model.train()
        x, y = get_batch(train_data)
        _, loss = model(x, y)
        
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
        if torch.isnan(loss) or torch.isinf(loss) or total_norm > 1e6:
            converged = False
            failed_at = it
            break
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        train_losses.append(loss.item())
        
        # Validation every 300 iters
        if it % 300 == 0:
            model.eval()
            with torch.no_grad():
                val_loss_sum = 0
                for _ in range(20):
                    x, y = get_batch(val_data)
                    _, vloss = model(x, y)
                    val_loss_sum += vloss.item()
                val_loss = val_loss_sum / 20
                val_losses.append((it, val_loss))
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
            
            # Progress
            print(f"    iter {it:4d}: train={loss.item():.4f}, val={val_loss:.4f}, "
                  f"grad={total_norm:.2f}, lr={current_lr:.1e}")
    
    train_time = time.time() - t0
    
    return {
        'n_params': n_params,
        'converged': converged,
        'failed_at': failed_at,
        'best_val_loss': best_val_loss,
        'final_train_loss': train_losses[-1] if train_losses else float('inf'),
        'train_losses': train_losses,
        'val_losses': val_losses,
        'grad_norms': grad_norms,
        'mean_grad_norm': np.mean(grad_norms[-500:]) if len(grad_norms) > 500 else np.mean(grad_norms),
        'max_grad_norm': np.max(grad_norms) if grad_norms else float('inf'),
        'train_time': train_time,
    }


# =============================================================================
# Main Experiment
# =============================================================================

def run_experiment():
    print("=" * 70)
    print("DEEP NETWORK COMPARISON: DDC vs DDL vs Baseline")
    print("Real Learnable Task + Very Deep Networks")
    print("=" * 70)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # Create learnable data
    print("\nCreating structured learnable data...")
    train_data, vocab_size, _ = create_learnable_data(500000)
    val_data, _, _ = create_learnable_data(50000)
    print(f"  Vocab size: {vocab_size}")
    print(f"  Train size: {len(train_data)}")
    print(f"  Val size: {len(val_data)}")
    
    # Test configurations
    depths = [32, 64, 128]
    block_types = ['baseline', 'ddl', 'ddc']
    
    results = {bt: {} for bt in block_types}
    
    print(f"\nTesting depths: {depths}")
    print(f"Block types: {block_types}")
    
    for depth in depths:
        print(f"\n{'='*70}")
        print(f"DEPTH = {depth} LAYERS")
        print(f"{'='*70}")
        
        config = Config(
            dim=128,
            n_layers=depth,
            vocab_size=vocab_size,
            block_size=64,
            n_head=4,
        )
        
        for block_type in block_types:
            print(f"\n  Training {block_type} ({depth} layers)...")
            
            try:
                metrics = train_and_evaluate(
                    config, block_type, train_data, val_data,
                    max_iters=2000, batch_size=32, lr=3e-4, device=device
                )
                results[block_type][depth] = metrics
                
                status = "✓" if metrics['converged'] else f"✗ FAILED at iter {metrics['failed_at']}"
                print(f"\n  {block_type}: {status}")
                print(f"    Best val loss: {metrics['best_val_loss']:.4f}")
                print(f"    Mean grad norm: {metrics['mean_grad_norm']:.2f}")
                print(f"    Max grad norm: {metrics['max_grad_norm']:.2f}")
                print(f"    Train time: {metrics['train_time']:.1f}s")
                
            except Exception as e:
                print(f"\n  {block_type}: ERROR - {e}")
                results[block_type][depth] = {
                    'converged': False,
                    'best_val_loss': float('inf'),
                    'mean_grad_norm': float('inf'),
                    'error': str(e)
                }
    
    # Summary
    print("\n" + "=" * 70)
    print("FINAL RESULTS SUMMARY")
    print("=" * 70)
    
    print(f"\n{'Depth':<8} {'Metric':<15}", end="")
    for bt in block_types:
        print(f"{bt:<15}", end="")
    print()
    print("-" * 70)
    
    for depth in depths:
        # Val loss row
        print(f"{depth:<8} {'Val Loss':<15}", end="")
        for bt in block_types:
            if depth in results[bt] and results[bt][depth].get('converged', False):
                print(f"{results[bt][depth]['best_val_loss']:<15.4f}", end="")
            else:
                print(f"{'FAILED':<15}", end="")
        print()
        
        # Grad norm row
        print(f"{'':8} {'Grad Norm':<15}", end="")
        for bt in block_types:
            if depth in results[bt] and results[bt][depth].get('converged', False):
                print(f"{results[bt][depth]['mean_grad_norm']:<15.2f}", end="")
            else:
                print(f"{'-':<15}", end="")
        print()
        print()
    
    # Analysis
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)
    
    for bt in block_types:
        max_depth_converged = max([d for d in depths if d in results[bt] and results[bt][d].get('converged', False)], default=0)
        print(f"\n{bt}:")
        print(f"  Max depth converged: {max_depth_converged}")
        if max_depth_converged > 0:
            print(f"  Best loss at max depth: {results[bt][max_depth_converged]['best_val_loss']:.4f}")
            print(f"  Grad norm at max depth: {results[bt][max_depth_converged]['mean_grad_norm']:.2f}")
    
    # Determine winner at each depth
    print("\n" + "=" * 70)
    print("WINNER AT EACH DEPTH")
    print("=" * 70)
    
    for depth in depths:
        converged_models = [(bt, results[bt][depth]['best_val_loss']) 
                           for bt in block_types 
                           if depth in results[bt] and results[bt][depth].get('converged', False)]
        if converged_models:
            winner = min(converged_models, key=lambda x: x[1])
            print(f"  Depth {depth}: {winner[0]} (loss={winner[1]:.4f})")
        else:
            print(f"  Depth {depth}: No model converged")
    
    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Val loss vs depth
    ax = axes[0, 0]
    for bt in block_types:
        losses = []
        valid_depths = []
        for d in depths:
            if d in results[bt] and results[bt][d].get('converged', False):
                losses.append(results[bt][d]['best_val_loss'])
                valid_depths.append(d)
        if losses:
            ax.plot(valid_depths, losses, 'o-', label=bt, markersize=10, linewidth=2)
    ax.set_xlabel('Depth (layers)', fontsize=12)
    ax.set_ylabel('Best Validation Loss', fontsize=12)
    ax.set_title('Validation Loss vs Depth', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Gradient norm vs depth
    ax = axes[0, 1]
    for bt in block_types:
        norms = []
        valid_depths = []
        for d in depths:
            if d in results[bt] and results[bt][d].get('converged', False):
                norms.append(results[bt][d]['mean_grad_norm'])
                valid_depths.append(d)
        if norms:
            ax.plot(valid_depths, norms, 'o-', label=bt, markersize=10, linewidth=2)
    ax.set_xlabel('Depth (layers)', fontsize=12)
    ax.set_ylabel('Mean Gradient Norm', fontsize=12)
    ax.set_title('Gradient Stability vs Depth', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Training curves at deepest depth
    ax = axes[1, 0]
    max_depth = max(depths)
    for bt in block_types:
        if max_depth in results[bt] and results[bt][max_depth].get('converged', False):
            losses = results[bt][max_depth]['train_losses']
            # Smooth
            window = 50
            smoothed = np.convolve(losses, np.ones(window)/window, mode='valid')
            ax.plot(smoothed, label=bt, alpha=0.8, linewidth=1.5)
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Training Loss', fontsize=12)
    ax.set_title(f'Training Curves (depth={max_depth})', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Gradient norm over training
    ax = axes[1, 1]
    for bt in block_types:
        if max_depth in results[bt] and results[bt][max_depth].get('converged', False):
            norms = results[bt][max_depth]['grad_norms']
            window = 50
            smoothed = np.convolve(norms, np.ones(window)/window, mode='valid')
            ax.plot(smoothed, label=bt, alpha=0.8, linewidth=1.5)
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Gradient Norm', fontsize=12)
    ax.set_title(f'Gradient Norm During Training (depth={max_depth})', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('deep_network_comparison.png', dpi=150)
    print(f"\nPlot saved to: deep_network_comparison.png")
    
    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    print("""
This experiment tests DDC's theoretical advantage with:
1. Real learnable task (character-level language modeling)
2. Very deep networks (32, 64, 128 layers)
3. Comprehensive metrics (loss, gradient stability)

DDC's unconditional orthogonality should provide:
- More stable gradients at extreme depths
- Better or equal validation loss
- More consistent training dynamics
""")


if __name__ == '__main__':
    run_experiment()
