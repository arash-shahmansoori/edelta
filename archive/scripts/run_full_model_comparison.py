"""
FULL MODEL COMPARISON: DDC vs DDL vs Baseline vs Hybrid vs Gearbox

Uses the ACTUAL full GPT implementations from the codebase:
- proposed_model_ddl.py (Pure DDL with Householder reflection)
- proposed_model_ddc.py (Data-Dependent Cayley - unconditional orthogonality)
- proposed_model_hybrid.py (E∆-Hybrid: Cayley + Householder)
- proposed_model_gearbox.py (E∆-Gearbox: DDL + Cayley switchable)
- model.py (Baseline standard Transformer)

Tests on character-level language modeling with varying depths.
"""

import os
import sys
import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass
import matplotlib.pyplot as plt

# Import the actual models from the codebase
sys.path.insert(0, '/root/edelta')


def create_shakespeare_like_data(size=500000):
    """Create structured character-level data for LM training."""
    patterns = [
        "the quick brown fox jumps over the lazy dog ",
        "to be or not to be that is the question ",
        "all that glitters is not gold ",
        "a journey of a thousand miles begins with a single step ",
        "actions speak louder than words ",
        "knowledge is power ",
        "time flies when you are having fun ",
        "practice makes perfect ",
        "better late than never ",
        "the pen is mightier than the sword ",
    ]
    
    all_chars = set()
    for p in patterns:
        all_chars.update(p)
    all_chars = sorted(list(all_chars))
    char_to_idx = {c: i for i, c in enumerate(all_chars)}
    vocab_size = len(all_chars)
    
    data = []
    while len(data) < size:
        pattern = patterns[np.random.randint(len(patterns))]
        for c in pattern:
            data.append(char_to_idx[c])
    
    return np.array(data[:size], dtype=np.uint8), vocab_size


def train_model(model, train_data, val_data, config, max_iters=2000, batch_size=32, 
                lr=3e-4, device='cuda', model_name='model'):
    """Train a model and return metrics."""
    
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    
    block_size = config['block_size']
    
    def get_batch(data):
        ix = torch.randint(len(data) - block_size, (batch_size,))
        x = torch.stack([torch.from_numpy(data[i:i+block_size].astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy(data[i+1:i+1+block_size].astype(np.int64)) for i in ix])
        return x.to(device), y.to(device)
    
    def get_lr(it):
        warmup = 200
        if it < warmup:
            return lr * it / warmup
        decay_ratio = (it - warmup) / (max_iters - warmup)
        return lr * 0.1 + 0.9 * lr * (1 + math.cos(math.pi * decay_ratio)) / 2
    
    # Metrics
    train_losses = []
    val_losses = []
    grad_norms = []
    
    best_val_loss = float('inf')
    converged = True
    
    t0 = time.time()
    
    for it in range(max_iters):
        # Update LR
        current_lr = get_lr(it)
        for pg in optimizer.param_groups:
            pg['lr'] = current_lr
        
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
            print(f"    {model_name}: EXPLODED at iter {it}")
            break
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        train_losses.append(loss.item())
        
        # Validation
        if it % 200 == 0:
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
            
            print(f"    {model_name} iter {it:4d}: train={loss.item():.4f}, val={val_loss:.4f}, grad={total_norm:.2f}")
    
    train_time = time.time() - t0
    
    return {
        'converged': converged,
        'best_val_loss': best_val_loss,
        'final_train_loss': train_losses[-1] if train_losses else float('inf'),
        'mean_grad_norm': np.mean(grad_norms[-500:]) if len(grad_norms) > 500 else np.mean(grad_norms),
        'max_grad_norm': np.max(grad_norms) if grad_norms else float('inf'),
        'train_time': train_time,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'grad_norms': grad_norms,
    }


def run_comparison():
    print("=" * 80)
    print("FULL MODEL COMPARISON: Actual GPT Implementations")
    print("DDC vs DDL vs Baseline vs Hybrid vs Gearbox")
    print("=" * 80)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # Create data
    print("\nCreating character-level data...")
    train_data, vocab_size = create_shakespeare_like_data(500000)
    val_data, _ = create_shakespeare_like_data(50000)
    print(f"  Vocab size: {vocab_size}")
    
    # Test configurations: varying depth
    depths = [16, 32, 48]  # Moderate to deep networks
    
    results = {}
    
    for depth in depths:
        print(f"\n{'='*80}")
        print(f"DEPTH = {depth} LAYERS")
        print(f"{'='*80}")
        
        config_dict = {
            'block_size': 64,
            'vocab_size': vocab_size,
            'n_layer': depth,
            'n_head': 4,
            'n_embd': 128,
            'dropout': 0.0,
            'bias': False,
        }
        
        results[depth] = {}
        
        # ============================================================
        # 1. BASELINE (Standard Transformer)
        # ============================================================
        print(f"\n  [1/5] Training BASELINE (depth={depth})...")
        try:
            from model import GPTConfig as BaselineConfig, GPT as BaselineGPT
            
            baseline_config = BaselineConfig(
                block_size=config_dict['block_size'],
                vocab_size=config_dict['vocab_size'],
                n_layer=config_dict['n_layer'],
                n_head=config_dict['n_head'],
                n_embd=config_dict['n_embd'],
                dropout=config_dict['dropout'],
                bias=config_dict['bias'],
            )
            baseline_model = BaselineGPT(baseline_config)
            
            results[depth]['baseline'] = train_model(
                baseline_model, train_data, val_data, config_dict,
                max_iters=1500, batch_size=32, lr=3e-4, device=device,
                model_name='Baseline'
            )
            del baseline_model
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"    Baseline ERROR: {e}")
            results[depth]['baseline'] = {'converged': False, 'best_val_loss': float('inf'), 'error': str(e)}
        
        # ============================================================
        # 2. DDL (Deep Delta Learning - Householder)
        # ============================================================
        print(f"\n  [2/5] Training DDL (depth={depth})...")
        try:
            from proposed_model_ddl import GPTConfig as DDLConfig, GPT as DDLGPT
            
            ddl_config = DDLConfig(
                block_size=config_dict['block_size'],
                vocab_size=config_dict['vocab_size'],
                n_layer=config_dict['n_layer'],
                n_head=config_dict['n_head'],
                n_embd=config_dict['n_embd'],
                dropout=config_dict['dropout'],
                bias=config_dict['bias'],
            )
            ddl_model = DDLGPT(ddl_config)
            
            results[depth]['ddl'] = train_model(
                ddl_model, train_data, val_data, config_dict,
                max_iters=1500, batch_size=32, lr=3e-4, device=device,
                model_name='DDL'
            )
            del ddl_model
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"    DDL ERROR: {e}")
            results[depth]['ddl'] = {'converged': False, 'best_val_loss': float('inf'), 'error': str(e)}
        
        # ============================================================
        # 3. DDC (Data-Dependent Cayley - Unconditional Orthogonality)
        # ============================================================
        print(f"\n  [3/5] Training DDC (depth={depth})...")
        try:
            from proposed_model_ddc import GPTConfig as DDCConfig, GPT as DDCGPT
            
            ddc_config = DDCConfig(
                block_size=config_dict['block_size'],
                vocab_size=config_dict['vocab_size'],
                n_layer=config_dict['n_layer'],
                n_head=config_dict['n_head'],
                n_embd=config_dict['n_embd'],
                dropout=config_dict['dropout'],
                bias=config_dict['bias'],
                n_streams=4,
            )
            ddc_model = DDCGPT(ddc_config)
            
            results[depth]['ddc'] = train_model(
                ddc_model, train_data, val_data, config_dict,
                max_iters=1500, batch_size=32, lr=3e-4, device=device,
                model_name='DDC'
            )
            del ddc_model
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"    DDC ERROR: {e}")
            results[depth]['ddc'] = {'converged': False, 'best_val_loss': float('inf'), 'error': str(e)}
        
        # ============================================================
        # 4. E∆-Hybrid (Cayley + Householder)
        # ============================================================
        print(f"\n  [4/5] Training E∆-Hybrid (depth={depth})...")
        try:
            from proposed_model_hybrid import GPTConfig as HybridConfig, GPT as HybridGPT
            
            hybrid_config = HybridConfig(
                block_size=config_dict['block_size'],
                vocab_size=config_dict['vocab_size'],
                n_layer=config_dict['n_layer'],
                n_head=config_dict['n_head'],
                n_embd=config_dict['n_embd'],
                dropout=config_dict['dropout'],
                bias=config_dict['bias'],
            )
            hybrid_model = HybridGPT(hybrid_config)
            
            results[depth]['hybrid'] = train_model(
                hybrid_model, train_data, val_data, config_dict,
                max_iters=1500, batch_size=32, lr=3e-4, device=device,
                model_name='Hybrid'
            )
            del hybrid_model
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"    Hybrid ERROR: {e}")
            results[depth]['hybrid'] = {'converged': False, 'best_val_loss': float('inf'), 'error': str(e)}
        
        # ============================================================
        # 5. E∆-Gearbox (DDL + Cayley switchable)
        # ============================================================
        print(f"\n  [5/5] Training E∆-Gearbox (depth={depth})...")
        try:
            from proposed_model_gearbox import GPTConfig as GearboxConfig, GPT as GearboxGPT
            
            gearbox_config = GearboxConfig(
                block_size=config_dict['block_size'],
                vocab_size=config_dict['vocab_size'],
                n_layer=config_dict['n_layer'],
                n_head=config_dict['n_head'],
                n_embd=config_dict['n_embd'],
                dropout=config_dict['dropout'],
                bias=config_dict['bias'],
                n_streams=4,
            )
            gearbox_model = GearboxGPT(gearbox_config)
            
            results[depth]['gearbox'] = train_model(
                gearbox_model, train_data, val_data, config_dict,
                max_iters=1500, batch_size=32, lr=3e-4, device=device,
                model_name='Gearbox'
            )
            del gearbox_model
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"    Gearbox ERROR: {e}")
            results[depth]['gearbox'] = {'converged': False, 'best_val_loss': float('inf'), 'error': str(e)}
    
    # ============================================================
    # SUMMARY
    # ============================================================
    print("\n" + "=" * 80)
    print("FINAL RESULTS SUMMARY")
    print("=" * 80)
    
    models = ['baseline', 'ddl', 'ddc', 'hybrid', 'gearbox']
    
    # Table header
    print(f"\n{'Depth':<8}", end="")
    for m in models:
        print(f"{m:<12}", end="")
    print()
    print("-" * 70)
    
    for depth in depths:
        print(f"{depth:<8}", end="")
        for m in models:
            if m in results[depth]:
                r = results[depth][m]
                if r.get('converged', False):
                    print(f"{r['best_val_loss']:<12.4f}", end="")
                else:
                    print(f"{'FAILED':<12}", end="")
            else:
                print(f"{'-':<12}", end="")
        print()
    
    # Gradient norms
    print(f"\n{'Depth':<8}", end="")
    for m in models:
        print(f"{m + '_grad':<12}", end="")
    print()
    print("-" * 70)
    
    for depth in depths:
        print(f"{depth:<8}", end="")
        for m in models:
            if m in results[depth]:
                r = results[depth][m]
                if r.get('converged', False):
                    print(f"{r['mean_grad_norm']:<12.2f}", end="")
                else:
                    print(f"{'-':<12}", end="")
            else:
                print(f"{'-':<12}", end="")
        print()
    
    # Winner analysis
    print("\n" + "=" * 80)
    print("WINNER AT EACH DEPTH (lowest val loss)")
    print("=" * 80)
    
    for depth in depths:
        converged = [(m, results[depth][m]['best_val_loss']) 
                     for m in models 
                     if m in results[depth] and results[depth][m].get('converged', False)]
        if converged:
            winner = min(converged, key=lambda x: x[1])
            print(f"  Depth {depth}: {winner[0].upper()} (loss={winner[1]:.4f})")
        else:
            print(f"  Depth {depth}: No model converged")
    
    # Stability analysis
    print("\n" + "=" * 80)
    print("STABILITY ANALYSIS (gradient norms)")
    print("=" * 80)
    
    for depth in depths:
        print(f"\n  Depth {depth}:")
        for m in models:
            if m in results[depth] and results[depth][m].get('converged', False):
                r = results[depth][m]
                print(f"    {m:<10}: mean_grad={r['mean_grad_norm']:.2f}, max_grad={r['max_grad_norm']:.2f}")
    
    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    colors = {'baseline': 'gray', 'ddl': 'red', 'ddc': 'blue', 'hybrid': 'green', 'gearbox': 'purple'}
    
    # Plot 1: Val loss vs depth
    ax = axes[0, 0]
    for m in models:
        losses = []
        valid_depths = []
        for d in depths:
            if m in results[d] and results[d][m].get('converged', False):
                losses.append(results[d][m]['best_val_loss'])
                valid_depths.append(d)
        if losses:
            ax.plot(valid_depths, losses, 'o-', label=m.upper(), color=colors[m], 
                   markersize=10, linewidth=2)
    ax.set_xlabel('Depth (layers)', fontsize=12)
    ax.set_ylabel('Best Validation Loss', fontsize=12)
    ax.set_title('Validation Loss vs Depth', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Gradient norm vs depth
    ax = axes[0, 1]
    for m in models:
        norms = []
        valid_depths = []
        for d in depths:
            if m in results[d] and results[d][m].get('converged', False):
                norms.append(results[d][m]['mean_grad_norm'])
                valid_depths.append(d)
        if norms:
            ax.plot(valid_depths, norms, 'o-', label=m.upper(), color=colors[m],
                   markersize=10, linewidth=2)
    ax.set_xlabel('Depth (layers)', fontsize=12)
    ax.set_ylabel('Mean Gradient Norm', fontsize=12)
    ax.set_title('Gradient Stability vs Depth', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Training curves at max depth
    ax = axes[1, 0]
    max_depth = max(depths)
    for m in models:
        if m in results[max_depth] and results[max_depth][m].get('converged', False):
            losses = results[max_depth][m]['train_losses']
            window = 30
            if len(losses) > window:
                smoothed = np.convolve(losses, np.ones(window)/window, mode='valid')
                ax.plot(smoothed, label=m.upper(), color=colors[m], alpha=0.8, linewidth=1.5)
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Training Loss', fontsize=12)
    ax.set_title(f'Training Curves (depth={max_depth})', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Gradient norms during training
    ax = axes[1, 1]
    for m in models:
        if m in results[max_depth] and results[max_depth][m].get('converged', False):
            norms = results[max_depth][m]['grad_norms']
            window = 30
            if len(norms) > window:
                smoothed = np.convolve(norms, np.ones(window)/window, mode='valid')
                ax.plot(smoothed, label=m.upper(), color=colors[m], alpha=0.8, linewidth=1.5)
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Gradient Norm', fontsize=12)
    ax.set_title(f'Gradient Norm During Training (depth={max_depth})', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('full_model_comparison.png', dpi=150)
    print(f"\nPlot saved to: full_model_comparison.png")
    
    # Key insights
    print("\n" + "=" * 80)
    print("KEY INSIGHTS")
    print("=" * 80)
    print("""
This experiment compares the ACTUAL full GPT implementations:

1. BASELINE: Standard Transformer with additive residual
2. DDL: Householder reflection (can negate, NOT always orthogonal)
3. DDC: Data-Dependent Cayley (UNCONDITIONALLY orthogonal)
4. HYBRID: Cayley rotation + Householder reflection + learned gate
5. GEARBOX: Linear DDL + exact Cayley with learned switching

Expected DDC advantages:
- More stable gradients at deep networks (orthogonality guarantee)
- Better or equal validation loss
- No singularity issues (unlike DDL at β=1)

DDL limitations:
- Only orthogonal at β=0 or β=2
- Singular at β=1
- May accumulate norm drift in deep networks
""")


if __name__ == '__main__':
    run_comparison()
