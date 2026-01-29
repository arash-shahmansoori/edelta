#!/usr/bin/env python3
"""
Stress Benchmark: Designed to Highlight Geometric Model Advantages

This benchmark uses conditions where geometric inductive bias matters most:
1. DEEP networks (12 layers) - errors accumulate in non-geometric models
2. FEW samples (200) - geometric structure helps generalization
3. MULTIPLE corrections per sequence - tests repeated belief revision
4. LONGER sequences (512 tokens) - tests long-horizon stability

Expected Results:
- GPT2: Degrades under stress (error accumulation, overfitting)
- DDL: Better stability via rank-1 perturbation with β control
- E∆-MHC-Geo: Best - full O(n) coverage with adaptive gating

Reference: DDL paper (arXiv:2601.00417) Section 4.3 on depth scaling
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
# MULTI-SHIFT DATASET (Harder than single shift)
# =============================================================================

def generate_multi_shift_dataset(
    n_samples: int = 500,
    seq_length: int = 512,
    n_shifts: int = 3,  # Multiple correction points!
    dim: int = 64,
    noise_std: float = 0.1,
    device: str = 'cuda'
):
    """
    Generate sequences with MULTIPLE correction points.
    
    Structure per sequence:
    - Segment 1: belief_1, then SHIFT to -belief_1
    - Segment 2: belief_2, then SHIFT to -belief_2
    - Segment 3: belief_3, then SHIFT to -belief_3
    
    This tests:
    - Can model handle REPEATED belief revisions?
    - Does accuracy degrade after multiple shifts?
    - Geometric models should maintain accuracy; GPT2 should degrade.
    """
    
    segment_length = seq_length // (n_shifts * 2)  # Each shift has before/after
    
    train_x = []
    train_y = []
    val_x = []
    val_y = []
    
    # Track shift positions for evaluation
    shift_positions = []
    for i in range(n_shifts):
        shift_pos = segment_length * (2 * i + 1)  # Position of each shift
        shift_positions.append(shift_pos)
    
    def make_sequence():
        x_seq = []
        y_seq = []
        
        for shift_idx in range(n_shifts):
            # Generate belief for this segment
            belief = torch.randn(dim, device=device)
            belief = belief / belief.norm() * 2.0  # Normalize
            
            # Before shift: maintain belief
            for t in range(segment_length):
                noise = torch.randn(dim, device=device) * noise_std
                x_seq.append(belief + noise)
                y_seq.append(belief.clone())
            
            # After shift: FLIP to -belief
            for t in range(segment_length):
                noise = torch.randn(dim, device=device) * noise_std
                # Add shift signal at first position of flipped segment
                if t == 0:
                    signal = torch.zeros(dim, device=device)
                    signal[0] = 5.0  # Shift trigger
                    x_seq.append(-belief + noise + signal)
                else:
                    x_seq.append(-belief + noise)
                y_seq.append(-belief.clone())
        
        # Pad if needed
        while len(x_seq) < seq_length:
            x_seq.append(torch.zeros(dim, device=device))
            y_seq.append(torch.zeros(dim, device=device))
        
        return torch.stack(x_seq[:seq_length]), torch.stack(y_seq[:seq_length])
    
    # Generate training data (FEW samples to test generalization)
    n_train = int(n_samples * 0.8)
    n_val = n_samples - n_train
    
    for _ in range(n_train):
        x, y = make_sequence()
        train_x.append(x)
        train_y.append(y)
    
    for _ in range(n_val):
        x, y = make_sequence()
        val_x.append(x)
        val_y.append(y)
    
    return {
        'train_x': torch.stack(train_x),
        'train_y': torch.stack(train_y),
        'val_x': torch.stack(val_x),
        'val_y': torch.stack(val_y),
        'shift_positions': shift_positions,
        'n_shifts': n_shifts,
        'segment_length': segment_length,
    }


# =============================================================================
# MODEL CREATION (DEEP networks)
# =============================================================================

def create_model(model_name: str, n_layer: int = 12, n_embd: int = 128, device: str = 'cuda'):
    """Create model with configurable depth."""
    
    base_config = {
        'vocab_size': 256,
        'block_size': 512,
        'n_layer': n_layer,  # DEEP!
        'n_head': 4,
        'n_embd': n_embd,
        'dropout': 0.1,
        'bias': True,
    }
    
    if model_name == 'gpt2':
        from model import GPT, GPTConfig
        config = GPTConfig(**base_config)
        model = GPT(config)
        desc = f'GPT2 (baseline, {n_layer}L)'
        
    elif model_name == 'ddl':
        from proposed_model_ddl import GPT, GPTConfig
        config = GPTConfig(**base_config)
        model = GPT(config)
        desc = f'DDL ({n_layer}L)'
        
    elif model_name == 'mhc':
        from proposed_model_mhc_real import GPT, GPTConfig
        config = GPTConfig(**base_config)
        model = GPT(config)
        desc = f'mHC ({n_layer}L)'
        
    elif model_name == 'proposed':
        from proposed_model_hybrid import GPT, GPTConfig
        config = GPTConfig(**base_config)
        model = GPT(config)
        desc = f'E∆-MHC-Geo ({n_layer}L)'
    
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    return model.to(device), desc


# =============================================================================
# CONTINUOUS MODEL WRAPPER
# =============================================================================

class ContinuousWrapper(nn.Module):
    """Wrap discrete token model for continuous vectors."""
    
    def __init__(self, model, input_dim: int, output_dim: int):
        super().__init__()
        self.model = model
        self.input_proj = nn.Linear(input_dim, model.config.n_embd)
        self.output_proj = nn.Linear(model.config.n_embd, output_dim)
        
    def forward(self, x, targets=None):
        # x: [B, T, D] continuous vectors
        B, T, D = x.shape
        
        # Project to embedding space
        h = self.input_proj(x)
        
        # Run through transformer (bypass embedding lookup)
        # Access internal transformer
        if hasattr(self.model, 'transformer'):
            for block in self.model.transformer.h:
                h = block(h)
            h = self.model.transformer.ln_f(h)
        else:
            # Fallback for different architectures
            for block in self.model.blocks:
                h = block(h)
            h = self.model.ln_f(h)
        
        # Project to output
        out = self.output_proj(h)
        
        if targets is not None:
            loss = F.mse_loss(out, targets)
            return out, loss
        
        return out, None


# =============================================================================
# TRAINING
# =============================================================================

def train_and_evaluate(
    model_name: str,
    data: dict,
    n_layer: int = 12,
    max_iters: int = 1500,
    batch_size: int = 16,
    learning_rate: float = 1e-4,
    eval_interval: int = 100,
    device: str = 'cuda',
):
    """Train and evaluate model on multi-shift task."""
    
    # Create model
    base_model, desc = create_model(model_name, n_layer=n_layer, device=device)
    
    # Wrap for continuous input
    input_dim = data['train_x'].shape[-1]
    model = ContinuousWrapper(base_model, input_dim, input_dim).to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    flush_print(f"\n{'='*60}")
    flush_print(f"Model: {desc}")
    flush_print(f"Parameters: {n_params:,}")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    train_x, train_y = data['train_x'], data['train_y']
    val_x, val_y = data['val_x'], data['val_y']
    n_train = len(train_x)
    
    history = {'train_loss': [], 'val_loss': [], 'shift_accuracy': []}
    best_val_loss = float('inf')
    
    for iter_num in range(max_iters):
        model.train()
        
        # Sample batch
        idx = torch.randint(0, n_train, (batch_size,))
        xb, yb = train_x[idx], train_y[idx]
        
        # Forward
        _, loss = model(xb, yb)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        history['train_loss'].append(loss.item())
        
        # Evaluate
        if (iter_num + 1) % eval_interval == 0 or iter_num == 0:
            model.eval()
            with torch.no_grad():
                # Validation loss
                pred, val_loss = model(val_x, val_y)
                
                # Compute accuracy at each shift position
                shift_accs = []
                for shift_pos in data['shift_positions']:
                    # Check if prediction flipped correctly at shift
                    if shift_pos < pred.shape[1] and shift_pos > 0:
                        before = pred[:, shift_pos - 1, :]
                        after = pred[:, shift_pos, :]
                        target_after = val_y[:, shift_pos, :]
                        
                        # Cosine similarity with target
                        cos_sim = F.cosine_similarity(after, target_after, dim=-1)
                        shift_accs.append(cos_sim.mean().item())
                
                avg_shift_acc = np.mean(shift_accs) if shift_accs else 0.0
                history['shift_accuracy'].append(avg_shift_acc)
                history['val_loss'].append(val_loss.item())
                
                if val_loss.item() < best_val_loss:
                    best_val_loss = val_loss.item()
                
                flush_print(f"  Step {iter_num+1:4d}: loss={loss.item():.4f}, "
                           f"val_loss={val_loss.item():.4f}, "
                           f"shift_acc={avg_shift_acc:.4f}")
    
    # Final evaluation with detailed breakdown
    model.eval()
    with torch.no_grad():
        pred, final_loss = model(val_x, val_y)
        
        # Per-shift accuracy
        per_shift_acc = []
        for i, shift_pos in enumerate(data['shift_positions']):
            if shift_pos < pred.shape[1]:
                target = val_y[:, shift_pos, :]
                output = pred[:, shift_pos, :]
                cos_sim = F.cosine_similarity(output, target, dim=-1).mean().item()
                per_shift_acc.append(cos_sim)
                flush_print(f"    Shift {i+1} (pos {shift_pos}) accuracy: {cos_sim:.4f}")
        
        # Degradation: how much does accuracy drop from shift 1 to shift N?
        if len(per_shift_acc) > 1:
            degradation = per_shift_acc[0] - per_shift_acc[-1]
            flush_print(f"    Degradation (shift 1 → shift {len(per_shift_acc)}): {degradation:+.4f}")
        else:
            degradation = 0.0
    
    return {
        'model_name': model_name,
        'description': desc,
        'n_params': n_params,
        'best_val_loss': best_val_loss,
        'final_val_loss': final_loss.item(),
        'per_shift_accuracy': per_shift_acc,
        'mean_shift_accuracy': np.mean(per_shift_acc),
        'degradation': degradation,
        'history': history,
    }


# =============================================================================
# MAIN BENCHMARK
# =============================================================================

def run_stress_benchmark(args):
    """Run stress benchmark designed to highlight geometric advantages."""
    
    flush_print("="*80)
    flush_print("STRESS BENCHMARK: Highlighting Geometric Model Advantages")
    flush_print("="*80)
    flush_print(f"""
Configuration (designed to stress non-geometric models):
  - Layers:        {args.n_layer} (DEEP - errors accumulate)
  - Train samples: {args.n_samples} (FEW - tests generalization)
  - Shifts/seq:    {args.n_shifts} (MULTIPLE - tests repeated revision)
  - Seq length:    {args.seq_length} (LONG - tests stability)
  - Dimensions:    {args.dim}

Expected: Geometric models (DDL, Proposed) should show:
  1. Lower validation loss (better generalization)
  2. Higher shift accuracy (better belief revision)
  3. Less degradation across multiple shifts (stability)
""")
    
    # Generate dataset
    flush_print("\n[1] Generating multi-shift dataset...")
    data = generate_multi_shift_dataset(
        n_samples=args.n_samples,
        seq_length=args.seq_length,
        n_shifts=args.n_shifts,
        dim=args.dim,
        device=args.device
    )
    flush_print(f"  Train: {len(data['train_x'])}, Val: {len(data['val_x'])}")
    flush_print(f"  Shift positions: {data['shift_positions']}")
    
    # Models to compare
    models_to_test = ['gpt2', 'ddl', 'mhc', 'proposed']
    results = {}
    
    flush_print("\n[2] Training and evaluating models...")
    
    for model_name in models_to_test:
        try:
            result = train_and_evaluate(
                model_name=model_name,
                data=data,
                n_layer=args.n_layer,
                max_iters=args.max_iters,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                eval_interval=args.eval_interval,
                device=args.device,
            )
            results[model_name] = result
        except Exception as e:
            flush_print(f"\n{'='*60}")
            flush_print(f"ERROR with {model_name}: {e}")
            import traceback
            traceback.print_exc()
            results[model_name] = {'error': str(e)}
    
    # Summary
    flush_print("\n" + "="*80)
    flush_print("STRESS BENCHMARK RESULTS")
    flush_print("="*80)
    
    flush_print(f"\n{'Model':<25} {'Val Loss':>10} {'Shift Acc':>10} {'Degradation':>12} {'Params':>12}")
    flush_print("-"*75)
    
    for name, result in results.items():
        if 'error' in result:
            flush_print(f"{name:<25} ERROR: {result['error'][:40]}")
        else:
            flush_print(f"{result['description']:<25} "
                       f"{result['final_val_loss']:>10.4f} "
                       f"{result['mean_shift_accuracy']:>10.4f} "
                       f"{result['degradation']:>+12.4f} "
                       f"{result['n_params']:>12,}")
    
    flush_print("-"*75)
    
    # Analysis
    flush_print("\n" + "="*80)
    flush_print("ANALYSIS: Geometric Advantage")
    flush_print("="*80)
    
    valid_results = {k: v for k, v in results.items() if 'error' not in v}
    
    if 'gpt2' in valid_results and ('ddl' in valid_results or 'proposed' in valid_results):
        gpt2_loss = valid_results['gpt2']['final_val_loss']
        gpt2_acc = valid_results['gpt2']['mean_shift_accuracy']
        gpt2_deg = valid_results['gpt2']['degradation']
        
        flush_print(f"\nBaseline (GPT2):")
        flush_print(f"  Val Loss: {gpt2_loss:.4f}")
        flush_print(f"  Shift Accuracy: {gpt2_acc:.4f}")
        flush_print(f"  Degradation: {gpt2_deg:+.4f}")
        
        for geo_name in ['ddl', 'proposed']:
            if geo_name in valid_results:
                geo = valid_results[geo_name]
                
                loss_improvement = (gpt2_loss - geo['final_val_loss']) / gpt2_loss * 100
                acc_improvement = geo['mean_shift_accuracy'] - gpt2_acc
                deg_improvement = gpt2_deg - geo['degradation']
                
                flush_print(f"\n{geo['description']} vs GPT2:")
                flush_print(f"  Loss improvement: {loss_improvement:+.1f}%")
                flush_print(f"  Accuracy improvement: {acc_improvement:+.4f}")
                flush_print(f"  Degradation improvement: {deg_improvement:+.4f}")
                
                if loss_improvement > 10 or acc_improvement > 0.05:
                    flush_print(f"  ✓ CLEAR ADVANTAGE for {geo_name.upper()}")
    
    # Per-shift breakdown
    flush_print("\n" + "="*80)
    flush_print("PER-SHIFT ACCURACY BREAKDOWN")
    flush_print("="*80)
    flush_print(f"\n{'Model':<25}", end='')
    for i in range(args.n_shifts):
        flush_print(f"{'Shift '+str(i+1):>10}", end='')
    flush_print(f"{'Degradation':>12}")
    flush_print("-"*75)
    
    for name, result in valid_results.items():
        flush_print(f"{result['description']:<25}", end='')
        for acc in result['per_shift_accuracy']:
            flush_print(f"{acc:>10.4f}", end='')
        flush_print(f"{result['degradation']:>+12.4f}")
    
    flush_print("\n" + "="*80)
    flush_print("Benchmark complete!")
    flush_print("="*80)
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Stress Benchmark for Geometric Models')
    
    # Stress conditions
    parser.add_argument('--n_layer', type=int, default=12,
                       help='Number of layers (deeper = more stress)')
    parser.add_argument('--n_samples', type=int, default=200,
                       help='Training samples (fewer = more stress)')
    parser.add_argument('--n_shifts', type=int, default=3,
                       help='Shifts per sequence (more = harder)')
    parser.add_argument('--seq_length', type=int, default=512,
                       help='Sequence length')
    parser.add_argument('--dim', type=int, default=64,
                       help='Vector dimension')
    
    # Training
    parser.add_argument('--max_iters', type=int, default=1500,
                       help='Training iterations')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--eval_interval', type=int, default=150,
                       help='Evaluation interval')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device')
    
    args = parser.parse_args()
    run_stress_benchmark(args)


if __name__ == '__main__':
    main()
