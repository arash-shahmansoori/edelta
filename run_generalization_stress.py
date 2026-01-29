#!/usr/bin/env python3
"""
Generalization Stress Test: Where Geometric Models Should CLEARLY Win

The key insight from DDL paper (arXiv:2601.00417):
- Geometric models learn the STRUCTURE of reflection, not just input-output mapping
- GPT2 learns f(x) ≈ -x for SPECIFIC directions seen in training
- When tested on UNSEEN directions, GPT2 should fail; geometric should generalize

Test Design:
1. Train on reflections in dimensions [0, d/2) only
2. Test on reflections in dimensions [d/2, d) - COMPLETELY UNSEEN
3. Use extreme few-shot (30-50 samples)
4. Deep networks (12 layers)

This is the DEFINITIVE test of geometric inductive bias.
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
# OUT-OF-DISTRIBUTION REFLECTION DATASET
# =============================================================================

def generate_ood_reflection_dataset(
    n_train: int = 50,
    n_val: int = 100,
    seq_length: int = 32,
    dim: int = 64,
    noise_std: float = 0.05,
    device: str = 'cuda'
):
    """
    Generate dataset where TRAINING and TESTING use DIFFERENT reflection directions.
    
    Training: Reflections in dimensions [0, dim//2)
    Testing: Reflections in dimensions [dim//2, dim)
    
    This tests whether models learned:
    - GPT2: A specific function f(x) ≈ -x for seen directions (WILL FAIL on unseen)
    - Geometric: The structure of reflection (SHOULD GENERALIZE)
    """
    
    train_dim_range = (0, dim // 2)
    test_dim_range = (dim // 2, dim)
    
    def make_reflection_sequence(dim_range, n_samples):
        """Create sequences with reflection at midpoint."""
        x_all = []
        y_all = []
        
        for _ in range(n_samples):
            # Create random reflection direction in specified dim range
            direction = torch.zeros(dim, device=device)
            active_dims = torch.randint(dim_range[0], dim_range[1], (dim // 4,))
            direction[active_dims] = torch.randn(len(active_dims), device=device)
            direction = direction / (direction.norm() + 1e-8)
            
            # Generate belief vector (random but structured)
            belief = torch.randn(dim, device=device)
            belief = belief / belief.norm() * 2.0
            
            x_seq = []
            y_seq = []
            
            # First half: maintain belief
            for t in range(seq_length // 2):
                noise = torch.randn(dim, device=device) * noise_std
                x_seq.append(belief + noise)
                y_seq.append(belief.clone())
            
            # Second half: REFLECT belief (Householder: v' = v - 2(v·d)d)
            reflected_belief = belief - 2 * (belief @ direction) * direction
            
            for t in range(seq_length // 2):
                noise = torch.randn(dim, device=device) * noise_std
                if t == 0:
                    # Add trigger signal
                    signal = torch.zeros(dim, device=device)
                    signal[0] = 5.0
                    x_seq.append(reflected_belief + noise + signal)
                else:
                    x_seq.append(reflected_belief + noise)
                y_seq.append(reflected_belief.clone())
            
            x_all.append(torch.stack(x_seq))
            y_all.append(torch.stack(y_seq))
        
        return torch.stack(x_all), torch.stack(y_all)
    
    # Training data: reflections in first half of dimensions
    train_x, train_y = make_reflection_sequence(train_dim_range, n_train)
    
    # Validation data (in-distribution): same dimension range as training
    val_in_x, val_in_y = make_reflection_sequence(train_dim_range, n_val // 2)
    
    # Validation data (OUT-OF-DISTRIBUTION): different dimension range!
    val_ood_x, val_ood_y = make_reflection_sequence(test_dim_range, n_val // 2)
    
    return {
        'train_x': train_x,
        'train_y': train_y,
        'val_in_x': val_in_x,  # In-distribution test
        'val_in_y': val_in_y,
        'val_ood_x': val_ood_x,  # Out-of-distribution test (KEY!)
        'val_ood_y': val_ood_y,
        'train_dim_range': train_dim_range,
        'test_dim_range': test_dim_range,
    }


# =============================================================================
# MODEL WRAPPER
# =============================================================================

class ContinuousWrapper(nn.Module):
    """Wrap discrete token model for continuous vectors."""
    
    def __init__(self, model, input_dim: int):
        super().__init__()
        self.model = model
        n_embd = model.config.n_embd
        self.input_proj = nn.Linear(input_dim, n_embd)
        self.output_proj = nn.Linear(n_embd, input_dim)
        
    def forward(self, x, targets=None):
        B, T, D = x.shape
        h = self.input_proj(x)
        
        if hasattr(self.model, 'transformer'):
            for block in self.model.transformer.h:
                h = block(h)
            h = self.model.transformer.ln_f(h)
        else:
            for block in self.model.blocks:
                h = block(h)
            h = self.model.ln_f(h)
        
        out = self.output_proj(h)
        
        if targets is not None:
            loss = F.mse_loss(out, targets)
            return out, loss
        return out, None


def create_model(model_name: str, n_layer: int = 12, n_embd: int = 128, device: str = 'cuda'):
    """Create model."""
    
    base_config = {
        'vocab_size': 256,
        'block_size': 64,
        'n_layer': n_layer,
        'n_head': 4,
        'n_embd': n_embd,
        'dropout': 0.0,  # No dropout for clean comparison
        'bias': True,
    }
    
    if model_name == 'gpt2':
        from model import GPT, GPTConfig
        config = GPTConfig(**base_config)
        return GPT(config).to(device), 'GPT2'
        
    elif model_name == 'ddl':
        from proposed_model_ddl import GPT, GPTConfig
        config = GPTConfig(**base_config)
        return GPT(config).to(device), 'DDL'
        
    elif model_name == 'proposed':
        from proposed_model_hybrid import GPT, GPTConfig
        config = GPTConfig(**base_config)
        return GPT(config).to(device), 'E∆-MHC-Geo'
    
    raise ValueError(f"Unknown model: {model_name}")


# =============================================================================
# TRAINING AND EVALUATION
# =============================================================================

def train_and_evaluate(
    model_name: str,
    data: dict,
    n_layer: int = 12,
    max_iters: int = 2000,
    batch_size: int = 16,
    learning_rate: float = 1e-4,
    eval_interval: int = 200,
    device: str = 'cuda',
):
    """Train and evaluate with in-distribution and out-of-distribution tests."""
    
    base_model, desc = create_model(model_name, n_layer=n_layer, device=device)
    input_dim = data['train_x'].shape[-1]
    model = ContinuousWrapper(base_model, input_dim).to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    flush_print(f"\n{'='*60}")
    flush_print(f"Model: {desc} ({n_layer}L, {n_params:,} params)")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    train_x, train_y = data['train_x'], data['train_y']
    val_in_x, val_in_y = data['val_in_x'], data['val_in_y']
    val_ood_x, val_ood_y = data['val_ood_x'], data['val_ood_y']
    
    n_train = len(train_x)
    best_in_loss = float('inf')
    best_ood_loss = float('inf')
    
    history = {'train_loss': [], 'val_in_loss': [], 'val_ood_loss': [], 
               'val_in_acc': [], 'val_ood_acc': []}
    
    for iter_num in range(max_iters):
        model.train()
        
        # Sample batch
        idx = torch.randint(0, n_train, (min(batch_size, n_train),))
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
                # In-distribution
                pred_in, loss_in = model(val_in_x, val_in_y)
                # Focus on second half (after reflection)
                T = pred_in.shape[1]
                cos_in = F.cosine_similarity(
                    pred_in[:, T//2:, :].reshape(-1, input_dim),
                    val_in_y[:, T//2:, :].reshape(-1, input_dim),
                    dim=-1
                ).mean().item()
                
                # Out-of-distribution (KEY METRIC!)
                pred_ood, loss_ood = model(val_ood_x, val_ood_y)
                cos_ood = F.cosine_similarity(
                    pred_ood[:, T//2:, :].reshape(-1, input_dim),
                    val_ood_y[:, T//2:, :].reshape(-1, input_dim),
                    dim=-1
                ).mean().item()
                
                history['val_in_loss'].append(loss_in.item())
                history['val_ood_loss'].append(loss_ood.item())
                history['val_in_acc'].append(cos_in)
                history['val_ood_acc'].append(cos_ood)
                
                if loss_in.item() < best_in_loss:
                    best_in_loss = loss_in.item()
                if loss_ood.item() < best_ood_loss:
                    best_ood_loss = loss_ood.item()
                
                flush_print(f"  Step {iter_num+1:4d}: train={loss.item():.4f}, "
                           f"val_in={loss_in.item():.4f} ({cos_in:.4f}), "
                           f"val_OOD={loss_ood.item():.4f} ({cos_ood:.4f})")
    
    # Final evaluation
    model.eval()
    with torch.no_grad():
        _, final_in_loss = model(val_in_x, val_in_y)
        _, final_ood_loss = model(val_ood_x, val_ood_y)
        
        pred_in, _ = model(val_in_x)
        pred_ood, _ = model(val_ood_x)
        
        T = pred_in.shape[1]
        final_in_acc = F.cosine_similarity(
            pred_in[:, T//2:, :].reshape(-1, input_dim),
            val_in_y[:, T//2:, :].reshape(-1, input_dim),
            dim=-1
        ).mean().item()
        
        final_ood_acc = F.cosine_similarity(
            pred_ood[:, T//2:, :].reshape(-1, input_dim),
            val_ood_y[:, T//2:, :].reshape(-1, input_dim),
            dim=-1
        ).mean().item()
    
    # Generalization gap = In-dist accuracy - OOD accuracy
    # Smaller gap = better generalization
    gen_gap = final_in_acc - final_ood_acc
    
    return {
        'model_name': model_name,
        'description': desc,
        'n_params': n_params,
        'final_in_loss': final_in_loss.item(),
        'final_ood_loss': final_ood_loss.item(),
        'final_in_acc': final_in_acc,
        'final_ood_acc': final_ood_acc,
        'generalization_gap': gen_gap,
        'history': history,
    }


# =============================================================================
# MAIN
# =============================================================================

def run_generalization_stress(args):
    """Run generalization stress test."""
    
    flush_print("="*80)
    flush_print("GENERALIZATION STRESS TEST")
    flush_print("The DEFINITIVE test of geometric inductive bias")
    flush_print("="*80)
    flush_print(f"""
Test Design:
  - Train samples: {args.n_train} (EXTREME few-shot)
  - Train dimensions: [0, {args.dim//2}) 
  - Test dimensions: [{args.dim//2}, {args.dim}) (COMPLETELY UNSEEN!)
  - Layers: {args.n_layer}

Expected Results:
  - GPT2: Good in-distribution, FAILS on out-of-distribution
  - DDL/Proposed: Good on BOTH (geometric structure generalizes)
  
The KEY metric is: Out-of-Distribution Accuracy (val_OOD)
""")
    
    # Generate dataset
    flush_print("\n[1] Generating OOD reflection dataset...")
    data = generate_ood_reflection_dataset(
        n_train=args.n_train,
        n_val=args.n_val,
        seq_length=args.seq_length,
        dim=args.dim,
        device=args.device
    )
    flush_print(f"  Train: {len(data['train_x'])} samples, dims [{data['train_dim_range'][0]}, {data['train_dim_range'][1]})")
    flush_print(f"  Val (in-dist): {len(data['val_in_x'])} samples")
    flush_print(f"  Val (OOD): {len(data['val_ood_x'])} samples, dims [{data['test_dim_range'][0]}, {data['test_dim_range'][1]})")
    
    # Models to compare
    models_to_test = ['gpt2', 'ddl', 'proposed']
    results = {}
    
    flush_print("\n[2] Training and evaluating...")
    
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
            flush_print(f"ERROR with {model_name}: {e}")
            import traceback
            traceback.print_exc()
            results[model_name] = {'error': str(e)}
    
    # Summary
    flush_print("\n" + "="*80)
    flush_print("GENERALIZATION STRESS TEST RESULTS")
    flush_print("="*80)
    
    flush_print(f"\n{'Model':<20} {'In-Dist Acc':>12} {'OOD Acc':>12} {'Gen Gap':>10} {'Params':>12}")
    flush_print("-"*70)
    
    for name, result in results.items():
        if 'error' in result:
            flush_print(f"{name:<20} ERROR")
        else:
            flush_print(f"{result['description']:<20} "
                       f"{result['final_in_acc']:>12.4f} "
                       f"{result['final_ood_acc']:>12.4f} "
                       f"{result['generalization_gap']:>+10.4f} "
                       f"{result['n_params']:>12,}")
    
    flush_print("-"*70)
    
    # Analysis
    flush_print("\n" + "="*80)
    flush_print("ANALYSIS")
    flush_print("="*80)
    
    valid = {k: v for k, v in results.items() if 'error' not in v}
    
    if 'gpt2' in valid:
        gpt2 = valid['gpt2']
        flush_print(f"\nGPT2 Baseline:")
        flush_print(f"  In-distribution accuracy:  {gpt2['final_in_acc']:.4f}")
        flush_print(f"  Out-of-distribution acc:   {gpt2['final_ood_acc']:.4f}")
        flush_print(f"  Generalization gap:        {gpt2['generalization_gap']:+.4f}")
        
        for geo_name in ['ddl', 'proposed']:
            if geo_name in valid:
                geo = valid[geo_name]
                
                # OOD improvement is the key metric
                ood_improvement = geo['final_ood_acc'] - gpt2['final_ood_acc']
                gap_improvement = gpt2['generalization_gap'] - geo['generalization_gap']
                
                flush_print(f"\n{geo['description']} vs GPT2:")
                flush_print(f"  OOD accuracy improvement:     {ood_improvement:+.4f}")
                flush_print(f"  Generalization gap reduction: {gap_improvement:+.4f}")
                
                if ood_improvement > 0.1:
                    flush_print(f"  ✓ CLEAR GENERALIZATION ADVANTAGE!")
                elif ood_improvement > 0.05:
                    flush_print(f"  ✓ Moderate generalization advantage")
    
    # Determine winner
    flush_print("\n" + "="*80)
    flush_print("CONCLUSION")
    flush_print("="*80)
    
    if valid:
        best_ood = max(valid.items(), key=lambda x: x[1]['final_ood_acc'])
        worst_gap = max(valid.items(), key=lambda x: x[1]['generalization_gap'])
        best_gap = min(valid.items(), key=lambda x: x[1]['generalization_gap'])
        
        flush_print(f"\nBest OOD Accuracy: {best_ood[1]['description']} ({best_ood[1]['final_ood_acc']:.4f})")
        flush_print(f"Best Generalization (smallest gap): {best_gap[1]['description']} ({best_gap[1]['generalization_gap']:+.4f})")
        flush_print(f"Worst Generalization (largest gap): {worst_gap[1]['description']} ({worst_gap[1]['generalization_gap']:+.4f})")
        
        if best_ood[0] in ['ddl', 'proposed']:
            flush_print(f"\n✓ GEOMETRIC MODELS SHOW SUPERIOR GENERALIZATION!")
            flush_print(f"  This confirms the inductive bias helps learn the STRUCTURE of reflection,")
            flush_print(f"  not just memorize specific input-output mappings.")
    
    flush_print("\n" + "="*80)
    flush_print("Benchmark complete!")
    flush_print("="*80)
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Generalization Stress Test')
    
    parser.add_argument('--n_train', type=int, default=50,
                       help='Training samples (fewer = harder)')
    parser.add_argument('--n_val', type=int, default=100,
                       help='Validation samples')
    parser.add_argument('--seq_length', type=int, default=32,
                       help='Sequence length')
    parser.add_argument('--dim', type=int, default=64,
                       help='Vector dimension')
    parser.add_argument('--n_layer', type=int, default=12,
                       help='Number of layers')
    parser.add_argument('--max_iters', type=int, default=2000,
                       help='Training iterations')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--eval_interval', type=int, default=200,
                       help='Eval interval')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device')
    
    args = parser.parse_args()
    run_generalization_stress(args)


if __name__ == '__main__':
    main()
