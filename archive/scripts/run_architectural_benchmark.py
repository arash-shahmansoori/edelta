#!/usr/bin/env python3
"""
Architectural Benchmark: Tests geometric capabilities of different models

Based on insights from "The Illusion of Insight in Reasoning Models" (arXiv:2601.00514v1)

Key Tests:
1. ROTATION: Can the model do rotations?
2. REFLECTION: Can the model do reflections?
3. MIXED: Can the model adapt between rotation/reflection?
4. BELIEF_FLIP: Paper's "Aha!" moment (reflection required)

Model Comparison:
================
| Model  | Mechanism                           | Can Reflect? |
|--------|-------------------------------------|--------------|
| GPT2   | Standard residual (no geometry)     | No inductive bias |
| DDL    | A = I - β·kk^T (rank-1 perturbation)| YES (β→2)    |
| Cayley | Q = (I+M)^{-1}(I-M) (rotation)      | NO (det=+1)  |
| E∆     | Adaptive Householder + Cayley       | YES          |

DDL Reference: "Deep Delta Learning" (arXiv:2601.00417)
- Uses rank-1 perturbation of identity
- β ∈ [0, 2]: β=0 (identity), β=1 (singular), β=2 (reflection)

Cayley: Pure rotation, det(Q) = +1 always, CANNOT do reflections.

E∆-MHC-Geo: Our proposed model with explicit Householder reflection.
"""

import os
import sys
import time
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, Any

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model import GPT as BaselineGPT, GPTConfig as BaselineConfig
from proposed_model_ddl import GPT as DDLGPT, GPTConfig as DDLConfig
from proposed_model_cayley import GPT as CayleyGPT, GPTConfig as CayleyConfig
from proposed_model_mhc_real import GPT as mHCGPT, GPTConfig as mHCConfig
from proposed_model_hybrid import GPT as EdeltaGPT, GPTConfig as EdeltaConfig
from train_continuous import ContinuousModelWrapper


def get_args():
    parser = argparse.ArgumentParser(description='Architectural Benchmark')
    
    parser.add_argument('--task', type=str, default='all',
                        choices=['all', 'rotation', 'reflection', 'mixed', 'belief_flip'])
    parser.add_argument('--model', type=str, default='all',
                        choices=['all', 'gpt2', 'ddl', 'cayley', 'mhc', 'edelta'])
    
    # Training
    parser.add_argument('--max_iters', type=int, default=3000)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--eval_interval', type=int, default=100)
    
    # Model
    parser.add_argument('--n_layer', type=int, default=6)
    parser.add_argument('--n_head', type=int, default=4)
    parser.add_argument('--n_embd', type=int, default=128)
    
    # Hardware
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--out_dir', type=str, default='out-architectural')
    
    return parser.parse_args()


def load_dataset(task: str):
    """Load architectural dataset."""
    task_to_dir = {
        'rotation': 'data/correction_rotation',
        'reflection': 'data/correction_reflection',
        'mixed': 'data/correction_mixed',
        'belief_flip': 'data/correction_belief_flip',
    }
    
    path = task_to_dir[task]
    if not os.path.exists(path):
        print(f"Dataset not found, generating...")
        import subprocess
        subprocess.run(['python', 'data/correction_architectural.py', '--dataset', task])
    
    data = {
        'train_x': torch.from_numpy(np.load(f'{path}/train_x.npy')),
        'train_y': torch.from_numpy(np.load(f'{path}/train_y.npy')),
        'val_x': torch.from_numpy(np.load(f'{path}/val_x.npy')),
        'val_y': torch.from_numpy(np.load(f'{path}/val_y.npy')),
    }
    
    return data


def create_model(model_type: str, args, input_dim: int, block_size: int):
    """Create model based on type."""
    
    if model_type == 'gpt2':
        config = BaselineConfig(
            n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd,
            dropout=0.0, bias=False, block_size=block_size, vocab_size=1
        )
        core = BaselineGPT(config)
        
    elif model_type == 'ddl':
        # Deep Delta Learning (arXiv:2601.00417)
        # Uses rank-1 perturbation: A = I - β·kk^T
        # CAN do reflections when β→2
        config = DDLConfig(
            n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd,
            dropout=0.0, bias=False, block_size=block_size, vocab_size=1
        )
        core = DDLGPT(config)
        
    elif model_type == 'cayley':
        # Pure Cayley rotation (det=+1 only)
        # CANNOT do reflections - this is the key baseline
        config = CayleyConfig(
            n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd,
            n_streams=4, dropout=0.0, bias=False, block_size=block_size, vocab_size=1
        )
        core = CayleyGPT(config)
        
    elif model_type == 'mhc':
        config = mHCConfig(
            n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd,
            n_streams=4, dropout=0.0, bias=False, block_size=block_size, vocab_size=1
        )
        core = mHCGPT(config)
        
    elif model_type == 'edelta':
        # E∆-MHC-Geo: Has BOTH Cayley AND Householder with adaptive gating
        config = EdeltaConfig(
            n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd,
            n_streams=4, dropout=0.0, bias=False, block_size=block_size, vocab_size=1,
            gate_reg_weight=0.1, init_gate_bias=0.0
        )
        core = EdeltaGPT(config)
    
    model = ContinuousModelWrapper(core, config, input_dim, block_size)
    return model


@torch.no_grad()
def evaluate_geometric_metrics(model, data, device, n_samples=500):
    """
    Evaluate with geometric metrics that expose architectural differences.
    
    Key metrics:
    - MSE loss (standard)
    - Cosine similarity (overall)
    - Flip accuracy: How well does model flip v → -v?
    - Isometry error: Does output preserve norms?
    """
    model.eval()
    
    val_x = data['val_x'][:n_samples].to(device)
    val_y = data['val_y'][:n_samples].to(device)
    
    pred, loss = model(val_x, val_y)
    
    # Overall cosine similarity
    pred_flat = pred.view(-1, pred.size(-1))
    y_flat = val_y.view(-1, val_y.size(-1))
    pred_norm = F.normalize(pred_flat, dim=-1)
    y_norm = F.normalize(y_flat, dim=-1)
    cos_sim = (pred_norm * y_norm).sum(dim=-1).mean().item()
    
    # Flip detection: At positions where target is -input, measure accuracy
    flip_scores = []
    isometry_errors = []
    
    for b in range(min(100, val_x.size(0))):
        for t in range(1, val_x.size(1)):
            # Check if this is a flip position
            inp_t = val_x[b, t-1]
            tgt_t = val_y[b, t]
            
            # Is target the negation of previous input?
            cos_inp_tgt = F.cosine_similarity(inp_t.unsqueeze(0), tgt_t.unsqueeze(0)).item()
            
            if cos_inp_tgt < -0.9:  # This is a flip
                pred_t = pred[b, t]
                flip_cos = F.cosine_similarity(pred_t.unsqueeze(0), tgt_t.unsqueeze(0)).item()
                flip_scores.append(flip_cos)
            
            # Isometry check: ||pred|| should equal ||target||
            pred_norm_t = pred[b, t].norm().item()
            tgt_norm_t = tgt_t.norm().item()
            if tgt_norm_t > 0.1:
                isometry_errors.append(abs(pred_norm_t - tgt_norm_t) / tgt_norm_t)
    
    metrics = {
        'val_loss': loss.item(),
        'cosine_sim': cos_sim,
        'flip_accuracy': np.mean(flip_scores) if flip_scores else 0.0,
        'n_flips': len(flip_scores),
        'isometry_error': np.mean(isometry_errors) if isometry_errors else 0.0,
    }
    
    model.train()
    return metrics


def train_model(model_type: str, task: str, args) -> Dict[str, Any]:
    """Train a single model on a single task."""
    
    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'
    
    print(f"\n{'='*60}")
    print(f"Training {model_type.upper()} on {task.upper()}")
    print(f"{'='*60}")
    
    # Load data
    data = load_dataset(task)
    input_dim = data['train_x'].shape[-1]
    seq_len = data['train_x'].shape[1]
    
    # Create model
    model = create_model(model_type, args, input_dim, seq_len + 1)
    model = model.to(device)
    n_params = model.get_num_params()
    print(f"Parameters: {n_params/1e6:.2f}M")
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.1)
    
    # Training
    best_val_loss = float('inf')
    t0 = time.time()
    
    for iter_num in range(args.max_iters):
        # LR schedule
        lr = args.learning_rate * (1 - iter_num / args.max_iters)
        for pg in optimizer.param_groups:
            pg['lr'] = lr
        
        # Evaluate
        if iter_num % args.eval_interval == 0:
            metrics = evaluate_geometric_metrics(model, data, device)
            print(f"step {iter_num}: loss {metrics['val_loss']:.4e}, "
                  f"cos {metrics['cosine_sim']:.4f}, flip {metrics['flip_accuracy']:.4f}")
            
            if metrics['val_loss'] < best_val_loss:
                best_val_loss = metrics['val_loss']
        
        # Training step
        ix = torch.randint(len(data['train_x']), (args.batch_size,))
        x = data['train_x'][ix].to(device)
        y = data['train_y'][ix].to(device)
        
        _, loss = model(x, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    
    t1 = time.time()
    
    # Final evaluation
    final_metrics = evaluate_geometric_metrics(model, data, device)
    
    results = {
        'model_type': model_type,
        'task': task,
        'best_val_loss': best_val_loss,
        'final_metrics': final_metrics,
        'training_time': t1 - t0,
        'n_params': n_params,
    }
    
    # Save
    out_path = os.path.join(args.out_dir, f'{model_type}_{task}')
    os.makedirs(out_path, exist_ok=True)
    np.save(os.path.join(out_path, 'results.npy'), results)
    torch.save(model.state_dict(), os.path.join(out_path, 'model.pt'))
    
    return results


def run_benchmark(args):
    """Run full benchmark."""
    
    tasks = ['rotation', 'reflection', 'mixed', 'belief_flip'] if args.task == 'all' else [args.task]
    models = ['gpt2', 'ddl', 'cayley', 'mhc', 'edelta'] if args.model == 'all' else [args.model]
    
    # Generate datasets first
    print("Generating datasets...")
    import subprocess
    subprocess.run(['python', 'data/correction_architectural.py', '--dataset', 'all'])
    
    results = {}
    
    for task in tasks:
        results[task] = {}
        for model_type in models:
            results[task][model_type] = train_model(model_type, task, args)
    
    # Print summary
    print("\n" + "="*80)
    print("ARCHITECTURAL BENCHMARK RESULTS")
    print("="*80)
    print("\nModel Architecture Summary:")
    print("  GPT2:   Standard residual (no geometric inductive bias)")
    print("  DDL:    A = I - β·kk^T (rank-1 perturbation, CAN reflect at β=2)")
    print("  Cayley: Q = (I+M)^{-1}(I-M) (rotation only, det=+1, CANNOT reflect)")
    print("  mHC:    Multi-head Householder chains")
    print("  E∆:     Adaptive Householder + Cayley with β-gating")
    
    print(f"\n{'Task':<15} {'Model':<10} {'Val Loss':<12} {'Cos Sim':<10} {'Flip Acc':<10}")
    print("-"*65)
    
    for task in tasks:
        for model_type in models:
            r = results[task][model_type]
            fm = r['final_metrics']
            print(f"{task:<15} {model_type:<10} {r['best_val_loss']:.4e}   "
                  f"{fm['cosine_sim']:.4f}     {fm['flip_accuracy']:.4f}")
        print()
    
    # Analysis
    print("\n" + "="*80)
    print("ANALYSIS")
    print("="*80)
    
    # Check REFLECTION task specifically (key differentiator)
    if 'reflection' in results:
        ref_results = results['reflection']
        print(f"\nREFLECTION TASK (Critical Differentiator):")
        print("-" * 50)
        
        cayley_flip = ref_results['cayley']['final_metrics']['flip_accuracy']
        ddl_flip = ref_results['ddl']['final_metrics']['flip_accuracy']
        edelta_flip = ref_results['edelta']['final_metrics']['flip_accuracy']
        
        print(f"  Cayley flip accuracy: {cayley_flip:.4f}")
        print(f"  DDL flip accuracy:    {ddl_flip:.4f}")
        print(f"  E∆ flip accuracy:     {edelta_flip:.4f}")
        
        # Cayley should fail on reflections
        if ddl_flip > cayley_flip + 0.1 or edelta_flip > cayley_flip + 0.1:
            print(f"\n  ✓ Cayley performs worse on reflections!")
            print(f"    This confirms: Cayley (det=+1) CANNOT do reflections (det=-1)")
        
        # Compare DDL vs E∆
        if abs(ddl_flip - edelta_flip) < 0.1:
            print(f"\n  DDL ≈ E∆: Both can do reflections via different mechanisms")
            print(f"    DDL: rank-1 perturbation with β→2")
            print(f"    E∆: explicit Householder reflection")
    
    # Check ROTATION task
    if 'rotation' in results:
        rot_results = results['rotation']
        print(f"\nROTATION TASK:")
        print("-" * 50)
        for model_type in models:
            cos = rot_results[model_type]['final_metrics']['cosine_sim']
            print(f"  {model_type:<10}: cos_sim = {cos:.4f}")
    
    if 'belief_flip' in results:
        bf_results = results['belief_flip']
        print(f"\nBELIEF FLIP TASK (Paper's 'Aha!' Moment):")
        print("-" * 50)
        for model_type in models:
            flip_acc = bf_results[model_type]['final_metrics']['flip_accuracy']
            print(f"  {model_type:<10}: flip_acc = {flip_acc:.4f}")
    
    # Save summary
    np.save(os.path.join(args.out_dir, 'benchmark_results.npy'), results)
    print(f"\nResults saved to: {args.out_dir}/")
    
    return results


if __name__ == '__main__':
    args = get_args()
    os.makedirs(args.out_dir, exist_ok=True)
    run_benchmark(args)
