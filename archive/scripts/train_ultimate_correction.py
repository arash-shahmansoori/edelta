#!/usr/bin/env python3
"""
Training Script for Ultimate Correction Benchmarks

Includes metrics inspired by:
"The Illusion of Insight in Reasoning Models" (arXiv:2601.00514v1)

Key insights from the paper:
- Spontaneous reasoning shifts are rare (~6.31%) and generally harmful
- Externally triggered reconsideration reliably improves accuracy (+8.41pp)
- Effect is amplified under high uncertainty (top 20% entropy: +15.38pp)

Tasks (ordered by difficulty):
1. rotation_reflection - Direct test of Cayley (rotation) vs Householder (reflection)
2. cumulative - Track flip parity (cumulative correction)
3. accumulation - Running sum with negation
4. echo - Long-range memory + conditional correction
5. isometry - Ultimate isometry preservation test
6. noise_robust - Correction under noise perturbation
"""

import os
import math
import time
import argparse
from typing import Dict, Any

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

from model import GPT as BaselineGPT, GPTConfig as BaselineConfig
from proposed_model_ddl import GPT as DDLGPT, GPTConfig as DDLConfig
from proposed_model_mhc_real import GPT as mHCGPT, GPTConfig as mHCConfig
from proposed_model_hybrid import GPT as EdeltaGPT, GPTConfig as EdeltaConfig
from train_continuous import ContinuousModelWrapper


def get_args():
    parser = argparse.ArgumentParser(description='Ultimate Correction Benchmarks')
    
    parser.add_argument('--model_type', type=str, default='edelta',
                        choices=['gpt2', 'ddl', 'mhc', 'edelta'])
    parser.add_argument('--task', type=str, default='rotation_reflection',
                        choices=['cumulative', 'accumulation', 'echo', 
                                'rotation_reflection', 'isometry', 'noise_robust'])
    
    # Training
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--max_iters', type=int, default=3000)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--min_lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.1)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    
    # Model
    parser.add_argument('--n_layer', type=int, default=6)
    parser.add_argument('--n_head', type=int, default=4)
    parser.add_argument('--n_embd', type=int, default=128)
    parser.add_argument('--n_streams', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.0)
    
    # Logging
    parser.add_argument('--out_dir', type=str, default='out-ultimate')
    parser.add_argument('--eval_interval', type=int, default=100)
    parser.add_argument('--eval_iters', type=int, default=50)
    parser.add_argument('--log_interval', type=int, default=100)
    
    # Hardware
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=42)
    
    # E∆-MHC-Geo
    parser.add_argument('--gate_reg_weight', type=float, default=0.1)
    parser.add_argument('--init_gate_bias', type=float, default=0.0)
    
    # Benchmark mode
    parser.add_argument('--compare', action='store_true',
                        help='Run comparison across all model types')
    
    return parser.parse_args()


def load_data(task):
    """Load ultimate correction dataset."""
    filepath = f'data/correction_ultimate_{task}.npz'
    data = np.load(filepath)
    
    return {
        'train_x': torch.from_numpy(data['train_x']),
        'train_y': torch.from_numpy(data['train_y']),
        'val_x': torch.from_numpy(data['val_x']),
        'val_y': torch.from_numpy(data['val_y']),
    }


def get_batch(data, split, batch_size, device):
    """Get a batch of data."""
    x = data[f'{split}_x']
    y = data[f'{split}_y']
    
    ix = torch.randint(len(x), (batch_size,))
    x_batch = x[ix].to(device)
    y_batch = y[ix].to(device)
    
    return x_batch, y_batch


def get_lr(it, warmup_iters, lr_decay_iters, learning_rate, min_lr):
    """Learning rate schedule."""
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)


@torch.no_grad()
def estimate_loss(model, data, batch_size, eval_iters, device):
    """Estimate loss on train and val sets."""
    model.eval()
    out = {}
    for split in ['train', 'val']:
        losses = []
        for _ in range(eval_iters):
            x, y = get_batch(data, split, batch_size, device)
            _, loss = model(x, y)
            losses.append(loss.item())
        out[split] = np.mean(losses)
    model.train()
    return out


@torch.no_grad()
def compute_paper_metrics(model, data, device, batch_size=64) -> Dict[str, float]:
    """
    Compute metrics inspired by arXiv:2601.00514v1 (Illusion of Insight paper).
    
    Metrics:
    - MSE loss
    - Cosine similarity (overall and at correction points)
    - Correction quality (how well model executes belief flips)
    - Shift detection accuracy (does model recognize correction signals?)
    """
    model.eval()
    
    val_x = data['val_x'].to(device)
    val_y = data['val_y'].to(device)
    
    all_cos_sim = []
    correction_quality = []
    
    n_batches = min(10, len(val_x) // batch_size)
    
    for i in range(n_batches):
        start = i * batch_size
        end = start + batch_size
        x, y = val_x[start:end], val_y[start:end]
        
        pred, _ = model(x, y)
        
        # Cosine similarity
        pred_flat = pred.view(-1, pred.size(-1))
        y_flat = y.view(-1, y.size(-1))
        pred_norm = F.normalize(pred_flat, dim=-1)
        y_norm = F.normalize(y_flat, dim=-1)
        cos_sim = (pred_norm * y_norm).sum(dim=-1)
        all_cos_sim.extend(cos_sim.cpu().tolist())
        
        # Correction quality: measure at positions where target flips
        for b in range(x.size(0)):
            for t in range(1, x.size(1)):
                # Detect flip in target
                prev_y = y[b, t-1]
                curr_y = y[b, t]
                target_cos = F.cosine_similarity(
                    prev_y.unsqueeze(0), curr_y.unsqueeze(0)
                ).item()
                
                if target_cos < -0.5:  # Target flipped sign
                    # Check if prediction also flipped correctly
                    pred_cos = F.cosine_similarity(
                        pred[b, t].unsqueeze(0), curr_y.unsqueeze(0)
                    ).item()
                    correction_quality.append(pred_cos)
    
    metrics = {
        'cosine_sim_mean': np.mean(all_cos_sim),
        'cosine_sim_std': np.std(all_cos_sim),
    }
    
    if correction_quality:
        metrics['correction_quality'] = np.mean(correction_quality)
        metrics['correction_std'] = np.std(correction_quality)
        metrics['n_corrections'] = len(correction_quality)
    else:
        metrics['correction_quality'] = 0.0
        metrics['correction_std'] = 0.0
        metrics['n_corrections'] = 0
    
    model.train()
    return metrics


def create_model(args, input_dim, block_size):
    """Create model based on model_type."""
    
    if args.model_type == 'gpt2':
        config = BaselineConfig(
            n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd,
            dropout=args.dropout, bias=False, block_size=block_size, vocab_size=1
        )
        core = BaselineGPT(config)
        
    elif args.model_type == 'ddl':
        config = DDLConfig(
            n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd,
            dropout=args.dropout, bias=False, block_size=block_size, vocab_size=1
        )
        core = DDLGPT(config)
        
    elif args.model_type == 'mhc':
        config = mHCConfig(
            n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd,
            n_streams=args.n_streams, dropout=args.dropout, bias=False,
            block_size=block_size, vocab_size=1
        )
        core = mHCGPT(config)
        
    elif args.model_type == 'edelta':
        config = EdeltaConfig(
            n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd,
            n_streams=args.n_streams, dropout=args.dropout, bias=False,
            block_size=block_size, vocab_size=1,
            gate_reg_weight=args.gate_reg_weight, init_gate_bias=args.init_gate_bias
        )
        core = EdeltaGPT(config)
    
    model = ContinuousModelWrapper(core, config, input_dim, block_size)
    return model


def train(args, model_type: str = None, task: str = None) -> Dict[str, Any]:
    """Main training loop."""
    
    if model_type:
        args.model_type = model_type
    if task:
        args.task = task
    
    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    out_dir = os.path.join(args.out_dir, f'{args.model_type}_{args.task}')
    os.makedirs(out_dir, exist_ok=True)
    
    # Load data
    print(f"\n{'='*60}")
    print(f"Training {args.model_type.upper()} on {args.task}")
    print(f"{'='*60}")
    
    data = load_data(args.task)
    
    input_dim = data['train_x'].shape[-1]
    seq_len = data['train_x'].shape[1]
    
    print(f"Input dimension: {input_dim}")
    print(f"Sequence length: {seq_len}")
    print(f"Train samples: {len(data['train_x'])}")
    
    # Create model
    model = create_model(args, input_dim, seq_len + 1)
    model = model.to(device)
    
    n_params = model.get_num_params()
    print(f"Model: {args.model_type} ({n_params / 1e6:.2f}M params)")
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate,
                                   weight_decay=args.weight_decay, betas=(0.9, 0.95))
    
    # Training
    warmup_iters = 100
    best_val_loss = float('inf')
    train_log = {'iter': [], 'train_loss': [], 'val_loss': [], 'grad_norm': []}
    
    t0 = time.time()
    for iter_num in range(args.max_iters):
        lr = get_lr(iter_num, warmup_iters, args.max_iters, args.learning_rate, args.min_lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        # Evaluate
        if iter_num % args.eval_interval == 0:
            losses = estimate_loss(model, data, args.batch_size, args.eval_iters, device)
            print(f"step {iter_num}: train {losses['train']:.4e}, val {losses['val']:.4e}")
            
            train_log['iter'].append(iter_num)
            train_log['train_loss'].append(losses['train'])
            train_log['val_loss'].append(losses['val'])
            
            if losses['val'] < best_val_loss:
                best_val_loss = losses['val']
                checkpoint = {
                    'model': model.state_dict(),
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                }
                torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
        
        # Training step
        x, y = get_batch(data, 'train', args.batch_size, device)
        _, loss = model(x, y)
        
        # Gate regularization for edelta
        if args.model_type == 'edelta' and hasattr(model.core, 'get_gate_regularization_loss'):
            gate_loss = model.core.get_gate_regularization_loss()
            loss = loss + gate_loss
        
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        
        if iter_num % args.log_interval == 0:
            train_log['grad_norm'].append(float(grad_norm))
    
    t1 = time.time()
    
    # Final evaluation with paper metrics
    final_losses = estimate_loss(model, data, args.batch_size, 100, device)
    paper_metrics = compute_paper_metrics(model, data, device)
    
    print(f"\n{'='*40}")
    print(f"Results ({args.model_type} on {args.task}):")
    print(f"  Best val loss: {best_val_loss:.4e}")
    print(f"  Cosine similarity: {paper_metrics['cosine_sim_mean']:.4f} ± {paper_metrics['cosine_sim_std']:.4f}")
    print(f"  Correction quality: {paper_metrics['correction_quality']:.4f}")
    print(f"  Training time: {(t1-t0)/60:.1f} min")
    
    # Save
    results = {
        'model_type': args.model_type,
        'task': args.task,
        'best_val_loss': best_val_loss,
        'final_val_loss': final_losses['val'],
        'final_train_loss': final_losses['train'],
        'paper_metrics': paper_metrics,
        'training_time': t1 - t0,
        'n_params': n_params,
    }
    np.save(os.path.join(out_dir, 'results.npy'), results)
    np.save(os.path.join(out_dir, 'train_log.npy'), train_log)
    
    return results


def run_comparison(args) -> Dict[str, Dict]:
    """Run comparison across all model types on a single task."""
    
    model_types = ['gpt2', 'ddl', 'mhc', 'edelta']
    all_results = {}
    
    print("\n" + "=" * 70)
    print(f"COMPARISON BENCHMARK: {args.task}")
    print("Metrics inspired by arXiv:2601.00514v1 (Illusion of Insight)")
    print("=" * 70)
    
    for model_type in model_types:
        results = train(args, model_type=model_type)
        all_results[model_type] = results
    
    # Print comparison table
    print("\n" + "=" * 70)
    print("COMPARISON RESULTS")
    print("=" * 70)
    print(f"\n{'Model':<12} {'Val Loss':<12} {'Cos Sim':<12} {'Correction':<12} {'Time':<8}")
    print("-" * 60)
    
    for model_type, results in all_results.items():
        pm = results['paper_metrics']
        print(f"{model_type:<12} {results['best_val_loss']:.4e}  "
              f"{pm['cosine_sim_mean']:.4f}       {pm['correction_quality']:.4f}       "
              f"{results['training_time']/60:.1f}m")
    
    # Relative improvements vs baseline
    baseline = all_results['gpt2']
    print(f"\n{'='*60}")
    print("Relative Improvement vs GPT2 Baseline:")
    print("-" * 60)
    
    for model_type in ['ddl', 'mhc', 'edelta']:
        results = all_results[model_type]
        loss_delta = (baseline['best_val_loss'] - results['best_val_loss']) / baseline['best_val_loss'] * 100
        cos_delta = (results['paper_metrics']['cosine_sim_mean'] - 
                    baseline['paper_metrics']['cosine_sim_mean']) * 100
        corr_delta = (results['paper_metrics']['correction_quality'] - 
                     baseline['paper_metrics']['correction_quality']) * 100
        
        print(f"{model_type:<12} Loss: {loss_delta:+.1f}%  CosSim: {cos_delta:+.2f}pp  Correction: {corr_delta:+.2f}pp")
    
    # Save comparison
    np.save(os.path.join(args.out_dir, f'comparison_{args.task}.npy'), all_results)
    
    return all_results


if __name__ == '__main__':
    args = get_args()
    
    # Generate dataset if needed
    task_file = f'data/correction_ultimate_{args.task}.npz'
    if not os.path.exists(task_file):
        print(f"Generating {args.task} dataset...")
        import subprocess
        subprocess.run(['python', 'data/correction_ultimate.py'])
    
    if args.compare:
        run_comparison(args)
    else:
        train(args)
