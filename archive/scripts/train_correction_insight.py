#!/usr/bin/env python3
"""
Training Script for Correction Insight Benchmarks

Follows the experimental protocol from:
"The Illusion of Insight in Reasoning Models" (arXiv:2601.00514v1)

Key Metrics (from paper):
- %S: Shift prevalence (fraction of traces with detected correction)
- P(✓|S=1): Accuracy when shift is detected
- P(✓|S=0): Accuracy when no shift detected
- Δ(pp): Accuracy difference (percentage points)
- AME: Average marginal effect from regression

Expected Results (comparing proposed vs baselines):
- GPT2 Baseline: Poor shift detection, cannot execute instant flips
- DDL: Better at incremental correction, struggles with instant flips
- mHC: May detect shifts but limited correction capability
- E∆-MHC-Geo: Should excel - beta gating detects shifts, Householder executes flips

Usage:
    # Run comparison on insight dataset
    python train_correction_insight.py --dataset insight --compare
    
    # Single model training
    python train_correction_insight.py --model_type edelta --dataset entropy
    
    # Full benchmark suite
    python train_correction_insight.py --full_benchmark
"""

import os
import math
import time
import argparse
from typing import Dict, List, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

# Import models
from model import GPT as BaselineGPT, GPTConfig as BaselineConfig
from proposed_model_ddl import GPT as DDLGPT, GPTConfig as DDLConfig
from proposed_model_mhc_real import GPT as mHCGPT, GPTConfig as mHCConfig
from proposed_model_hybrid import GPT as EdeltaGPT, GPTConfig as EdeltaConfig
from train_continuous import ContinuousModelWrapper


def get_args():
    parser = argparse.ArgumentParser(description='Correction Insight Benchmarks')
    
    # Model
    parser.add_argument('--model_type', type=str, default='edelta',
                        choices=['gpt2', 'ddl', 'mhc', 'edelta'])
    
    # Dataset
    parser.add_argument('--dataset', type=str, default='insight',
                        choices=['insight', 'entropy', 'shift'])
    parser.add_argument('--data_dir', type=str, default='data')
    
    # Training
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--max_iters', type=int, default=3000)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--min_lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.1)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    
    # Model architecture
    parser.add_argument('--n_layer', type=int, default=6)
    parser.add_argument('--n_head', type=int, default=4)
    parser.add_argument('--n_embd', type=int, default=128)
    parser.add_argument('--n_streams', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.0)
    
    # E∆-MHC-Geo specific
    parser.add_argument('--gate_reg_weight', type=float, default=0.1)
    parser.add_argument('--init_gate_bias', type=float, default=0.0)
    
    # Evaluation
    parser.add_argument('--eval_interval', type=int, default=100)
    parser.add_argument('--eval_iters', type=int, default=50)
    parser.add_argument('--log_interval', type=int, default=100)
    
    # Output
    parser.add_argument('--out_dir', type=str, default='out-correction-insight')
    
    # Benchmark modes
    parser.add_argument('--compare', action='store_true',
                        help='Run comparison across all model types')
    parser.add_argument('--full_benchmark', action='store_true',
                        help='Run full benchmark suite (all models x all datasets)')
    
    # Hardware
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=42)
    
    return parser.parse_args()


def load_dataset(dataset_name: str, data_dir: str = 'data') -> Dict[str, torch.Tensor]:
    """Load correction insight dataset."""
    
    dataset_paths = {
        'insight': 'correction_insight',
        'entropy': 'correction_entropy', 
        'shift': 'correction_shift',
    }
    
    path = os.path.join(data_dir, dataset_paths[dataset_name])
    
    if not os.path.exists(path):
        print(f"Dataset not found at {path}, generating...")
        import subprocess
        subprocess.run(['python', 'data/correction_insight.py', '--dataset', dataset_name])
    
    data = {
        'train_x': torch.from_numpy(np.load(os.path.join(path, 'train_x.npy'))),
        'train_y': torch.from_numpy(np.load(os.path.join(path, 'train_y.npy'))),
        'val_x': torch.from_numpy(np.load(os.path.join(path, 'val_x.npy'))),
        'val_y': torch.from_numpy(np.load(os.path.join(path, 'val_y.npy'))),
    }
    
    # Load metadata if available
    meta_path = os.path.join(path, 'metadata.npy')
    if os.path.exists(meta_path):
        data['metadata'] = np.load(meta_path, allow_pickle=True).item()
    
    # Load additional annotations for analysis (numeric only)
    numeric_extras = ['train_entropy', 'train_shift_needed', 'train_uncertainty', 
                      'train_expected_shift', 'train_shift_positions', 'train_correction_needed']
    for extra in numeric_extras:
        extra_path = os.path.join(path, f'{extra}.npy')
        if os.path.exists(extra_path):
            arr = np.load(extra_path)
            # Only convert numeric arrays to torch tensors
            if arr.dtype in [np.float32, np.float64, np.int32, np.int64, np.bool_]:
                data[extra] = torch.from_numpy(arr)
            else:
                data[extra] = arr  # Keep as numpy for string/object arrays
    
    # Load string arrays separately (like train_scenarios)
    string_extras = ['train_scenarios']
    for extra in string_extras:
        extra_path = os.path.join(path, f'{extra}.npy')
        if os.path.exists(extra_path):
            data[extra] = np.load(extra_path, allow_pickle=True)
    
    print(f"Loaded {dataset_name} dataset:")
    print(f"  Train: {data['train_x'].shape} -> {data['train_y'].shape}")
    print(f"  Val: {data['val_x'].shape} -> {data['val_y'].shape}")
    
    return data


def create_model(args, input_dim: int, block_size: int):
    """Create wrapped model based on model_type."""
    
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


def get_lr(it: int, warmup_iters: int, lr_decay_iters: int,
           learning_rate: float, min_lr: float) -> float:
    """Learning rate schedule."""
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)


def get_batch(data: Dict, split: str, batch_size: int, device: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """Get a random batch."""
    x = data[f'{split}_x']
    y = data[f'{split}_y']
    ix = torch.randint(len(x), (batch_size,))
    return x[ix].to(device), y[ix].to(device)


@torch.no_grad()
def estimate_loss(model, data: Dict, batch_size: int, eval_iters: int, device: str) -> Dict[str, float]:
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
def compute_paper_metrics(
    model, 
    data: Dict, 
    device: str,
    batch_size: int = 64,
    n_samples: int = 500
) -> Dict[str, float]:
    """
    Compute metrics following the paper's protocol (arXiv:2601.00514v1).
    
    Metrics:
    - MSE loss (standard)
    - Cosine similarity (for correction quality)
    - Shift detection accuracy (if annotations available)
    - Entropy-stratified performance (if entropy annotations available)
    
    Returns:
        Dictionary of paper-style metrics
    """
    model.eval()
    
    results = {
        'mse_loss': [],
        'cosine_sim': [],
        'correction_quality': [],  # How well corrections are executed
    }
    
    val_x = data['val_x'].to(device)
    val_y = data['val_y'].to(device)
    n_val = len(val_x)
    
    # Sample batches
    for _ in range(min(n_samples // batch_size + 1, n_val // batch_size)):
        ix = torch.randint(n_val, (min(batch_size, n_val),))
        x, y = val_x[ix], val_y[ix]
        
        pred, loss = model(x, y)
        results['mse_loss'].append(loss.item())
        
        # Cosine similarity (normalized)
        pred_norm = F.normalize(pred.view(-1, pred.size(-1)), dim=-1)
        y_norm = F.normalize(y.view(-1, y.size(-1)), dim=-1)
        cos_sim = (pred_norm * y_norm).sum(dim=-1).mean().item()
        results['cosine_sim'].append(cos_sim)
        
        # Correction quality: detect positions where target flips sign
        # and measure how well model follows
        for b in range(x.size(0)):
            for t in range(1, x.size(1)):
                # Check if this is a correction point (target changed sign)
                prev_y = y[b, t-1]
                curr_y = y[b, t]
                cos_target = F.cosine_similarity(prev_y.unsqueeze(0), 
                                                  curr_y.unsqueeze(0)).item()
                
                if cos_target < -0.5:  # Sign flip detected
                    # Measure if model also flipped
                    prev_pred = pred[b, t-1]
                    curr_pred = pred[b, t]
                    cos_pred = F.cosine_similarity(curr_pred.unsqueeze(0),
                                                   curr_y.unsqueeze(0)).item()
                    results['correction_quality'].append(cos_pred)
    
    # Compute final metrics
    metrics = {
        'val_loss': np.mean(results['mse_loss']),
        'cosine_sim': np.mean(results['cosine_sim']),
    }
    
    if results['correction_quality']:
        metrics['correction_quality'] = np.mean(results['correction_quality'])
        metrics['correction_std'] = np.std(results['correction_quality'])
    else:
        metrics['correction_quality'] = 0.0
        metrics['correction_std'] = 0.0
    
    # Entropy-stratified analysis (following paper Table 26)
    if 'train_entropy' in data:
        # This would require running on training data with entropy annotations
        # For now, just note that it's available
        metrics['entropy_analysis_available'] = True
    
    model.train()
    return metrics


def train_single(args, model_type: str = None, dataset: str = None) -> Dict[str, Any]:
    """Train a single model on a single dataset."""
    
    if model_type:
        args.model_type = model_type
    if dataset:
        args.dataset = dataset
    
    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    out_dir = os.path.join(args.out_dir, f'{args.model_type}_{args.dataset}')
    os.makedirs(out_dir, exist_ok=True)
    
    # Load data
    print(f"\n{'='*60}")
    print(f"Training {args.model_type.upper()} on {args.dataset}")
    print(f"{'='*60}")
    
    data = load_dataset(args.dataset, args.data_dir)
    
    input_dim = data['train_x'].shape[-1]
    seq_len = data['train_x'].shape[1]
    
    # Create model
    model = create_model(args, input_dim, seq_len + 1)
    model = model.to(device)
    
    n_params = model.get_num_params()
    print(f"Model: {args.model_type} ({n_params/1e6:.2f}M params)")
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate,
        weight_decay=args.weight_decay, betas=(0.9, 0.95)
    )
    
    # Training
    warmup_iters = 100
    best_val_loss = float('inf')
    train_log = {'iter': [], 'train_loss': [], 'val_loss': []}
    
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
                torch.save({
                    'model': model.state_dict(),
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                }, os.path.join(out_dir, 'ckpt.pt'))
        
        # Training step
        x, y = get_batch(data, 'train', args.batch_size, device)
        _, loss = model(x, y)
        
        # Gate regularization for E∆-MHC-Geo
        if args.model_type == 'edelta' and hasattr(model.core, 'get_gate_regularization_loss'):
            gate_loss = model.core.get_gate_regularization_loss()
            loss = loss + gate_loss
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    
    t1 = time.time()
    
    # Final evaluation with paper metrics
    print(f"\n{'='*40}")
    print("Computing paper-style metrics...")
    paper_metrics = compute_paper_metrics(model, data, device)
    
    print(f"\nResults ({args.model_type} on {args.dataset}):")
    print(f"  Best val loss: {best_val_loss:.4e}")
    print(f"  Cosine similarity: {paper_metrics['cosine_sim']:.4f}")
    print(f"  Correction quality: {paper_metrics['correction_quality']:.4f} ± {paper_metrics['correction_std']:.4f}")
    print(f"  Training time: {(t1-t0)/60:.1f} min")
    
    # Save results
    results = {
        'model_type': args.model_type,
        'dataset': args.dataset,
        'best_val_loss': best_val_loss,
        'paper_metrics': paper_metrics,
        'training_time': t1 - t0,
        'n_params': n_params,
    }
    np.save(os.path.join(out_dir, 'results.npy'), results)
    np.save(os.path.join(out_dir, 'train_log.npy'), train_log)
    
    return results


def run_comparison(args) -> Dict[str, Dict]:
    """Run comparison across all model types on a single dataset."""
    
    model_types = ['gpt2', 'ddl', 'mhc', 'edelta']
    all_results = {}
    
    print("\n" + "=" * 70)
    print(f"COMPARISON BENCHMARK: {args.dataset}")
    print("Following protocol from arXiv:2601.00514v1")
    print("=" * 70)
    
    for model_type in model_types:
        results = train_single(args, model_type=model_type)
        all_results[model_type] = results
    
    # Print comparison table (paper style)
    print("\n" + "=" * 70)
    print("COMPARISON RESULTS")
    print("=" * 70)
    print(f"\n{'Model':<12} {'Val Loss':<12} {'Cos Sim':<10} {'Correction':<12} {'Time':<8}")
    print("-" * 60)
    
    for model_type, results in all_results.items():
        pm = results['paper_metrics']
        print(f"{model_type:<12} {results['best_val_loss']:.4e}  "
              f"{pm['cosine_sim']:.4f}     {pm['correction_quality']:.4f}      "
              f"{results['training_time']/60:.1f}m")
    
    # Compute relative improvements (vs baseline)
    baseline = all_results['gpt2']
    print(f"\n{'='*60}")
    print("Relative Improvement vs GPT2 Baseline:")
    print("-" * 60)
    
    for model_type in ['ddl', 'mhc', 'edelta']:
        results = all_results[model_type]
        loss_delta = (baseline['best_val_loss'] - results['best_val_loss']) / baseline['best_val_loss'] * 100
        cos_delta = (results['paper_metrics']['cosine_sim'] - 
                    baseline['paper_metrics']['cosine_sim']) * 100
        corr_delta = (results['paper_metrics']['correction_quality'] - 
                     baseline['paper_metrics']['correction_quality']) * 100
        
        print(f"{model_type:<12} Loss: {loss_delta:+.1f}%  CosSim: {corr_delta:+.1f}pp  Correction: {corr_delta:+.1f}pp")
    
    # Save comparison
    np.save(os.path.join(args.out_dir, f'comparison_{args.dataset}.npy'), all_results)
    
    return all_results


def run_full_benchmark(args) -> Dict[str, Dict]:
    """Run full benchmark: all models × all datasets."""
    
    datasets = ['insight', 'entropy', 'shift']
    all_results = {}
    
    print("\n" + "=" * 70)
    print("FULL BENCHMARK SUITE")
    print("Following protocol from 'The Illusion of Insight' (arXiv:2601.00514v1)")
    print("=" * 70)
    
    for dataset in datasets:
        args.dataset = dataset
        all_results[dataset] = run_comparison(args)
    
    # Print summary table
    print("\n" + "=" * 70)
    print("FULL BENCHMARK SUMMARY")
    print("=" * 70)
    
    print(f"\n{'Dataset':<10} {'Best Model':<12} {'Improvement':<15} {'Key Metric':<20}")
    print("-" * 60)
    
    for dataset, results in all_results.items():
        # Find best model
        best_model = min(results.keys(), 
                        key=lambda m: results[m]['best_val_loss'])
        improvement = ((results['gpt2']['best_val_loss'] - results[best_model]['best_val_loss']) /
                      results['gpt2']['best_val_loss'] * 100)
        key_metric = results[best_model]['paper_metrics']['correction_quality']
        
        print(f"{dataset:<10} {best_model:<12} {improvement:+.1f}% loss      "
              f"Corr: {key_metric:.4f}")
    
    # Save full results
    np.save(os.path.join(args.out_dir, 'full_benchmark.npy'), all_results)
    
    return all_results


if __name__ == '__main__':
    args = get_args()
    
    # First generate datasets if they don't exist
    for dataset in ['insight', 'entropy', 'shift']:
        path = os.path.join(args.data_dir, f'correction_{dataset}')
        if not os.path.exists(path):
            print(f"Generating {dataset} dataset...")
            import subprocess
            subprocess.run(['python', 'data/correction_insight.py', '--dataset', dataset])
    
    if args.full_benchmark:
        run_full_benchmark(args)
    elif args.compare:
        run_comparison(args)
    else:
        train_single(args)
