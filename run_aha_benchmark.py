#!/usr/bin/env python3
"""
"Aha!" Moment Benchmark: Tests Correction/Reflection Capability

Based on "The Illusion of Insight in Reasoning Models" (arXiv:2601.00514v1)

Models Compared:
1. GPT2 (Naive Baseline) - Standard transformer, no geometric bias
2. DDL (Deep Delta Learning, arXiv:2601.00417) - Rank-1 perturbation
3. mHC (DeepSeek mHC, arXiv:2512.24880) - Doubly stochastic mixing
4. Proposed (E∆-MHC-Geo) - Householder + Cayley hybrid

Key Test: Can the model do INSTANT COMPLETE BELIEF FLIPS (v → -v)?
This requires REFLECTION (det=-1), which:
- DDL can do via β→2 in A = I - β·kk^T
- Proposed can do via Householder H = I - 2kk^T
- mHC uses doubly stochastic (limited reflection capability)
- GPT2 has no geometric inductive bias

NOTE: The "pure_flip" task tests LEARNED conditional flipping, which any
expressive network can approximate. For proper geometric advantage testing,
use --test_generalization to test zero-shot transfer to unseen directions.
"""

import os
import sys
import time
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, Any

# Force unbuffered output for terminal file visibility
sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None
os.environ['PYTHONUNBUFFERED'] = '1'

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model import GPT as GPT2, GPTConfig as GPT2Config
from proposed_model_ddl import GPT as DDLGPT, GPTConfig as DDLConfig
from proposed_model_mhc_real import GPT as mHCGPT, GPTConfig as mHCConfig
from proposed_model_hybrid import GPT as ProposedGPT, GPTConfig as ProposedConfig
from train_continuous import ContinuousModelWrapper


MODEL_NAMES = {
    'gpt2': 'GPT2 (Baseline)',
    'ddl': 'DDL',
    'mhc': 'DeepSeek mHC',
    'proposed': 'Proposed (E∆)',
}


def flush_print(*args, **kwargs):
    """Print with immediate flush for terminal file visibility."""
    print(*args, **kwargs)
    sys.stdout.flush()


def get_args():
    parser = argparse.ArgumentParser(description='"Aha!" Moment Benchmark')
    
    parser.add_argument('--dataset', type=str, default='pure_flip',
                        choices=['aha', 'pure_flip', 'entropy', 'generalization'])
    parser.add_argument('--model', type=str, default='all',
                        choices=['all', 'gpt2', 'ddl', 'mhc', 'proposed'])
    
    # Training
    parser.add_argument('--max_iters', type=int, default=3000)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--eval_interval', type=int, default=200)
    
    # Model
    parser.add_argument('--n_layer', type=int, default=6)
    parser.add_argument('--n_head', type=int, default=4)
    parser.add_argument('--n_embd', type=int, default=128)
    
    # Generalization test
    parser.add_argument('--test_generalization', action='store_true',
                        help='Test zero-shot generalization to unseen reflection directions')
    parser.add_argument('--few_shot', type=int, default=None,
                        help='Test sample efficiency with limited training data')
    
    # Hardware
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--out_dir', type=str, default='out-aha-benchmark')
    
    return parser.parse_args()


def load_dataset(dataset_name: str):
    """Load correction dataset."""
    dataset_paths = {
        'aha': 'data/correction_aha',
        'pure_flip': 'data/correction_pure_flip',
        'entropy': 'data/correction_entropy_flip',
        'generalization': 'data/correction_generalization',
    }
    
    path = dataset_paths[dataset_name]
    
    if not os.path.exists(path):
        flush_print(f"Dataset not found at {path}, generating...")
        if dataset_name == 'generalization':
            generate_generalization_dataset(path)
        else:
            import subprocess
            subprocess.run(['python', 'data/correction_aha.py', '--dataset', dataset_name])
    
    data = {
        'train_x': torch.from_numpy(np.load(f'{path}/train_x.npy')),
        'train_y': torch.from_numpy(np.load(f'{path}/train_y.npy')),
        'val_x': torch.from_numpy(np.load(f'{path}/val_x.npy')),
        'val_y': torch.from_numpy(np.load(f'{path}/val_y.npy')),
    }
    
    # Load metadata if available
    if os.path.exists(f'{path}/val_scenarios.npy'):
        data['val_scenarios'] = np.load(f'{path}/val_scenarios.npy', allow_pickle=True)
    if os.path.exists(f'{path}/val_flip_positions.npy'):
        data['val_flip_positions'] = np.load(f'{path}/val_flip_positions.npy')
    if os.path.exists(f'{path}/val_entropy.npy'):
        data['val_entropy'] = np.load(f'{path}/val_entropy.npy')
    
    return data


def create_model(model_type: str, args, input_dim: int, block_size: int):
    """Create model based on type."""
    
    if model_type == 'gpt2':
        config = GPT2Config(
            n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd,
            dropout=0.0, bias=False, block_size=block_size, vocab_size=1
        )
        core = GPT2(config)
        
    elif model_type == 'ddl':
        config = DDLConfig(
            n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd,
            dropout=0.0, bias=False, block_size=block_size, vocab_size=1
        )
        core = DDLGPT(config)
        
    elif model_type == 'mhc':
        config = mHCConfig(
            n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd,
            n_streams=4, dropout=0.0, bias=False, block_size=block_size, vocab_size=1
        )
        core = mHCGPT(config)
        
    elif model_type == 'proposed':
        config = ProposedConfig(
            n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd,
            n_streams=4, dropout=0.0, bias=False, block_size=block_size, vocab_size=1,
            gate_reg_weight=0.1, init_gate_bias=-1.0  # Bias toward Householder for flips
        )
        core = ProposedGPT(config)
    
    model = ContinuousModelWrapper(core, config, input_dim, block_size)
    return model


@torch.no_grad()
def evaluate_flip_metrics(model, data, device, n_samples=None):
    """
    Evaluate with metrics specific to "Aha!" moment / flip capability.
    
    Key metrics:
    1. val_loss: Standard MSE loss
    2. cosine_sim: Overall cosine similarity
    3. flip_accuracy: Cosine similarity at FLIP positions (v → -v)
    4. flip_detection: Did the model detect and respond to flip signal?
    """
    model.eval()
    
    val_x = data['val_x']
    val_y = data['val_y']
    
    if n_samples is not None:
        val_x = val_x[:n_samples]
        val_y = val_y[:n_samples]
    
    val_x = val_x.to(device)
    val_y = val_y.to(device)
    
    # Forward pass
    pred, loss = model(val_x, val_y)
    
    # Overall cosine similarity
    pred_flat = pred.view(-1, pred.size(-1))
    y_flat = val_y.view(-1, val_y.size(-1))
    pred_norm = F.normalize(pred_flat, dim=-1)
    y_norm = F.normalize(y_flat, dim=-1)
    cos_sim = (pred_norm * y_norm).sum(dim=-1).mean().item()
    
    # Flip accuracy: measure at positions where target flips sign
    flip_scores = []
    non_flip_scores = []
    
    for b in range(min(200, val_x.size(0))):
        for t in range(1, val_x.size(1)):
            prev_y = val_y[b, t-1]
            curr_y = val_y[b, t]
            
            # Check if this is a flip position (target reverses sign)
            target_cos = F.cosine_similarity(prev_y.unsqueeze(0), curr_y.unsqueeze(0)).item()
            
            if target_cos < -0.9:  # This is a flip (v → -v)
                pred_cos = F.cosine_similarity(pred[b, t].unsqueeze(0), curr_y.unsqueeze(0)).item()
                flip_scores.append(pred_cos)
            elif abs(target_cos - 1.0) < 0.1:  # Not a flip (maintain)
                pred_cos = F.cosine_similarity(pred[b, t].unsqueeze(0), curr_y.unsqueeze(0)).item()
                non_flip_scores.append(pred_cos)
    
    metrics = {
        'val_loss': loss.item(),
        'cosine_sim': cos_sim,
        'flip_accuracy': np.mean(flip_scores) if flip_scores else 0.0,
        'flip_std': np.std(flip_scores) if len(flip_scores) > 1 else 0.0,
        'maintain_accuracy': np.mean(non_flip_scores) if non_flip_scores else 0.0,
        'n_flips': len(flip_scores),
        'n_maintains': len(non_flip_scores),
    }
    
    model.train()
    return metrics


def generate_generalization_dataset(output_dir: str = 'data/correction_generalization', dim: int = 32):
    """
    Generate dataset that tests ZERO-SHOT GENERALIZATION to unseen reflection directions.
    
    THIS IS THE PROPER TEST for geometric inductive bias (DDL paper arXiv:2601.00417):
    - Train on reflections in dimensions [0, dim//2)
    - Test on reflections in dimensions [dim//2, dim)
    
    Expected Results:
    - DDL/Proposed: Should generalize (geometric structure is universal)
    - GPT2: Should FAIL (learned f(x) ≈ -2x is direction-specific)
    """
    os.makedirs(output_dir, exist_ok=True)
    np.random.seed(42)
    
    seq_len = 48
    n_train, n_val = 6000, 750
    
    flush_print("="*70)
    flush_print("GENERATING GENERALIZATION TEST DATASET")
    flush_print("Train: reflections in dims [0, dim//2)")
    flush_print("Test:  reflections in dims [dim//2, dim) (UNSEEN)")
    flush_print("="*70)
    
    def make_samples(n, dim_range, noise_level=0.1):
        data_x, data_y = [], []
        for _ in range(n):
            # Belief restricted to specific dimensions
            belief = np.zeros(dim, dtype=np.float32)
            active_dims = list(range(dim_range[0], dim_range[1]))
            for d in active_dims:
                belief[d] = np.random.randn()
            belief = belief / (np.linalg.norm(belief) + 1e-8)
            
            signal_pos = np.random.randint(seq_len // 3, 2 * seq_len // 3)
            
            seq_x = np.zeros((seq_len, dim), dtype=np.float32)
            seq_y = np.zeros((seq_len, dim), dtype=np.float32)
            
            signal = np.zeros(dim, dtype=np.float32)
            signal[0] = 5.0
            
            for t in range(seq_len):
                noise = np.random.randn(dim).astype(np.float32) * noise_level
                if t < signal_pos:
                    seq_x[t] = belief + noise
                    seq_y[t] = belief
                elif t == signal_pos:
                    seq_x[t] = signal
                    seq_y[t] = -belief  # FLIP
                else:
                    seq_x[t] = -belief + noise
                    seq_y[t] = -belief
            
            data_x.append(seq_x)
            data_y.append(seq_y)
        return np.stack(data_x), np.stack(data_y)
    
    # Train on first half of dimensions
    train_x, train_y = make_samples(n_train, (0, dim // 2))
    
    # Test on SECOND half of dimensions (unseen!)
    val_x, val_y = make_samples(n_val, (dim // 2, dim))
    
    np.save(os.path.join(output_dir, 'train_x.npy'), train_x)
    np.save(os.path.join(output_dir, 'train_y.npy'), train_y)
    np.save(os.path.join(output_dir, 'val_x.npy'), val_x)
    np.save(os.path.join(output_dir, 'val_y.npy'), val_y)
    
    flush_print(f"\nTrain shapes: {train_x.shape}, Test shapes: {val_x.shape}")
    flush_print(f"Saved to: {output_dir}/")
    
    return output_dir


def train_model(model_type: str, dataset_name: str, args) -> Dict[str, Any]:
    """Train a single model on the correction dataset."""
    
    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'
    
    flush_print(f"\n{'='*60}")
    flush_print(f"Training {MODEL_NAMES[model_type]} on {dataset_name}")
    flush_print(f"{'='*60}")
    
    # Load data
    data = load_dataset(dataset_name)
    input_dim = data['train_x'].shape[-1]
    seq_len = data['train_x'].shape[1]
    
    # Few-shot mode
    if args.few_shot is not None:
        flush_print(f"FEW-SHOT MODE: Using only {args.few_shot} training samples")
        data['train_x'] = data['train_x'][:args.few_shot]
        data['train_y'] = data['train_y'][:args.few_shot]
    
    # Create model
    model = create_model(model_type, args, input_dim, seq_len + 1)
    model = model.to(device)
    n_params = model.get_num_params()
    flush_print(f"Parameters: {n_params/1e6:.2f}M")
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.1)
    
    # Training
    best_flip_acc = -1.0
    best_val_loss = float('inf')
    learning_curve = []
    t0 = time.time()
    
    for iter_num in range(args.max_iters):
        # LR schedule (cosine decay)
        lr = args.learning_rate * 0.5 * (1 + np.cos(np.pi * iter_num / args.max_iters))
        for pg in optimizer.param_groups:
            pg['lr'] = lr
        
        # Evaluate periodically
        if iter_num % args.eval_interval == 0:
            metrics = evaluate_flip_metrics(model, data, device)
            flush_print(f"step {iter_num:>5}: loss {metrics['val_loss']:.4e}, "
                  f"cos {metrics['cosine_sim']:.4f}, "
                  f"flip_acc {metrics['flip_accuracy']:.4f} ({metrics['n_flips']} flips)")
            
            learning_curve.append({
                'iter': iter_num,
                'val_loss': metrics['val_loss'],
                'flip_accuracy': metrics['flip_accuracy']
            })
            
            if metrics['flip_accuracy'] > best_flip_acc:
                best_flip_acc = metrics['flip_accuracy']
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
    final_metrics = evaluate_flip_metrics(model, data, device)
    
    results = {
        'model_type': model_type,
        'model_name': MODEL_NAMES[model_type],
        'dataset': dataset_name,
        'best_val_loss': best_val_loss,
        'best_flip_acc': best_flip_acc,
        'final_metrics': final_metrics,
        'training_time': t1 - t0,
        'n_params': n_params,
        'learning_curve': learning_curve,
    }
    
    # Save
    out_path = os.path.join(args.out_dir, f'{model_type}_{dataset_name}')
    os.makedirs(out_path, exist_ok=True)
    np.save(os.path.join(out_path, 'results.npy'), results)
    torch.save(model.state_dict(), os.path.join(out_path, 'model.pt'))
    
    return results


def run_benchmark(args):
    """Run full benchmark."""
    
    models = ['gpt2', 'ddl', 'mhc', 'proposed'] if args.model == 'all' else [args.model]
    
    # Generate dataset
    flush_print("Generating dataset...")
    if args.dataset == 'generalization' or args.test_generalization:
        args.dataset = 'generalization'
        generate_generalization_dataset()
    else:
        import subprocess
        subprocess.run(['python', 'data/correction_aha.py', '--dataset', args.dataset])
    
    results = {}
    for model_type in models:
        results[model_type] = train_model(model_type, args.dataset, args)
    
    # Print summary
    flush_print("\n" + "="*80)
    flush_print('"AHA!" MOMENT BENCHMARK RESULTS')
    flush_print("="*80)
    flush_print(f"\nDataset: {args.dataset}")
    flush_print(f"Training iterations: {args.max_iters}")
    if args.few_shot:
        flush_print(f"Few-shot training samples: {args.few_shot}")
    
    flush_print("\n" + "-"*80)
    flush_print(f"{'Model':<20} {'Val Loss':<12} {'Cos Sim':<10} {'Flip Acc':<12} {'Params'}")
    flush_print("-"*80)
    
    for model_type in models:
        r = results[model_type]
        fm = r['final_metrics']
        flush_print(f"{r['model_name']:<20} {r['best_val_loss']:.4e}   "
              f"{fm['cosine_sim']:.4f}     {fm['flip_accuracy']:.4f}       "
              f"{r['n_params']/1e6:.2f}M")
    
    flush_print("-"*80)
    
    # Analysis
    flush_print("\n" + "="*80)
    flush_print("ANALYSIS: Reflection / 'Aha!' Moment Capability")
    flush_print("="*80)
    
    ddl_flip = results['ddl']['final_metrics']['flip_accuracy']
    proposed_flip = results['proposed']['final_metrics']['flip_accuracy']
    mhc_flip = results['mhc']['final_metrics']['flip_accuracy']
    gpt2_flip = results['gpt2']['final_metrics']['flip_accuracy']
    
    flush_print(f"\nFlip Accuracy (higher = better at 'Aha!' moments):")
    flush_print(f"  DDL:      {ddl_flip:.4f} (rank-1 perturbation, β→2 for reflection)")
    flush_print(f"  Proposed: {proposed_flip:.4f} (Householder H = I - 2kk^T)")
    flush_print(f"  mHC:      {mhc_flip:.4f} (doubly stochastic mixing)")
    flush_print(f"  GPT2:     {gpt2_flip:.4f} (standard residual, no geometric bias)")
    
    # Winner analysis
    flip_scores = [(ddl_flip, 'DDL'), (proposed_flip, 'Proposed'), 
                   (mhc_flip, 'mHC'), (gpt2_flip, 'GPT2')]
    flip_scores.sort(reverse=True)
    
    flush_print(f"\nRanking (by flip accuracy):")
    for i, (score, name) in enumerate(flip_scores, 1):
        flush_print(f"  {i}. {name}: {score:.4f}")
    
    winner = flip_scores[0][1]
    flush_print(f"\n✓ Best model for 'Aha!' moments: {winner}")
    
    if proposed_flip > mhc_flip:
        flush_print(f"  Proposed outperforms mHC by {100*(proposed_flip-mhc_flip):.1f}%")
    
    # Special analysis for generalization test
    if args.dataset == 'generalization':
        flush_print("\n" + "="*80)
        flush_print("GENERALIZATION TEST ANALYSIS")
        flush_print("="*80)
        flush_print("\nThis test trains on dims [0, d//2) and tests on dims [d//2, d)")
        flush_print("Key insight: Geometric models should generalize, GPT2 should NOT")
        flush_print("")
        
        geo_avg = (ddl_flip + proposed_flip) / 2
        
        if gpt2_flip < 0.5 and geo_avg > 0.7:
            flush_print("✓ CONFIRMED: Geometric inductive bias enables generalization!")
            flush_print(f"  GPT2 (learned): {gpt2_flip:.4f} (FAILED on unseen directions)")
            flush_print(f"  DDL/Proposed:   {geo_avg:.4f} (SUCCEEDED on unseen directions)")
        elif gpt2_flip > 0.9:
            flush_print("⚠ GPT2 also generalizes well - task may be too easy")
            flush_print("  Consider: deeper networks, harder distribution shift")
        else:
            flush_print(f"  Results: GPT2={gpt2_flip:.4f}, DDL={ddl_flip:.4f}, Proposed={proposed_flip:.4f}")
    
    # Save results
    np.save(os.path.join(args.out_dir, 'benchmark_results.npy'), results)
    flush_print(f"\nResults saved to: {args.out_dir}/")
    
    return results


if __name__ == '__main__':
    args = get_args()
    os.makedirs(args.out_dir, exist_ok=True)
    run_benchmark(args)
