#!/usr/bin/env python3
"""
Run a single dataset comparison across all models with unified config.

Usage:
    python run_single_comparison.py --dataset=rotation3d
    python run_single_comparison.py --dataset=correction --max_iters=5000
"""

import os
import sys
import argparse
import subprocess
import time
from datetime import datetime

# Unified hyperparameters (same for all models)
UNIFIED_CONFIG = {
    'n_layer': 4,
    'n_head': 4,
    'n_embd': 128,
    'batch_size': 64,
    'block_size': 128,
    'learning_rate': 1e-3,
    'weight_decay': 0.1,
    'dropout': 0.0,
    'gradient_accumulation_steps': 1,
    'warmup_iters': 500,
    'eval_interval': 500,
    'log_interval': 100,
    'eval_iters': 50,
}

MODELS = {
    'baseline': {
        'script': 'train.py',
        'name': 'Baseline Transformer',
    },
    'edelta': {
        'script': 'train_geodesic.py',
        'name': 'E∆-MHC-Geo (Proposed)',
    },
    'mhc_real': {
        'script': 'train_mhc_real.py',
        'name': 'Pure mHC (Sinkhorn)',
    },
    'ddl': {
        'script': 'train_ddl.py',
        'name': 'Pure DDL (Householder)',
    },
}


def run_model(model_key, dataset, max_iters, compile_model=False):
    """Run a single model on a dataset."""
    model = MODELS[model_key]
    out_dir = f"out-unified-{model_key}-{dataset}"
    
    print(f"\n{'='*60}")
    print(f"Running: {model['name']}")
    print(f"Dataset: {dataset}")
    print(f"Output: {out_dir}")
    print(f"{'='*60}")
    
    # Build command with unified config
    cmd = [
        sys.executable, model['script'],
        f"--dataset={dataset}",
        f"--out_dir={out_dir}",
        f"--max_iters={max_iters}",
        f"--compile={'true' if compile_model else 'false'}",
        f"--eval_interval={UNIFIED_CONFIG['eval_interval']}",
        f"--log_interval={UNIFIED_CONFIG['log_interval']}",
        f"--n_layer={UNIFIED_CONFIG['n_layer']}",
        f"--n_head={UNIFIED_CONFIG['n_head']}",
        f"--n_embd={UNIFIED_CONFIG['n_embd']}",
        f"--batch_size={UNIFIED_CONFIG['batch_size']}",
        f"--block_size={UNIFIED_CONFIG['block_size']}",
        f"--learning_rate={UNIFIED_CONFIG['learning_rate']}",
        f"--weight_decay={UNIFIED_CONFIG['weight_decay']}",
        f"--dropout={UNIFIED_CONFIG['dropout']}",
        f"--gradient_accumulation_steps={UNIFIED_CONFIG['gradient_accumulation_steps']}",
        f"--warmup_iters={UNIFIED_CONFIG['warmup_iters']}",
    ]
    
    start_time = time.time()
    
    # Run and capture output
    log_file = f"{out_dir}.log"
    with open(log_file, 'w') as f:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        final_val_loss = None
        for line in process.stdout:
            print(line, end='')
            f.write(line)
            
            # Extract val loss
            if 'val loss' in line:
                try:
                    parts = line.split('val loss')
                    if len(parts) > 1:
                        final_val_loss = float(parts[1].split()[0].strip(','))
                except:
                    pass
        
        process.wait()
    
    elapsed = time.time() - start_time
    
    return {
        'model': model_key,
        'name': model['name'],
        'dataset': dataset,
        'val_loss': final_val_loss,
        'time': elapsed,
        'success': process.returncode == 0,
    }


def main():
    parser = argparse.ArgumentParser(description='Run unified comparison experiments')
    parser.add_argument('--dataset', type=str, required=True, 
                        help='Dataset to test (rotation2d, rotation3d, correction, etc.)')
    parser.add_argument('--max_iters', type=int, default=10000,
                        help='Maximum training iterations')
    parser.add_argument('--compile', action='store_true',
                        help='Use torch.compile (slower startup, faster training)')
    parser.add_argument('--models', type=str, nargs='+', 
                        default=['baseline', 'edelta', 'mhc_real', 'ddl'],
                        help='Models to run')
    
    args = parser.parse_args()
    
    print("="*60)
    print("UNIFIED COMPARISON EXPERIMENT")
    print("="*60)
    print(f"Dataset: {args.dataset}")
    print(f"Max iterations: {args.max_iters}")
    print(f"Models: {args.models}")
    print(f"Compile: {args.compile}")
    print()
    print("Unified Hyperparameters:")
    for k, v in UNIFIED_CONFIG.items():
        print(f"  {k}: {v}")
    print("="*60)
    
    results = []
    
    for model_key in args.models:
        if model_key not in MODELS:
            print(f"Unknown model: {model_key}, skipping...")
            continue
        
        result = run_model(model_key, args.dataset, args.max_iters, args.compile)
        results.append(result)
    
    # Print summary
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    print(f"Dataset: {args.dataset}")
    print()
    
    # Sort by val loss
    results.sort(key=lambda x: x['val_loss'] if x['val_loss'] else float('inf'))
    
    for i, r in enumerate(results):
        marker = "🏆" if i == 0 and r['success'] else "  "
        status = f"{r['val_loss']:.4f}" if r['val_loss'] else "FAILED"
        print(f"{marker} {r['name']}: {status} ({r['time']:.1f}s)")
    
    # Save results
    results_file = f"results_{args.dataset}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(results_file, 'w') as f:
        f.write(f"Dataset: {args.dataset}\n")
        f.write(f"Date: {datetime.now()}\n")
        f.write(f"Max iters: {args.max_iters}\n\n")
        for r in results:
            f.write(f"{r['name']}: {r['val_loss']}\n")
    
    print(f"\nResults saved to: {results_file}")


if __name__ == "__main__":
    main()
