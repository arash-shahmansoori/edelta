"""
Comparative Experiment Runner

This script runs comparative experiments across all models and datasets
to validate E∆-MHC-Geo's unified claims.

Models:
1. Baseline Transformer
2. Pure mHC (Sinkhorn-Knopp)
3. Pure DDL (Householder)
4. E∆-MHC-Geo (Your proposed method)

Datasets:
1. Deep Signal - Tests energy conservation (mHC claim)
2. Rotation3D - Tests geometric expressivity (DDL claim)
3. Correction - Tests Aha! moment detection (Insight claim)
4. Strategy Shift - Tests reasoning pivots (Insight claim)
5. Entropy Probe - Tests β-entropy correlation (Insight claim)

Usage:
    python run_comparative_experiments.py --dataset rotation3d --max_iters 5000
    python run_comparative_experiments.py --all  # Run all experiments
"""

import os
import sys
import argparse
import subprocess
import time
from datetime import datetime

# Experiment configurations
MODELS = {
    'baseline': {
        'script': 'train.py',
        'name': 'Baseline Transformer'
    },
    'mhc_real': {
        'script': 'train_mhc_real.py',
        'name': 'Pure mHC (Sinkhorn-Knopp)'
    },
    'ddl': {
        'script': 'train_ddl.py',
        'name': 'Pure DDL (Householder)'
    },
    'edelta': {
        'script': 'train_geodesic.py',
        'name': 'E∆-MHC-Geo (Proposed)'
    },
}

DATASETS = {
    'deep_signal': {
        'purpose': 'Test energy conservation (mHC claim)',
        'expected_winner': 'mHC = E∆ > DDL'
    },
    'rotation3d': {
        'purpose': 'Test geometric expressivity (DDL claim)',
        'expected_winner': 'E∆ > DDL > mHC'
    },
    'correction': {
        'purpose': 'Test Aha! moment detection (Insight claim)',
        'expected_winner': 'E∆ (has thermodynamic gating)'
    },
    'strategy_shift': {
        'purpose': 'Test reasoning pivots (Insight claim)',
        'expected_winner': 'E∆ (entropy-based gating)'
    },
    'entropy_probe': {
        'purpose': 'Test β-entropy correlation (Insight claim)',
        'expected_winner': 'E∆ (designed for this)'
    },
    'rotation2d': {
        'purpose': 'Test 2D geometric reasoning',
        'expected_winner': 'E∆ (Cayley on SO(2))'
    },
}


def prepare_dataset(dataset_name):
    """Run dataset preparation script."""
    prepare_script = f'data/{dataset_name}/prepare.py'
    if os.path.exists(prepare_script):
        print(f"\n{'='*60}")
        print(f"Preparing dataset: {dataset_name}")
        print(f"{'='*60}")
        subprocess.run([sys.executable, prepare_script], check=True)
        return True
    else:
        print(f"Warning: No prepare.py found for {dataset_name}")
        return False


def run_experiment(model_key, dataset_name, max_iters=5000, compile_model=False):
    """Run a single experiment."""
    model_info = MODELS[model_key]
    out_dir = f"out-{model_key}-{dataset_name}"
    
    print(f"\n{'='*60}")
    print(f"Running: {model_info['name']} on {dataset_name}")
    print(f"Output: {out_dir}")
    print(f"{'='*60}")
    
    cmd = [
        sys.executable, model_info['script'],
        f'--dataset={dataset_name}',
        f'--out_dir={out_dir}',
        f'--max_iters={max_iters}',
        f'--compile={str(compile_model).lower()}',
        '--eval_interval=500',
        '--log_interval=100',
    ]
    
    start_time = time.time()
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        elapsed = time.time() - start_time
        
        # Parse final loss from output
        output_lines = result.stdout.split('\n')
        final_val_loss = None
        for line in reversed(output_lines):
            if 'val loss' in line:
                parts = line.split('val loss')
                if len(parts) > 1:
                    try:
                        final_val_loss = float(parts[1].split()[0])
                        break
                    except:
                        pass
        
        return {
            'model': model_key,
            'dataset': dataset_name,
            'val_loss': final_val_loss,
            'time': elapsed,
            'success': result.returncode == 0,
            'stdout': result.stdout,
            'stderr': result.stderr,
        }
    
    except Exception as e:
        return {
            'model': model_key,
            'dataset': dataset_name,
            'val_loss': None,
            'time': 0,
            'success': False,
            'error': str(e),
        }


def run_all_experiments(datasets=None, models=None, max_iters=5000):
    """Run experiments for all model-dataset combinations."""
    if datasets is None:
        datasets = list(DATASETS.keys())
    if models is None:
        models = list(MODELS.keys())
    
    results = []
    
    # Prepare datasets first
    for dataset in datasets:
        if dataset not in ['rotation2d', 'grokking', 'erasure', 'isometry', 'reversibility']:
            prepare_dataset(dataset)
    
    # Run experiments
    for dataset in datasets:
        print(f"\n{'#'*60}")
        print(f"# Dataset: {dataset}")
        print(f"# Purpose: {DATASETS.get(dataset, {}).get('purpose', 'N/A')}")
        print(f"# Expected winner: {DATASETS.get(dataset, {}).get('expected_winner', 'N/A')}")
        print(f"{'#'*60}")
        
        for model in models:
            result = run_experiment(model, dataset, max_iters)
            results.append(result)
            
            if result['success']:
                print(f"  ✓ {MODELS[model]['name']}: val_loss = {result['val_loss']:.4f} ({result['time']:.1f}s)")
            else:
                print(f"  ✗ {MODELS[model]['name']}: FAILED")
    
    return results


def print_summary(results):
    """Print summary table of results."""
    print(f"\n{'='*80}")
    print("EXPERIMENT SUMMARY")
    print(f"{'='*80}")
    
    # Group by dataset
    datasets = set(r['dataset'] for r in results)
    
    for dataset in sorted(datasets):
        print(f"\n{dataset}:")
        dataset_results = [r for r in results if r['dataset'] == dataset and r['success']]
        dataset_results.sort(key=lambda x: x['val_loss'] if x['val_loss'] else float('inf'))
        
        for i, r in enumerate(dataset_results):
            marker = "🏆" if i == 0 else "  "
            print(f"  {marker} {MODELS[r['model']]['name']}: {r['val_loss']:.4f}")


def main():
    parser = argparse.ArgumentParser(description='Run comparative experiments')
    parser.add_argument('--dataset', type=str, help='Specific dataset to test')
    parser.add_argument('--model', type=str, help='Specific model to test')
    parser.add_argument('--max_iters', type=int, default=5000, help='Training iterations')
    parser.add_argument('--all', action='store_true', help='Run all experiments')
    parser.add_argument('--compile', action='store_true', help='Use torch.compile')
    parser.add_argument('--prepare-only', action='store_true', help='Only prepare datasets')
    
    args = parser.parse_args()
    
    if args.prepare_only:
        for dataset in DATASETS.keys():
            prepare_dataset(dataset)
        return
    
    if args.all:
        results = run_all_experiments(max_iters=args.max_iters)
        print_summary(results)
    elif args.dataset:
        datasets = [args.dataset]
        models = [args.model] if args.model else None
        results = run_all_experiments(datasets=datasets, models=models, max_iters=args.max_iters)
        print_summary(results)
    else:
        print("Usage:")
        print("  python run_comparative_experiments.py --all")
        print("  python run_comparative_experiments.py --dataset rotation3d")
        print("  python run_comparative_experiments.py --prepare-only")


if __name__ == "__main__":
    main()
