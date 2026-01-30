#!/usr/bin/env python3
"""
Ablation Study Framework for E∆-MHC-Geo

Following mHC paper (arXiv:2512.24880) Table 1 ablation style, we test:
1. Cayley-only: Disable Householder (γ=1 always)
2. Householder-only: Disable Cayley (γ=0 always)
3. Full E∆-MHC-Geo: Both with thermodynamic gating
4. Different gate regularization weights
5. Different initial gate biases

Plus baseline comparisons: GPT, DDL, mHC
"""

import os
import sys
import argparse
import subprocess
import time
import json
from datetime import datetime

# Ablation configurations
ABLATION_CONFIGS = {
    # Core model ablations
    'cayley_only': {
        'description': 'Cayley rotation only (no Householder), γ→1',
        'model_type': 'edelta',
        'init_gate_bias': 10.0,  # Large positive bias forces γ→1
        'gate_reg_weight': 0.0,  # No regularization needed
    },
    'householder_only': {
        'description': 'Householder reflection only (no Cayley), γ→0',
        'model_type': 'edelta',
        'init_gate_bias': -10.0,  # Large negative bias forces γ→0
        'gate_reg_weight': 0.0,
    },
    'edelta_full': {
        'description': 'Full E∆-MHC-Geo with thermodynamic gating',
        'model_type': 'edelta',
        'init_gate_bias': 0.0,  # Neutral
        'gate_reg_weight': 0.1,  # Midpoint collapse regularization
    },
    
    # Gate regularization ablations
    'edelta_no_reg': {
        'description': 'E∆-MHC-Geo without gate regularization',
        'model_type': 'edelta',
        'init_gate_bias': 0.0,
        'gate_reg_weight': 0.0,  # No regularization
    },
    'edelta_strong_reg': {
        'description': 'E∆-MHC-Geo with strong gate regularization',
        'model_type': 'edelta',
        'init_gate_bias': 0.0,
        'gate_reg_weight': 0.5,  # Strong regularization
    },
    
    # Gate bias ablations
    'edelta_rotate_bias': {
        'description': 'E∆-MHC-Geo biased toward rotation',
        'model_type': 'edelta',
        'init_gate_bias': 2.0,  # Prefer rotation
        'gate_reg_weight': 0.1,
    },
    'edelta_reflect_bias': {
        'description': 'E∆-MHC-Geo biased toward reflection',
        'model_type': 'edelta',
        'init_gate_bias': -2.0,  # Prefer reflection
        'gate_reg_weight': 0.1,
    },
    
    # Baselines for comparison
    'gpt_baseline': {
        'description': 'Standard GPT (naive residual)',
        'model_type': 'gpt2',
    },
    'ddl_baseline': {
        'description': 'DDL baseline',
        'model_type': 'ddl',
    },
    'mhc_baseline': {
        'description': 'mHC baseline (data-dependent)',
        'model_type': 'mhc',
    },
}

# Datasets to test
DATASETS = ['gyroscope', 'correction', 'stability']

# Fair Fight hyperparameters (matching your original specs)
FAIR_FIGHT_CONFIG = {
    'n_layer': 6,
    'n_embd': 128,
    'n_head': 4,
    'n_streams': 4,
    'dropout': 0.0,
    'batch_size': 64,
    'learning_rate': 1e-3,
    'max_iters': 2000,
    'weight_decay': 0.1,
    'grad_clip': 1.0,
    'eval_interval': 100,
    'log_interval': 50,
}


def run_training(config_name, dataset, config, output_base='out-ablation'):
    """Run a single training job."""
    
    out_dir = f"{output_base}/{config_name}-{dataset}"
    
    # Build command
    cmd = [
        'uv', 'run', 'python', 'train_continuous.py',
        f'--model_type={config["model_type"]}',
        f'--dataset={dataset}',
        f'--out_dir={out_dir}',
    ]
    
    # Add fair fight config
    for key, value in FAIR_FIGHT_CONFIG.items():
        cmd.append(f'--{key}={value}')
    
    # Add model-specific config
    if config['model_type'] == 'edelta':
        cmd.append(f'--init_gate_bias={config.get("init_gate_bias", 0.0)}')
        cmd.append(f'--gate_reg_weight={config.get("gate_reg_weight", 0.1)}')
    
    print(f"\n{'='*60}")
    print(f"Running: {config_name} on {dataset}")
    print(f"Description: {config['description']}")
    print(f"Output: {out_dir}")
    print(f"{'='*60}\n")
    
    # Run training
    start_time = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - start_time
    
    if result.returncode != 0:
        print(f"ERROR: {result.stderr}")
        return None
    
    return {
        'config_name': config_name,
        'dataset': dataset,
        'out_dir': out_dir,
        'elapsed_seconds': elapsed,
        'success': result.returncode == 0,
    }


def run_ablation_study(configs_to_run=None, datasets_to_run=None, output_base='out-ablation'):
    """Run full ablation study."""
    
    if configs_to_run is None:
        configs_to_run = list(ABLATION_CONFIGS.keys())
    
    if datasets_to_run is None:
        datasets_to_run = DATASETS
    
    results = []
    total_jobs = len(configs_to_run) * len(datasets_to_run)
    completed = 0
    
    print(f"\n{'#'*60}")
    print(f"# ABLATION STUDY: {total_jobs} jobs")
    print(f"# Configs: {configs_to_run}")
    print(f"# Datasets: {datasets_to_run}")
    print(f"{'#'*60}\n")
    
    os.makedirs(output_base, exist_ok=True)
    
    for dataset in datasets_to_run:
        for config_name in configs_to_run:
            completed += 1
            print(f"\n[{completed}/{total_jobs}] ", end='')
            
            config = ABLATION_CONFIGS[config_name]
            result = run_training(config_name, dataset, config, output_base)
            
            if result:
                results.append(result)
    
    # Save results summary
    summary_path = os.path.join(output_base, 'ablation_summary.json')
    with open(summary_path, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'configs': ABLATION_CONFIGS,
            'fair_fight_config': FAIR_FIGHT_CONFIG,
            'results': results,
        }, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Ablation study complete!")
    print(f"Results saved to: {summary_path}")
    print(f"{'='*60}\n")
    
    return results


def run_quick_ablation():
    """Run a quick ablation study with core configs only."""
    core_configs = [
        'cayley_only',
        'householder_only', 
        'edelta_full',
        'gpt_baseline',
        'ddl_baseline',
        'mhc_baseline',
    ]
    return run_ablation_study(configs_to_run=core_configs)


def run_full_ablation():
    """Run full ablation study with all configs."""
    return run_ablation_study()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run E∆-MHC-Geo Ablation Study')
    parser.add_argument('--mode', type=str, default='quick',
                       choices=['quick', 'full', 'single'],
                       help='Ablation mode: quick (core only), full (all), single (one config)')
    parser.add_argument('--config', type=str, default=None,
                       help='Single config to run (with --mode single)')
    parser.add_argument('--dataset', type=str, default=None,
                       help='Single dataset to run (optional)')
    parser.add_argument('--output_base', type=str, default='out-ablation',
                       help='Output base directory')
    parser.add_argument('--list', action='store_true',
                       help='List available configs')
    
    args = parser.parse_args()
    
    if args.list:
        print("\nAvailable ablation configurations:")
        print("-" * 60)
        for name, config in ABLATION_CONFIGS.items():
            print(f"  {name:25s} - {config['description']}")
        print()
        sys.exit(0)
    
    if args.mode == 'single':
        if args.config is None:
            print("Error: --config required with --mode single")
            sys.exit(1)
        datasets = [args.dataset] if args.dataset else DATASETS
        run_ablation_study(
            configs_to_run=[args.config],
            datasets_to_run=datasets,
            output_base=args.output_base
        )
    elif args.mode == 'quick':
        run_quick_ablation()
    else:
        run_full_ablation()
