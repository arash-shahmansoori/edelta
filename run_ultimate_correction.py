#!/usr/bin/env python3
"""
Run experiments on Ultimate Correction datasets.
Focus on tasks that expose architectural differences.
"""

import os
import subprocess
import numpy as np
import json

# Most differentiating tasks
TASKS = ['rotation_reflection', 'isometry', 'cumulative']

# Models to compare
MODELS = ['gpt2', 'ddl', 'mhc', 'edelta']
MODEL_NAMES = {'gpt2': 'GPT', 'ddl': 'DDL', 'mhc': 'mHC', 'edelta': 'E∆-MHC-Geo'}

# Training config
CONFIG = {
    'n_layer': 6,
    'n_embd': 128,
    'n_head': 4,
    'n_streams': 4,
    'dropout': 0.0,
    'batch_size': 64,
    'learning_rate': 1e-3,
    'max_iters': 3000,  # Longer training for harder tasks
    'weight_decay': 0.1,
    'grad_clip': 1.0,
    'eval_interval': 100,
    'log_interval': 100,
}


def run_experiment(model_type, task, output_base='out-ultimate'):
    """Run a single experiment."""
    out_dir = f"{output_base}/{task}-{model_type}"
    
    cmd = [
        'uv', 'run', 'python', 'train_ultimate_correction.py',
        f'--model_type={model_type}',
        f'--task={task}',
        f'--out_dir={out_dir}',
    ]
    
    for key, value in CONFIG.items():
        cmd.append(f'--{key}={value}')
    
    print(f"\n{'='*50}")
    print(f"Running: {MODEL_NAMES[model_type]} on {task}")
    print(f"{'='*50}\n")
    
    result = subprocess.run(cmd, capture_output=False)
    return result.returncode == 0


def main():
    print("=" * 60)
    print("Ultimate Correction Experiments")
    print("=" * 60)
    
    os.makedirs('out-ultimate', exist_ok=True)
    
    all_results = {}
    
    for task in TASKS:
        print(f"\n\n{'#'*60}")
        print(f"# TASK: {task}")
        print(f"{'#'*60}")
        
        all_results[task] = {}
        
        for model in MODELS:
            run_experiment(model, task)
            
            # Load results
            result_path = f'out-ultimate/{task}-{model}/results.npy'
            if os.path.exists(result_path):
                res = np.load(result_path, allow_pickle=True).item()
                all_results[task][model] = {
                    'best_val': res['best_val_loss'],
                    'final_val': res['final_val_loss'],
                }
    
    # Print summary
    print("\n\n" + "=" * 70)
    print("ULTIMATE CORRECTION RESULTS SUMMARY")
    print("=" * 70)
    
    for task in TASKS:
        print(f"\n{task.upper()}:")
        print("-" * 50)
        print(f"{'Model':<15} {'Best Val':<15} {'Final Val':<15}")
        print("-" * 50)
        
        for model in MODELS:
            if model in all_results.get(task, {}):
                r = all_results[task][model]
                print(f"{MODEL_NAMES[model]:<15} {r['best_val']:.2e}      {r['final_val']:.2e}")
    
    # Save
    with open('out-ultimate/results_summary.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nResults saved to: out-ultimate/results_summary.json")


if __name__ == '__main__':
    main()
