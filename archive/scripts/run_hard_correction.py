#!/usr/bin/env python3
"""
Run experiments on Hard Correction datasets to expose architectural differences.
"""

import os
import subprocess
import numpy as np
import json
from datetime import datetime

# Datasets to test
DIFFICULTIES = ['multiple', 'orthogonal']

# Models to compare
MODELS = ['gpt2', 'ddl', 'mhc', 'edelta']
MODEL_NAMES = ['GPT', 'DDL', 'mHC', 'E∆-MHC-Geo']

# Training config (Fair Fight)
CONFIG = {
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


def load_hard_correction_data(difficulty):
    """Load hard correction dataset."""
    filepath = f'data/correction_hard_{difficulty}.npz'
    data = np.load(filepath)
    return {
        'train_x': data['train_x'],
        'train_y': data['train_y'],
        'val_x': data['val_x'],
        'val_y': data['val_y'],
    }


def run_experiment(model_type, difficulty, output_base='out-hard-correction'):
    """Run a single experiment."""
    out_dir = f"{output_base}/{difficulty}-{model_type}"
    
    # Build command
    cmd = [
        'uv', 'run', 'python', 'train_hard_correction.py',
        f'--model_type={model_type}',
        f'--difficulty={difficulty}',
        f'--out_dir={out_dir}',
    ]
    
    # Add config
    for key, value in CONFIG.items():
        cmd.append(f'--{key}={value}')
    
    print(f"\n{'='*50}")
    print(f"Running: {model_type} on correction_{difficulty}")
    print(f"Output: {out_dir}")
    print(f"{'='*50}\n")
    
    result = subprocess.run(cmd, capture_output=False)
    return result.returncode == 0


def main():
    print("=" * 60)
    print("Hard Correction Experiment")
    print("Testing: multiple negations & orthogonal negation")
    print("=" * 60)
    
    results = {}
    
    for difficulty in DIFFICULTIES:
        print(f"\n\n{'#'*60}")
        print(f"# DIFFICULTY: {difficulty}")
        print(f"{'#'*60}")
        
        results[difficulty] = {}
        
        for model in MODELS:
            success = run_experiment(model, difficulty)
            
            # Load results
            result_path = f'out-hard-correction/{difficulty}-{model}/results.npy'
            if os.path.exists(result_path):
                res = np.load(result_path, allow_pickle=True).item()
                results[difficulty][model] = res['final_val_loss']
                print(f"\n  {model}: {res['final_val_loss']:.6e}")
    
    # Print summary
    print("\n\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    
    print(f"\n{'Model':<15} {'Multiple':<15} {'Orthogonal':<15}")
    print("-" * 45)
    for model, name in zip(MODELS, MODEL_NAMES):
        mult = results.get('multiple', {}).get(model, 'N/A')
        orth = results.get('orthogonal', {}).get(model, 'N/A')
        if isinstance(mult, float):
            mult = f"{mult:.2e}"
        if isinstance(orth, float):
            orth = f"{orth:.2e}"
        print(f"{name:<15} {mult:<15} {orth:<15}")
    
    # Save results
    with open('out-hard-correction/results_summary.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: out-hard-correction/results_summary.json")


if __name__ == '__main__':
    main()
