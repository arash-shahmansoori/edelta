"""
Run Complete Gyroscope Experiments

This script runs all models on the Continuous Gyroscope task:
1. Baseline (standard transformer)
2. DDL (Householder reflection) 
3. Cayley (pure rotation)
4. Gearbox (intelligent DDL/Cayley switching)

Expected Results:
- DDL will struggle with large rotation angles
- Cayley will excel at geometric rotation
- Gearbox will learn to use Cayley for large angles

Author: Arash Shahmansoori (2026)
"""

import os
import subprocess
import sys
import time
from datetime import datetime


def run_experiment(model_script: str, out_dir: str, extra_args: dict = None) -> dict:
    """Run a single training experiment."""
    
    # Base arguments for gyroscope task
    base_args = {
        'dataset': 'gyroscope',
        'n_layer': 6,
        'n_head': 6,
        'n_embd': 384,
        'n_streams': 4,
        'batch_size': 64,
        'block_size': 256,
        'max_iters': 5000,
        'eval_interval': 500,
        'log_interval': 100,
        'learning_rate': '6e-4',
        'dropout': 0.0,
        'compile': 'False',
    }
    
    if extra_args:
        base_args.update(extra_args)
    
    base_args['out_dir'] = out_dir
    
    # Build command
    cmd = ['python', '-u', model_script]
    for key, value in base_args.items():
        cmd.append(f'--{key}={value}')
    
    print(f"\n{'='*60}")
    print(f"Running: {model_script}")
    print(f"Output: {out_dir}")
    print(f"{'='*60}\n")
    
    start_time = time.time()
    
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd='/root/edelta'
    )
    
    elapsed = time.time() - start_time
    
    # Extract final validation loss from output
    val_loss = None
    for line in result.stdout.split('\n')[::-1]:
        if 'val loss' in line.lower() or 'val_loss' in line.lower():
            try:
                # Try to parse val loss from line
                parts = line.split()
                for i, part in enumerate(parts):
                    if 'val' in part.lower() and i+1 < len(parts):
                        val_loss = float(parts[i+1].rstrip(','))
                        break
            except:
                pass
            if val_loss:
                break
    
    return {
        'model': model_script,
        'out_dir': out_dir,
        'elapsed': elapsed,
        'val_loss': val_loss,
        'success': result.returncode == 0,
        'stdout': result.stdout,
        'stderr': result.stderr,
    }


def main():
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results = []
    
    print("="*60)
    print("CONTINUOUS GYROSCOPE EXPERIMENT")
    print("Testing DDL Failure vs Cayley Success")
    print("="*60)
    
    # Define experiments
    experiments = [
        ('train.py', 'out-gyro-baseline', {}),
        ('train_ddl.py', 'out-gyro-ddl', {}),
        ('train_cayley.py', 'out-gyro-cayley', {}),
        ('train_gearbox.py', 'out-gyro-gearbox', {'init_gate_bias': '0.0'}),
    ]
    
    for script, out_dir, extra_args in experiments:
        result = run_experiment(script, out_dir, extra_args)
        results.append(result)
        
        if result['success']:
            print(f"✓ {script}: val_loss = {result['val_loss']}, time = {result['elapsed']:.1f}s")
        else:
            print(f"✗ {script}: FAILED")
            print(result['stderr'][:500] if result['stderr'] else 'No error output')
    
    # Print summary
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    
    print(f"\n{'Model':<30} {'Val Loss':<15} {'Status':<10}")
    print("-"*60)
    
    for r in results:
        model_name = r['model'].replace('train_', '').replace('.py', '').upper()
        val_loss = f"{r['val_loss']:.4f}" if r['val_loss'] else "N/A"
        status = "✓" if r['success'] else "✗"
        print(f"{model_name:<30} {val_loss:<15} {status:<10}")
    
    print("\n" + "="*60)
    
    # Save results
    results_file = f'gyroscope_results_{timestamp}.txt'
    with open(results_file, 'w') as f:
        f.write("CONTINUOUS GYROSCOPE EXPERIMENT RESULTS\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write("="*60 + "\n\n")
        
        for r in results:
            f.write(f"Model: {r['model']}\n")
            f.write(f"Val Loss: {r['val_loss']}\n")
            f.write(f"Time: {r['elapsed']:.1f}s\n")
            f.write(f"Success: {r['success']}\n")
            f.write("-"*40 + "\n")
    
    print(f"Results saved to {results_file}")


if __name__ == '__main__':
    main()
