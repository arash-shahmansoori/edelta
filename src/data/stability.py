"""
Dataset 3: The "Infinite Echo" (Stability Test)

Target: Proves UNCONDITIONAL ISOMETRY vs. Baseline NORM DRIFT.

Why this is the "Kill-Shot":
- A "Pass-the-Parcel" game over 1,000+ steps. The model must preserve a specific 
  noise vector perfectly without growth (explosion) or shrinkage (vanishing).
- Baseline Transformers with standard residuals (x + f(x)) are expansive. The norm 
  grows slightly at every step. Over 1,000 steps, the vector becomes massive (NaN).
- E∆-MHC-Geo Cayley operator has eigenvalues of EXACTLY 1.0. The norm at step 
  1,000 is identical to step 0, down to floating-point precision.

Expected Results:
- Baseline/DDL norms drift to infinity or zero
- E∆-MHC-Geo norm is a perfectly horizontal line at ||v|| = 1
"""

import numpy as np
import os
import argparse


def generate_stability_data(
    output_dir: str = 'data/stability',
    dim: int = 64,
    n_train: int = 900,
    n_val: int = 100,
    train_seq_len: int = 128,
    noise_scale: float = 0.001,
    seed: int = 42
):
    """
    Generate stability test dataset for isometry verification.
    
    The task: Identity mapping with tiny noise - output = input
    But we test on VERY LONG sequences during inference to detect norm drift.
    
    Training: Short sequences (128 steps) for backprop
    Testing: Run inference loop for 10,000+ steps to observe stability
    
    Args:
        output_dir: Directory to save the dataset
        dim: Vector dimension (default 64)
        n_train: Number of training sequences
        n_val: Number of validation sequences
        train_seq_len: Training sequence length
        noise_scale: Scale of tiny noise added (forces active denoising)
        seed: Random seed
    """
    os.makedirs(output_dir, exist_ok=True)
    np.random.seed(seed)
    
    n_total = n_train + n_val
    
    print(f"Generating {n_total} infinite echo streams...")
    print(f"  Dimension: {dim}")
    print(f"  Training sequence length: {train_seq_len}")
    print(f"  Noise scale: {noise_scale}")
    
    data_x = []
    data_y = []
    keys = []  # Store the original key vectors for analysis
    
    for i in range(n_total):
        # 1. The "Key" (Random unit noise pattern to preserve)
        key = np.random.randn(dim)
        key = key / np.linalg.norm(key)
        keys.append(key.copy())
        
        # 2. Generate trajectory with tiny noise
        # The task is: maintain the key vector despite noise
        traj = []
        for _ in range(train_seq_len):
            # Add tiny jitter - forces the model to actively denoise/maintain
            noise = np.random.randn(dim) * noise_scale
            traj.append(key + noise)
        
        traj = np.stack(traj)
        
        # Input: t=0 to T-1, Target: t=1 to T (shifted identity)
        data_x.append(traj[:-1])
        data_y.append(traj[1:])
        
        if (i + 1) % 200 == 0:
            print(f"  Generated {i + 1}/{n_total} sequences")
    
    # Convert to Float32
    X = np.stack(data_x).astype(np.float32)
    Y = np.stack(data_y).astype(np.float32)
    keys = np.stack(keys).astype(np.float32)
    
    # Split
    train_x, val_x = X[:n_train], X[n_train:]
    train_y, val_y = Y[:n_train], Y[n_train:]
    train_keys, val_keys = keys[:n_train], keys[n_train:]
    
    # Save training data
    np.save(os.path.join(output_dir, 'train_x.npy'), train_x)
    np.save(os.path.join(output_dir, 'train_y.npy'), train_y)
    np.save(os.path.join(output_dir, 'train_keys.npy'), train_keys)
    
    # Save validation data
    np.save(os.path.join(output_dir, 'val_x.npy'), val_x)
    np.save(os.path.join(output_dir, 'val_y.npy'), val_y)
    np.save(os.path.join(output_dir, 'val_keys.npy'), val_keys)
    
    print(f"\nStability dataset saved to {output_dir}/")
    print(f"  Train: {train_x.shape} -> {train_y.shape}")
    print(f"  Val: {val_x.shape} -> {val_y.shape}")
    print(f"  Keys saved for long-horizon testing")
    
    # Save metadata
    metadata = {
        'dim': dim,
        'train_seq_len': train_seq_len,
        'noise_scale': noise_scale,
        'n_train': n_train,
        'n_val': n_val,
    }
    np.save(os.path.join(output_dir, 'metadata.npy'), metadata)


def generate_long_test_sequences(
    output_dir: str = 'data/stability',
    dim: int = 64,
    n_sequences: int = 100,
    seq_len: int = 10000,
    noise_scale: float = 0.0,  # No noise for pure stability test
    seed: int = 123
):
    """
    Generate very long test sequences for stability analysis.
    
    These are used ONLY for inference testing - we run the model in a loop
    and measure norm drift over 10,000 steps.
    
    Args:
        output_dir: Directory to save
        dim: Vector dimension
        n_sequences: Number of test sequences
        seq_len: Very long sequence length (10,000+)
        noise_scale: Noise scale (0 for pure stability test)
        seed: Random seed (different from training)
    """
    os.makedirs(output_dir, exist_ok=True)
    np.random.seed(seed)
    
    print(f"Generating {n_sequences} long stability test sequences...")
    print(f"  Sequence length: {seq_len} (for norm drift analysis)")
    
    # Generate initial key vectors
    keys = []
    for _ in range(n_sequences):
        key = np.random.randn(dim)
        key = key / np.linalg.norm(key)
        keys.append(key)
    
    keys = np.stack(keys).astype(np.float32)
    
    # Save only the initial keys - we'll run inference loops
    np.save(os.path.join(output_dir, 'long_test_keys.npy'), keys)
    
    # Also save configuration for the test
    test_config = {
        'dim': dim,
        'n_sequences': n_sequences,
        'target_seq_len': seq_len,
        'noise_scale': noise_scale,
    }
    np.save(os.path.join(output_dir, 'long_test_config.npy'), test_config)
    
    print(f"  Saved {n_sequences} initial keys for {seq_len}-step inference test")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate Stability dataset')
    parser.add_argument('--output_dir', type=str, default='data/stability')
    parser.add_argument('--dim', type=int, default=64)
    parser.add_argument('--n_train', type=int, default=900)
    parser.add_argument('--n_val', type=int, default=100)
    parser.add_argument('--train_seq_len', type=int, default=128)
    parser.add_argument('--long_test', action='store_true', help='Also generate long test sequences')
    parser.add_argument('--long_test_len', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    # Generate training/validation data
    generate_stability_data(
        output_dir=args.output_dir,
        dim=args.dim,
        n_train=args.n_train,
        n_val=args.n_val,
        train_seq_len=args.train_seq_len,
        seed=args.seed
    )
    
    # Optionally generate long test sequences
    if args.long_test:
        generate_long_test_sequences(
            output_dir=args.output_dir,
            dim=args.dim,
            n_sequences=100,
            seq_len=args.long_test_len,
            seed=args.seed + 1000  # Different seed
        )
