"""
Dataset 2: The "Correction Protocol" (Logical Negation)

Target: Proves TOPOLOGICAL COMPLETENESS (Negation) vs. Pure Cayley's BLIND SPOT.

Why this is the "Kill-Shot":
- Requires the model to perform "belief flips" - mapping x → -x instantly.
- Pure Cayley (rotations, det=+1) cannot map v to -v in odd dimensions, or 
  without infinite energy in even dimensions.
- E∆-MHC-Geo Hybrid gate detects the "Correction" signal, sets γ→0 (Reflection),
  and flips the vector instantly via Householder.

Expected Results:
- Pure Cayley struggles to invert (slow curve around the sphere)
- E∆-MHC-Geo Hybrid snaps instantly to perfect flip (cosine similarity → -1.0)

Dataset Specification (from comparative study):
- Training: 4,500 sequences
- Validation: 500 sequences  
- Sequence length: 32
- Vector dimension: 32
- Metric: Cosine Similarity
"""

import numpy as np
import os
import argparse


def generate_correction_data(
    output_dir: str = 'data/correction',
    dim: int = 32,
    seq_len: int = 32,
    n_train: int = 4500,
    n_val: int = 500,
    seed: int = 42
):
    """
    Generate correction protocol dataset for belief flip prediction.
    
    The task: Learn that a "signal" token means "flip the sign of the concept".
    
    Sequence structure (variable signal position):
    [concept, concept, ..., SIGNAL, -concept, -concept, ...]
    
    The model must:
    1. Maintain identity during "maintenance" phase (x → x)
    2. Instantly flip at the signal (signal → -x)
    3. Maintain inverted state afterward (-x → -x)
    
    Args:
        output_dir: Directory to save the dataset
        dim: Vector dimension (default 32)
        seq_len: Sequence length (default 32)
        n_train: Number of training sequences (default 4500)
        n_val: Number of validation sequences (default 500)
        seed: Random seed
    """
    os.makedirs(output_dir, exist_ok=True)
    np.random.seed(seed)
    
    n_total = n_train + n_val
    
    print(f"Generating {n_total} correction protocol sequences...")
    print(f"  Dimension: {dim}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Training samples: {n_train}")
    print(f"  Validation samples: {n_val}")
    
    data_x = []
    data_y = []
    signal_positions = []
    
    for i in range(n_total):
        # 1. Generate random base concept vector (unit vector)
        concept = np.random.randn(dim)
        concept = concept / np.linalg.norm(concept)
        
        # 2. Generate the signal vector (distinct from concept space)
        # Use high magnitude in first dimension as "attention flag"
        signal = np.zeros(dim)
        signal[0] = 5.0  # High magnitude "Correction" flag
        # Add some noise to other dimensions for variety
        signal[1:4] = np.random.randn(3) * 0.1
        
        # 3. Random signal position (between 25% and 75% of sequence)
        min_pos = max(3, seq_len // 4)
        max_pos = min(seq_len - 3, 3 * seq_len // 4)
        signal_pos = np.random.randint(min_pos, max_pos)
        signal_positions.append(signal_pos)
        
        # 4. Build sequence
        seq_input = []
        seq_target = []
        
        for t in range(seq_len):
            if t < signal_pos:
                # Before signal: maintain concept
                seq_input.append(concept.copy())
                seq_target.append(concept.copy())  # x → x
            elif t == signal_pos:
                # At signal: flip
                seq_input.append(signal.copy())
                seq_target.append(-concept.copy())  # signal → -x (THE CRITICAL FLIP!)
            else:
                # After signal: maintain inverted concept
                seq_input.append(-concept.copy())
                seq_target.append(-concept.copy())  # -x → -x
        
        data_x.append(np.stack(seq_input))
        data_y.append(np.stack(seq_target))
        
        if (i + 1) % 1000 == 0:
            print(f"  Generated {i + 1}/{n_total} sequences")
    
    # Convert to Float32
    X = np.stack(data_x).astype(np.float32)
    Y = np.stack(data_y).astype(np.float32)
    signal_positions = np.array(signal_positions).astype(np.int32)
    
    # Split
    train_x, val_x = X[:n_train], X[n_train:]
    train_y, val_y = Y[:n_train], Y[n_train:]
    train_signal_pos = signal_positions[:n_train]
    val_signal_pos = signal_positions[n_train:]
    
    # Save
    np.save(os.path.join(output_dir, 'train_x.npy'), train_x)
    np.save(os.path.join(output_dir, 'train_y.npy'), train_y)
    np.save(os.path.join(output_dir, 'val_x.npy'), val_x)
    np.save(os.path.join(output_dir, 'val_y.npy'), val_y)
    np.save(os.path.join(output_dir, 'signal_positions.npy'), signal_positions)
    
    # Save metadata
    metadata = {
        'dim': dim,
        'seq_len': seq_len,
        'n_train': n_train,
        'n_val': n_val,
        'signal_pos_mean': float(signal_positions.mean()),
        'signal_pos_std': float(signal_positions.std()),
    }
    np.save(os.path.join(output_dir, 'metadata.npy'), metadata)
    
    print(f"\nCorrection dataset saved to {output_dir}/")
    print(f"  Train: {train_x.shape} -> {train_y.shape}")
    print(f"  Val: {val_x.shape} -> {val_y.shape}")
    print(f"  Signal position range: [{min_pos}, {max_pos})")
    print(f"  Signal position mean: {signal_positions.mean():.1f} ± {signal_positions.std():.1f}")
    print(f"  Storage: {(train_x.nbytes + train_y.nbytes + val_x.nbytes + val_y.nbytes) / 1e6:.1f} MB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate Correction dataset')
    parser.add_argument('--output_dir', type=str, default='data/correction')
    parser.add_argument('--dim', type=int, default=32)
    parser.add_argument('--seq_len', type=int, default=32)
    parser.add_argument('--n_train', type=int, default=4500)
    parser.add_argument('--n_val', type=int, default=500)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    generate_correction_data(
        output_dir=args.output_dir,
        dim=args.dim,
        seq_len=args.seq_len,
        n_train=args.n_train,
        n_val=args.n_val,
        seed=args.seed
    )
