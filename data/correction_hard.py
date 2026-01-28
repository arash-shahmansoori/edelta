"""
Hard Correction Dataset - Designed to Expose Architectural Limitations

This dataset tests the model's ability to perform EXACT negation/reflection,
which requires eigenvalue -1 in the transformation matrix.

Key insight from theory:
- E∆-MHC-Geo: Householder H = I - 2kkᵀ has eigenvalue -1 → EXACT negation
- GPT/DDL: Can only approximate negation through learned weights
- mHC: Doubly stochastic matrices have all positive entries → CANNOT negate

Three difficulty levels:
1. Single negation (current baseline)
2. Multiple negations (tests accumulation of errors)
3. Long-range negation (tests memory + precision)
4. Selective negation (tests partial reflection)
"""

import numpy as np
import torch
import os


def generate_single_negation(n_samples, seq_len, dim, signal_pos=None):
    """
    Level 1: Single negation (baseline difficulty)
    
    Sequence: [noise, noise, SIGNAL, noise, ..., NEGATED_SIGNAL]
    """
    X = np.random.randn(n_samples, seq_len, dim).astype(np.float32)
    Y = X.copy()
    
    for i in range(n_samples):
        # Random signal position in first half
        if signal_pos is None:
            pos = np.random.randint(1, seq_len // 3)
        else:
            pos = signal_pos
        
        # Create unit signal
        signal = np.random.randn(dim).astype(np.float32)
        signal = signal / (np.linalg.norm(signal) + 1e-8)
        
        X[i, pos] = signal
        Y[i, -1] = -signal  # Target: exact negation
    
    return X, Y


def generate_multiple_negations(n_samples, seq_len, dim, n_flips=3):
    """
    Level 2: Multiple sequential negations
    
    Sequence: [SIGNAL, marker1, ..., marker2, ..., marker3, ..., TARGET]
    Target should be: (-1)^n_flips * SIGNAL
    
    This tests ERROR ACCUMULATION:
    - Approximate negation: error compounds with each flip
    - Exact negation (Householder): no error accumulation
    """
    X = np.random.randn(n_samples, seq_len, dim).astype(np.float32)
    Y = X.copy()
    
    # Marker vector (fixed, learnable pattern)
    marker = np.zeros(dim, dtype=np.float32)
    marker[0] = 1.0  # Simple marker
    
    for i in range(n_samples):
        # Signal at position 0
        signal = np.random.randn(dim).astype(np.float32)
        signal = signal / (np.linalg.norm(signal) + 1e-8)
        X[i, 0] = signal
        
        # Place n_flips markers at regular intervals
        flip_positions = np.linspace(seq_len // 4, 3 * seq_len // 4, n_flips, dtype=int)
        for pos in flip_positions:
            X[i, pos] = marker
        
        # Target: (-1)^n_flips * signal
        if n_flips % 2 == 1:
            Y[i, -1] = -signal
        else:
            Y[i, -1] = signal  # Even flips = identity
    
    return X, Y


def generate_long_range_negation(n_samples, seq_len, dim, min_distance=100):
    """
    Level 3: Long-range negation
    
    Signal at position ~10, target at position ~seq_len-1
    The model must:
    1. Remember the signal over long distance
    2. Apply exact negation
    
    This tests MEMORY + PRECISION:
    - GPT: Attention might lose precision over distance
    - mHC: Doubly stochastic averaging destroys signal
    - E∆-MHC-Geo: Orthogonal operations preserve norm and enable exact negation
    """
    assert seq_len > min_distance + 20, f"seq_len must be > {min_distance + 20}"
    
    X = np.random.randn(n_samples, seq_len, dim).astype(np.float32)
    Y = X.copy()
    
    for i in range(n_samples):
        # Signal early in sequence
        signal_pos = np.random.randint(5, 15)
        
        # Create unit signal
        signal = np.random.randn(dim).astype(np.float32)
        signal = signal / (np.linalg.norm(signal) + 1e-8)
        
        X[i, signal_pos] = signal
        
        # Negation marker somewhere in the middle
        marker_pos = np.random.randint(seq_len // 3, 2 * seq_len // 3)
        marker = np.zeros(dim, dtype=np.float32)
        marker[0] = 1.0
        X[i, marker_pos] = marker
        
        # Target at end: exact negation
        Y[i, -1] = -signal
    
    return X, Y


def generate_selective_negation(n_samples, seq_len, dim, negate_dims=None):
    """
    Level 4: Selective dimension negation
    
    Negate only specific dimensions, preserve others.
    
    This tests PARTIAL REFLECTION:
    - Requires the model to learn dimension-specific transformations
    - Householder can do this with appropriate k vector
    - Baselines must learn separate weights for each dimension
    """
    if negate_dims is None:
        negate_dims = list(range(dim // 2))  # Negate first half
    
    X = np.random.randn(n_samples, seq_len, dim).astype(np.float32)
    Y = X.copy()
    
    for i in range(n_samples):
        signal_pos = np.random.randint(1, seq_len // 3)
        
        signal = np.random.randn(dim).astype(np.float32)
        signal = signal / (np.linalg.norm(signal) + 1e-8)
        
        X[i, signal_pos] = signal
        
        # Target: negate only specified dimensions
        target = signal.copy()
        target[negate_dims] = -target[negate_dims]
        Y[i, -1] = target
    
    return X, Y


def generate_orthogonal_negation_test(n_samples, seq_len, dim):
    """
    Level 5: Orthogonal Negation Test (HARDEST)
    
    Input: Sequence of orthonormal vectors
    Target: Exact reflection of each vector
    
    This directly tests the isometry property:
    - Input norms: all ||x|| = 1
    - Target norms: all ||-x|| = 1
    - Cosine similarity: cos(pred, target) should be 1.0
    
    Baselines will show:
    - Norm drift: ||pred|| ≠ 1
    - Angular error: cos(pred, -x) < 1
    """
    X = np.random.randn(n_samples, seq_len, dim).astype(np.float32)
    
    # Normalize ALL vectors to unit length
    norms = np.linalg.norm(X, axis=-1, keepdims=True)
    X = X / (norms + 1e-8)
    
    # Target: exact negation of each position
    Y = -X.copy()
    
    return X, Y


def generate_hard_correction_data(
    difficulty='multiple',  # 'single', 'multiple', 'long_range', 'selective', 'orthogonal'
    n_train=4500,
    n_val=500,
    seq_len=128,  # Longer sequences
    dim=32,
    n_flips=3,
    save_dir='data'
):
    """
    Generate hard correction dataset.
    
    Args:
        difficulty: Which difficulty level
        n_train: Number of training samples
        n_val: Number of validation samples
        seq_len: Sequence length (longer = harder)
        dim: Vector dimension
        n_flips: Number of negations for 'multiple' difficulty
        save_dir: Directory to save data
    """
    
    print(f"Generating HARD correction data (difficulty={difficulty})...")
    print(f"  Sequences: {n_train + n_val}")
    print(f"  Seq length: {seq_len}")
    print(f"  Dimension: {dim}")
    
    if difficulty == 'single':
        train_x, train_y = generate_single_negation(n_train, seq_len, dim)
        val_x, val_y = generate_single_negation(n_val, seq_len, dim)
    elif difficulty == 'multiple':
        print(f"  Number of flips: {n_flips}")
        train_x, train_y = generate_multiple_negations(n_train, seq_len, dim, n_flips)
        val_x, val_y = generate_multiple_negations(n_val, seq_len, dim, n_flips)
    elif difficulty == 'long_range':
        train_x, train_y = generate_long_range_negation(n_train, seq_len, dim)
        val_x, val_y = generate_long_range_negation(n_val, seq_len, dim)
    elif difficulty == 'selective':
        train_x, train_y = generate_selective_negation(n_train, seq_len, dim)
        val_x, val_y = generate_selective_negation(n_val, seq_len, dim)
    elif difficulty == 'orthogonal':
        train_x, train_y = generate_orthogonal_negation_test(n_train, seq_len, dim)
        val_x, val_y = generate_orthogonal_negation_test(n_val, seq_len, dim)
    else:
        raise ValueError(f"Unknown difficulty: {difficulty}")
    
    # Save
    os.makedirs(save_dir, exist_ok=True)
    filename = f'correction_hard_{difficulty}.npz'
    filepath = os.path.join(save_dir, filename)
    
    np.savez(filepath,
             train_x=train_x, train_y=train_y,
             val_x=val_x, val_y=val_y)
    
    print(f"  Saved to: {filepath}")
    print(f"  Train shape: {train_x.shape}")
    print(f"  Val shape: {val_x.shape}")
    
    return train_x, train_y, val_x, val_y


if __name__ == '__main__':
    # Generate all difficulty levels
    for difficulty in ['single', 'multiple', 'long_range', 'selective', 'orthogonal']:
        print()
        generate_hard_correction_data(
            difficulty=difficulty,
            n_train=4500,
            n_val=500,
            seq_len=128,
            dim=32,
            n_flips=5,  # For 'multiple' difficulty
        )
    
    print("\n" + "=" * 60)
    print("All hard correction datasets generated!")
    print("=" * 60)
