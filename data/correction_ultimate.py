#!/usr/bin/env python3
"""
Ultimate Correction Task - Designed to expose architectural differences

Key challenges:
1. Random number of flips (unknown at test time)
2. Long-range dependencies (flips affect distant positions)
3. Cumulative transformations (each flip builds on previous state)
4. Signal preservation under noise
"""

import os
import numpy as np


def generate_cumulative_negation(n_samples, seq_len, dim, max_flips=10):
    """
    Level 1: Cumulative Negation
    
    Each flip marker triggers negation of ALL subsequent positions.
    Model must track parity of flips seen so far.
    
    Example (1D): [1, FLIP, 2, 3, FLIP, 4] -> [1, -, -2, -3, -, 4]
    """
    X = np.zeros((n_samples, seq_len, dim), dtype=np.float32)
    Y = np.zeros((n_samples, seq_len, dim), dtype=np.float32)
    
    for i in range(n_samples):
        # Random number of flips (1 to max_flips)
        n_flips = np.random.randint(1, max_flips + 1)
        
        # Random flip positions (sorted)
        flip_positions = sorted(np.random.choice(seq_len - 1, n_flips, replace=False))
        
        # Generate random signal
        signal = np.random.randn(seq_len, dim).astype(np.float32) * 0.5
        
        # Mark flip positions with special pattern (high magnitude in first dim)
        for pos in flip_positions:
            signal[pos, 0] = 5.0  # Marker
        
        X[i] = signal
        
        # Compute output with cumulative parity
        parity = 1  # 1 = positive, -1 = negative
        for t in range(seq_len):
            if t in flip_positions:
                parity *= -1
                Y[i, t] = signal[t]  # Marker unchanged
            else:
                Y[i, t] = parity * signal[t]
    
    return X, Y


def generate_selective_accumulation(n_samples, seq_len, dim):
    """
    Level 2: Selective Accumulation with Negation
    
    Input has two types of tokens:
    - Values: random vectors
    - Operations: +1 (accumulate) or -1 (negate and accumulate)
    
    Output is the running sum with sign operations applied.
    Tests: Long-range dependency + exact negation + accumulation
    """
    X = np.zeros((n_samples, seq_len, dim), dtype=np.float32)
    Y = np.zeros((n_samples, seq_len, dim), dtype=np.float32)
    
    for i in range(n_samples):
        accumulator = np.zeros(dim, dtype=np.float32)
        
        for t in range(seq_len):
            # Random operation: 70% add, 30% negate-add
            op = 1 if np.random.random() > 0.3 else -1
            
            # Random value
            value = np.random.randn(dim).astype(np.float32) * 0.3
            
            # Store input (operation encoded in first dim magnitude)
            X[i, t] = value.copy()
            X[i, t, 0] = op * 2.0  # Operation marker
            
            # Apply operation
            accumulator += op * value
            
            # Output is current accumulator state
            Y[i, t] = accumulator.copy()
    
    return X, Y


def generate_echo_with_negation(n_samples, seq_len, dim, echo_delay=16):
    """
    Level 3: Echo with Conditional Negation
    
    Input: [signal, ..., marker, ...]
    Output: Echo of signal at delay, negated if marker is negative
    
    Tests: Long-range memory + conditional negation
    """
    X = np.zeros((n_samples, seq_len, dim), dtype=np.float32)
    Y = np.zeros((n_samples, seq_len, dim), dtype=np.float32)
    
    for i in range(n_samples):
        # Generate signal in first half
        signal_len = seq_len // 2
        signal = np.random.randn(signal_len, dim).astype(np.float32) * 0.5
        
        # Random negation marker
        negate = np.random.random() > 0.5
        marker = -1.0 if negate else 1.0
        
        # Fill input
        X[i, :signal_len] = signal
        X[i, signal_len, 0] = marker * 5.0  # Marker position
        
        # Output: echo with conditional negation
        multiplier = -1 if negate else 1
        for t in range(seq_len):
            if t < signal_len:
                Y[i, t] = 0  # No output during signal
            elif t == signal_len:
                Y[i, t] = 0  # Marker position
            else:
                echo_pos = t - signal_len - 1
                if echo_pos < signal_len:
                    Y[i, t] = multiplier * signal[echo_pos]
    
    return X, Y


def generate_rotation_vs_reflection(n_samples, seq_len, dim):
    """
    Level 4: Rotation vs Reflection Discrimination
    
    Input specifies either:
    - Rotation: Apply orthogonal rotation (preserve orientation)
    - Reflection: Apply reflection (flip orientation)
    
    This DIRECTLY tests Cayley (rotation) vs Householder (reflection)!
    """
    X = np.zeros((n_samples, seq_len, dim), dtype=np.float32)
    Y = np.zeros((n_samples, seq_len, dim), dtype=np.float32)
    
    for i in range(n_samples):
        # Generate random orthogonal transformation
        # For rotation: det = +1, for reflection: det = -1
        
        # Random operation
        is_reflection = np.random.random() > 0.5
        
        # Generate random orthogonal matrix
        A = np.random.randn(dim, dim).astype(np.float32)
        Q, _ = np.linalg.qr(A)
        
        if is_reflection:
            # Ensure det = -1 (reflection)
            if np.linalg.det(Q) > 0:
                Q[:, 0] *= -1
            marker = -1.0
        else:
            # Ensure det = +1 (rotation)
            if np.linalg.det(Q) < 0:
                Q[:, 0] *= -1
            marker = 1.0
        
        # Input: marker + sequence of vectors
        for t in range(seq_len):
            vec = np.random.randn(dim).astype(np.float32)
            vec = vec / (np.linalg.norm(vec) + 1e-8)  # Normalize
            X[i, t] = vec
            X[i, t, 0] += marker * 3.0  # Embed marker in first dim
            
            # Output: transformed vector
            Y[i, t] = Q @ vec
    
    return X, Y


def generate_isometry_stress_test(n_samples, seq_len, dim):
    """
    Level 5: Isometry Stress Test (HARDEST)
    
    Input: Sequence of orthonormal vectors
    Output: Must preserve orthonormality after transformation
    
    Any non-isometric operation will accumulate errors!
    
    This is the ULTIMATE test - only true isometries will succeed.
    """
    X = np.zeros((n_samples, seq_len, dim), dtype=np.float32)
    Y = np.zeros((n_samples, seq_len, dim), dtype=np.float32)
    
    for i in range(n_samples):
        # Generate random target orthogonal transformation
        A = np.random.randn(dim, dim).astype(np.float32)
        Q, _ = np.linalg.qr(A)
        
        # Randomly make it rotation or reflection
        if np.random.random() > 0.5 and np.linalg.det(Q) > 0:
            Q[:, 0] *= -1  # Make it a reflection
        
        # Generate orthonormal input vectors
        for t in range(seq_len):
            # Generate a random unit vector
            vec = np.random.randn(dim).astype(np.float32)
            vec = vec / (np.linalg.norm(vec) + 1e-8)
            
            X[i, t] = vec
            Y[i, t] = Q @ vec
    
    return X, Y


def generate_noise_robust_negation(n_samples, seq_len, dim, noise_level=0.1):
    """
    Level 6: Noise-Robust Negation
    
    Input: Noisy signal with flip markers
    Output: Clean negated signal (denoise + negate)
    
    Tests: Robustness of negation under perturbation
    """
    X = np.zeros((n_samples, seq_len, dim), dtype=np.float32)
    Y = np.zeros((n_samples, seq_len, dim), dtype=np.float32)
    
    for i in range(n_samples):
        # Clean signal
        clean = np.random.randn(seq_len, dim).astype(np.float32) * 0.5
        
        # Add noise to input
        noise = np.random.randn(seq_len, dim).astype(np.float32) * noise_level
        X[i] = clean + noise
        
        # Random flip pattern
        n_flips = np.random.randint(1, 6)
        flip_positions = sorted(np.random.choice(seq_len, n_flips, replace=False))
        
        # Mark flips
        for pos in flip_positions:
            X[i, pos, 0] = 5.0
        
        # Output: clean + negated based on cumulative parity
        parity = 1
        for t in range(seq_len):
            if t in flip_positions:
                parity *= -1
                Y[i, t] = clean[t]  # Marker position: clean, no negate
            else:
                Y[i, t] = parity * clean[t]  # Clean + negated
    
    return X, Y


def generate_ultimate_data(difficulty, n_train=4500, n_val=500, seq_len=128, dim=32):
    """Generate ultimate correction dataset."""
    
    os.makedirs('data', exist_ok=True)
    
    generators = {
        'cumulative': lambda n: generate_cumulative_negation(n, seq_len, dim),
        'accumulation': lambda n: generate_selective_accumulation(n, seq_len, dim),
        'echo': lambda n: generate_echo_with_negation(n, seq_len, dim),
        'rotation_reflection': lambda n: generate_rotation_vs_reflection(n, seq_len, dim),
        'isometry': lambda n: generate_isometry_stress_test(n, seq_len, dim),
        'noise_robust': lambda n: generate_noise_robust_negation(n, seq_len, dim),
    }
    
    gen = generators[difficulty]
    
    print(f"Generating {difficulty} data...")
    train_x, train_y = gen(n_train)
    val_x, val_y = gen(n_val)
    
    filepath = f'data/correction_ultimate_{difficulty}.npz'
    np.savez(filepath,
             train_x=train_x, train_y=train_y,
             val_x=val_x, val_y=val_y)
    
    print(f"  Saved: {filepath}")
    print(f"  Train: {train_x.shape}, Val: {val_x.shape}")
    
    # Statistics
    print(f"  Input range: [{train_x.min():.2f}, {train_x.max():.2f}]")
    print(f"  Output range: [{train_y.min():.2f}, {train_y.max():.2f}]")
    
    return filepath


if __name__ == '__main__':
    print("=" * 60)
    print("Ultimate Correction Dataset Generator")
    print("=" * 60)
    
    difficulties = [
        'cumulative',        # Track flip parity
        'accumulation',      # Running sum with negation
        'echo',              # Long-range memory + conditional negate
        'rotation_reflection',  # Direct test of Cayley vs Householder
        'isometry',          # Preserve orthonormality (HARDEST)
        'noise_robust',      # Denoise + negate
    ]
    
    for diff in difficulties:
        print()
        generate_ultimate_data(diff)
    
    print("\n" + "=" * 60)
    print("All ultimate datasets generated!")
    print("=" * 60)
    print("\nRecommended test order:")
    print("1. rotation_reflection - Direct Cayley vs Householder test")
    print("2. isometry - Ultimate isometry stress test")
    print("3. cumulative - Parity tracking")
