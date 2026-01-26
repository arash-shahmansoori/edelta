"""
Continuous Gyroscope Dataset - The "Kill Shot" for DDL

This dataset generates sequences of continuously rotating N-dimensional vectors.
It is specifically designed to expose the fundamental weakness of Deep Delta Learning (DDL):

MATHEMATICAL INSIGHT:
- DDL uses rank-1 linear updates: x' = x - β(k·x)k
- This walks in STRAIGHT LINES (chords) instead of CURVES (arcs)
- Result: Norm drift - vectors spiral outward (energy leak)

- Cayley uses manifold-preserving rotations: Q = (I+M)^{-1}(I-M)  
- This walks EXACTLY on the rotation manifold SO(n)
- Result: Perfect norm preservation ||Qx|| = ||x||

THE TASK:
- Input: Sequence of rotating vector coordinates (tokenized)
- Goal: Predict the next rotated state
- Success metric: Norm stability over long sequences

WHY DDL FAILS:
- Small angles (θ < 15°): DDL ≈ Cayley (linear approximation is OK)
- Large angles (θ > 45°): DDL accumulates error, vector explodes
- Over 100 steps at θ=30°: DDL norm grows by ~4x, Cayley stays at 1.0

EXPECTED RESULTS:
| Method      | Norm After 100 Steps | Status      |
|-------------|---------------------|-------------|
| DDL         | 2.0 - 10.0          | EXPLODED    |
| Hybrid      | 1.0 - 1.01          | STABLE      |
| Cayley      | 1.000000            | PERFECT     |

Author: Arash Shahmansoori (2026)
"""

import os
import numpy as np


def get_rotation_matrix(dim: int, theta: float) -> np.ndarray:
    """
    Creates a block-diagonal rotation matrix for N-dimensions.
    Rotates in planes (0,1), (2,3), (4,5), etc. by angle theta.
    
    This is a standard way to construct SO(n) rotations.
    Each 2x2 block is:
        [[cos(θ), -sin(θ)],
         [sin(θ),  cos(θ)]]
    
    Args:
        dim: Dimension of the space (must be even for all components to rotate)
        theta: Rotation angle in radians
        
    Returns:
        R: Orthogonal rotation matrix with det(R) = +1
    """
    R = np.eye(dim)
    c, s = np.cos(theta), np.sin(theta)
    
    # Fill diagonal 2x2 blocks
    for i in range(0, dim - 1, 2):
        R[i, i] = c
        R[i, i + 1] = -s
        R[i + 1, i] = s
        R[i + 1, i + 1] = c
    
    return R


def quantize(vector: np.ndarray, bins: int = 256, 
             range_min: float = -2.0, range_max: float = 2.0) -> np.ndarray:
    """
    Maps continuous floats to discrete token IDs for language model processing.
    
    Args:
        vector: Continuous vector to quantize
        bins: Number of discrete bins (vocabulary size)
        range_min: Minimum expected value
        range_max: Maximum expected value
        
    Returns:
        Token IDs in range [0, bins-1]
    """
    vector = np.clip(vector, range_min, range_max)
    norm = (vector - range_min) / (range_max - range_min)
    ids = (norm * (bins - 1)).astype(np.int32)
    return ids


def dequantize(ids: np.ndarray, bins: int = 256,
               range_min: float = -2.0, range_max: float = 2.0) -> np.ndarray:
    """
    Inverse of quantize - maps token IDs back to continuous values.
    Used for validation/visualization.
    """
    norm = ids / (bins - 1)
    vector = norm * (range_max - range_min) + range_min
    return vector


def generate_gyroscope_sequence(dim: int, seq_len: int, theta: float, 
                                 bins: int, initial_v: np.ndarray = None) -> tuple:
    """
    Generate a single spinning gyroscope sequence.
    
    Args:
        dim: Dimension of the vector
        seq_len: Number of rotation steps
        theta: Rotation angle per step (radians)
        bins: Quantization bins
        initial_v: Initial vector (random unit vector if None)
        
    Returns:
        token_ids: Flattened sequence of quantized coordinates
        metadata: Dict with theta, initial/final vectors, norms
    """
    # Initialize with random unit vector
    if initial_v is None:
        v = np.random.randn(dim)
        v = v / np.linalg.norm(v)
    else:
        v = initial_v.copy()
    
    # Get rotation matrix
    R = get_rotation_matrix(dim, theta)
    
    # Generate sequence
    token_ids = []
    norms = [np.linalg.norm(v)]
    vectors = [v.copy()]
    
    current_v = v
    for t in range(seq_len):
        # Tokenize current vector
        toks = quantize(current_v, bins=bins)
        token_ids.extend(toks.tolist())
        
        # Rotate for next step
        current_v = R @ current_v
        norms.append(np.linalg.norm(current_v))
        vectors.append(current_v.copy())
    
    metadata = {
        'theta': theta,
        'initial_v': v,
        'final_v': current_v,
        'norms': norms,
        'vectors': vectors
    }
    
    return token_ids, metadata


def prepare_dataset(
    dim: int = 16,
    seq_len: int = 64,      # Number of rotation steps per sequence
    num_samples: int = 10000,
    bins: int = 128,        # Vocabulary size
    theta_range: tuple = (0.1, 1.5),  # Radians: ~5° to ~86°
    include_large_angles: bool = True,  # Include "kill zone" for DDL
    output_dir: str = 'data/gyroscope'
):
    """
    Generate the full Continuous Gyroscope dataset.
    
    The dataset includes a mix of rotation speeds:
    - Small angles (θ < 0.3): DDL works reasonably well
    - Medium angles (0.3 < θ < 0.8): DDL starts to struggle  
    - Large angles (θ > 0.8): DDL fails catastrophically
    
    This allows us to demonstrate the "intelligent gearbox" behavior
    where the Hybrid model learns to use Cayley for large angles.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    all_ids = []
    all_metadata = []
    
    print(f"Generating {num_samples} gyroscope sequences...")
    print(f"  Dimension: {dim}")
    print(f"  Sequence length: {seq_len} rotation steps")
    print(f"  Tokens per sequence: {seq_len * dim}")
    print(f"  Vocabulary size: {bins}")
    print(f"  Theta range: {np.degrees(theta_range[0]):.1f}° to {np.degrees(theta_range[1]):.1f}°")
    
    # Distribution of angles (weighted towards challenging cases)
    angle_distribution = {
        'small': (0.05, 0.3),   # 3° - 17° (DDL OK)
        'medium': (0.3, 0.8),   # 17° - 46° (DDL struggles)
        'large': (0.8, 1.57),   # 46° - 90° (DDL fails)
    }
    
    for i in range(num_samples):
        # Sample rotation angle with bias towards challenging cases
        if include_large_angles:
            category = np.random.choice(['small', 'medium', 'large'], 
                                        p=[0.2, 0.3, 0.5])  # More large angles
            theta_min, theta_max = angle_distribution[category]
        else:
            theta_min, theta_max = theta_range
        
        theta = np.random.uniform(theta_min, theta_max)
        
        # Generate sequence
        token_ids, metadata = generate_gyroscope_sequence(
            dim=dim, seq_len=seq_len, theta=theta, bins=bins
        )
        
        all_ids.extend(token_ids)
        all_metadata.append({
            'sample_idx': i,
            'theta': theta,
            'theta_degrees': np.degrees(theta),
            'category': category if include_large_angles else 'mixed'
        })
        
        if (i + 1) % 1000 == 0:
            print(f"  Generated {i + 1}/{num_samples} sequences")
    
    # Split into train/val
    total_tokens = len(all_ids)
    val_split = int(total_tokens * 0.9)
    
    train_ids = np.array(all_ids[:val_split], dtype=np.uint16)
    val_ids = np.array(all_ids[val_split:], dtype=np.uint16)
    
    # Save binary files
    train_ids.tofile(os.path.join(output_dir, 'train.bin'))
    val_ids.tofile(os.path.join(output_dir, 'val.bin'))
    
    # Save metadata for analysis
    meta_file = os.path.join(output_dir, 'meta.txt')
    with open(meta_file, 'w') as f:
        f.write(f"Continuous Gyroscope Dataset\n")
        f.write(f"=" * 50 + "\n")
        f.write(f"Vector dimension: {dim}\n")
        f.write(f"Sequence length: {seq_len} steps\n")
        f.write(f"Tokens per step: {dim}\n")
        f.write(f"Total tokens per sequence: {seq_len * dim}\n")
        f.write(f"Number of samples: {num_samples}\n")
        f.write(f"Vocabulary size: {bins}\n")
        f.write(f"Theta range: {np.degrees(theta_range[0]):.1f}° to {np.degrees(theta_range[1]):.1f}°\n")
        f.write(f"\nTrain tokens: {len(train_ids)}\n")
        f.write(f"Val tokens: {len(val_ids)}\n")
        f.write(f"\nAngle distribution:\n")
        for cat, (tmin, tmax) in angle_distribution.items():
            count = sum(1 for m in all_metadata if m.get('category') == cat)
            f.write(f"  {cat}: {np.degrees(tmin):.1f}° - {np.degrees(tmax):.1f}° ({count} samples)\n")
    
    print(f"\nDataset saved to {output_dir}/")
    print(f"  train.bin: {len(train_ids):,} tokens")
    print(f"  val.bin: {len(val_ids):,} tokens")
    print(f"  meta.txt: Dataset metadata")
    
    return train_ids, val_ids, all_metadata


def generate_stability_test(dim: int = 16, 
                            num_steps: int = 200,
                            theta_values: list = None,
                            output_file: str = 'data/gyroscope/stability_test.npz'):
    """
    Generate a separate test set for stability analysis.
    
    This is used to measure:
    1. Norm drift over time for each method
    2. "Time to explosion" (when ||x|| > 2)
    3. Gate behavior in Hybrid model
    
    Saves raw vectors (not tokenized) for precise analysis.
    """
    if theta_values is None:
        # Test at multiple angles: 5°, 15°, 30°, 45°, 60°, 90°
        theta_values = [np.radians(d) for d in [5, 15, 30, 45, 60, 90]]
    
    test_data = {}
    
    for theta in theta_values:
        # Start with standard unit vectors
        sequences = []
        for _ in range(10):  # 10 random starting vectors per angle
            v = np.random.randn(dim)
            v = v / np.linalg.norm(v)
            
            R = get_rotation_matrix(dim, theta)
            trajectory = [v.copy()]
            current = v
            
            for _ in range(num_steps):
                current = R @ current
                trajectory.append(current.copy())
            
            sequences.append(np.array(trajectory))
        
        test_data[f'theta_{np.degrees(theta):.0f}'] = np.array(sequences)
    
    np.savez(output_file, **test_data)
    print(f"Stability test data saved to {output_file}")
    
    return test_data


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate Continuous Gyroscope Dataset')
    parser.add_argument('--dim', type=int, default=16, help='Vector dimension')
    parser.add_argument('--seq_len', type=int, default=64, help='Steps per sequence')
    parser.add_argument('--num_samples', type=int, default=10000, help='Number of sequences')
    parser.add_argument('--bins', type=int, default=128, help='Vocabulary size')
    parser.add_argument('--stability_test', action='store_true', help='Also generate stability test')
    
    args = parser.parse_args()
    
    # Generate main dataset
    prepare_dataset(
        dim=args.dim,
        seq_len=args.seq_len,
        num_samples=args.num_samples,
        bins=args.bins,
        include_large_angles=True
    )
    
    # Optionally generate stability test data
    if args.stability_test:
        generate_stability_test(dim=args.dim)
