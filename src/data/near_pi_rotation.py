"""
Near-π Rotation Task Dataset: Testing Cayley's Extreme Angle Capability

============================================================================
Purpose: Test whether Cayley can handle rotations approaching π (180°).
This complements the reflection task by testing rotation limits rather than
requiring the unreachable eigenvalue λ = -1.
============================================================================

Mathematical Insight:
- Cayley eigenvalues: λ_k = exp(-2i·arctan(βμ_k/2))
- As θ → π, we need arctan(βμ/2) → π/2, which requires βμ → ∞
- Cayley CAN theoretically reach any angle in (-π, π), but π is excluded
- This dataset tests how close to π the model can successfully rotate

Key Differences from Reflection Task:
| Task        | Target      | Eigenvalue | Cayley? | Expected γ |
|-------------|-------------|------------|---------|------------|
| Reflection  | y = -x      | λ = -1     | NO      | γ → 0      |
| Near-π Rot  | y = R_θ·x   | λ ≈ e^{iπ} | YES*    | γ → 1      |

*Cayley can approach but never exactly reach θ = π.

Why This Matters:
1. Tests Cayley's behavior at its theoretical boundary
2. Validates that γ → 1 (selects Cayley) for rotation tasks
3. Complements reflection task which requires γ → 0 (Householder)
4. No symmetry-breaking initialization needed (unlike reflection)

Expected Results:
| Model      | θ = 3.0 rad | θ = 3.1 rad | θ = 3.14 rad | Mechanism           |
|------------|-------------|-------------|--------------|---------------------|
| GPT        | ~10^-3      | ~10^-2      | ~10^-1       | Approximation       |
| DDL        | ~10^-3      | ~10^-2      | ~10^-1       | Householder limited |
| E∆-MHC-Geo | ~10^-4      | ~10^-4      | ~10^-3       | Cayley rotation     |

Author: Arash Shahmansoori (2026)
"""

import os
import argparse
import numpy as np
import torch


def get_rotation_matrix(dim: int, theta: float, plane: tuple = (0, 1)) -> np.ndarray:
    """
    Generate a rotation matrix in SO(n) that rotates in the specified plane.
    
    Args:
        dim: Dimension of the space
        theta: Rotation angle in radians
        plane: Tuple (i, j) specifying the plane of rotation
        
    Returns:
        R: (dim, dim) rotation matrix in SO(dim)
    """
    R = np.eye(dim)
    i, j = plane
    c, s = np.cos(theta), np.sin(theta)
    R[i, i] = c
    R[i, j] = -s
    R[j, i] = s
    R[j, j] = c
    return R


def get_multi_plane_rotation_matrix(dim: int, theta: float, n_planes: int = None) -> np.ndarray:
    """
    Generate a rotation matrix that rotates by theta in multiple orthogonal planes.
    
    For even dimensions, this can create rotations that approach -I as θ → π.
    
    Args:
        dim: Dimension of the space (should be even for full coverage)
        theta: Rotation angle in radians (applied to each plane)
        n_planes: Number of planes to rotate in (default: dim // 2)
        
    Returns:
        R: (dim, dim) rotation matrix in SO(dim)
    """
    if n_planes is None:
        n_planes = dim // 2
    
    R = np.eye(dim)
    c, s = np.cos(theta), np.sin(theta)
    
    for p in range(n_planes):
        i, j = 2 * p, 2 * p + 1
        if j < dim:
            R[i, i] = c
            R[i, j] = -s
            R[j, i] = s
            R[j, j] = c
    
    return R


def generate_near_pi_rotation_data(
    n_samples: int,
    dim: int,
    theta: float = 3.1,
    rotation_mode: str = 'single_plane',
    device: str = 'cuda',
    seed: int = None
) -> tuple:
    """
    Generate near-π rotation task data: y = R_θ · x where θ ≈ π.
    
    This tests Cayley's ability to handle extreme rotation angles.
    Unlike reflection (y = -x), this IS achievable by Cayley (for θ < π).
    
    Args:
        n_samples: Number of samples to generate
        dim: Vector dimension
        theta: Rotation angle in radians (should be close to but less than π)
        rotation_mode: 'single_plane' or 'multi_plane'
            - 'single_plane': Rotate only in plane (0, 1)
            - 'multi_plane': Rotate in all dim//2 planes (approaches -I as θ→π)
        device: torch device ('cuda' or 'cpu')
        seed: Random seed for reproducibility
        
    Returns:
        x: Input tensor of shape (n_samples, dim)
        y: Target tensor of shape (n_samples, dim), where y = R_θ · x
    """
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
    
    # Validate theta
    if theta >= np.pi:
        print(f"Warning: theta={theta:.4f} >= π. Clamping to π - 0.001")
        theta = np.pi - 0.001
    
    # Generate rotation matrix
    if rotation_mode == 'single_plane':
        R = get_rotation_matrix(dim, theta, plane=(0, 1))
    elif rotation_mode == 'multi_plane':
        R = get_multi_plane_rotation_matrix(dim, theta)
    else:
        raise ValueError(f"Unknown rotation_mode: {rotation_mode}")
    
    R_torch = torch.from_numpy(R).float().to(device)
    
    # Generate random unit vectors
    x = torch.randn(n_samples, dim, device=device)
    x = torch.nn.functional.normalize(x, dim=-1)  # Unit vectors
    
    # Apply rotation: y = R · x
    y = x @ R_torch.T
    
    return x, y


def generate_near_pi_trajectory_data(
    n_trajectories: int,
    dim: int,
    seq_len: int = 128,
    theta: float = 3.1,
    rotation_mode: str = 'single_plane',
    seed: int = None
) -> tuple:
    """
    Generate sequential near-π rotation trajectories (3D data like gyroscope).
    
    Each trajectory applies the rotation repeatedly:
    v_0 → v_1 = R·v_0 → v_2 = R·v_1 → ... → v_T = R^T·v_0
    
    This allows:
    1. Testing model behavior over multiple steps
    2. Observing cumulative rotation effects
    3. Consistent 3D format with gyroscope/stability datasets
    
    Args:
        n_trajectories: Number of trajectories to generate
        dim: Vector dimension
        seq_len: Trajectory length (number of steps)
        theta: Rotation angle per step in radians
        rotation_mode: 'single_plane' or 'multi_plane'
        seed: Random seed
        
    Returns:
        X: Input trajectories (n_trajectories, seq_len-1, dim) - positions t=0 to T-1
        Y: Target trajectories (n_trajectories, seq_len-1, dim) - positions t=1 to T
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Validate theta
    if theta >= np.pi:
        print(f"Warning: theta={theta:.4f} >= π. Clamping to π - 0.001")
        theta = np.pi - 0.001
    
    # Generate rotation matrix
    if rotation_mode == 'single_plane':
        R = get_rotation_matrix(dim, theta, plane=(0, 1))
    elif rotation_mode == 'multi_plane':
        R = get_multi_plane_rotation_matrix(dim, theta)
    else:
        raise ValueError(f"Unknown rotation_mode: {rotation_mode}")
    
    data_x = []
    data_y = []
    
    for i in range(n_trajectories):
        # Start with random unit vector
        v = np.random.randn(dim)
        v = v / np.linalg.norm(v)
        
        # Generate trajectory by repeated rotation
        traj = [v.copy()]
        for _ in range(seq_len - 1):
            v = R @ v
            traj.append(v.copy())
        
        traj = np.stack(traj)  # (seq_len, dim)
        
        # Input: t=0 to T-1, Target: t=1 to T
        data_x.append(traj[:-1])
        data_y.append(traj[1:])
    
    X = np.stack(data_x).astype(np.float32)
    Y = np.stack(data_y).astype(np.float32)
    
    return X, Y


def generate_near_pi_rotation_dataset(
    output_dir: str = 'data/near_pi_rotation',
    dim: int = 64,
    seq_len: int = 128,
    n_train: int = 1000,
    n_val: int = 500,
    theta: float = 3.1,
    rotation_mode: str = 'single_plane',
    seed: int = 42
):
    """
    Generate and save near-π rotation TRAJECTORY dataset to disk.
    
    Generates 3D sequential data (like gyroscope) where each step applies
    a rotation by θ. This tests whether the model can handle repeated
    near-π rotations over a sequence.
    
    Args:
        output_dir: Directory to save the dataset
        dim: Vector dimension
        seq_len: Trajectory length (number of steps)
        n_train: Number of training trajectories
        n_val: Number of validation trajectories
        theta: Rotation angle per step in radians (close to π ≈ 3.14159)
        rotation_mode: 'single_plane' or 'multi_plane'
        seed: Random seed for reproducibility
    """
    os.makedirs(output_dir, exist_ok=True)
    np.random.seed(seed)
    
    print(f"Generating near-π rotation TRAJECTORY dataset...")
    print(f"  Dimension: {dim}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Train trajectories: {n_train}")
    print(f"  Val trajectories: {n_val}")
    print(f"  Rotation angle θ: {theta:.4f} rad ({np.degrees(theta):.2f}°)")
    print(f"  Distance from π: {np.pi - theta:.6f} rad ({np.degrees(np.pi - theta):.4f}°)")
    print(f"  Rotation mode: {rotation_mode}")
    
    # Generate rotation matrix
    if rotation_mode == 'single_plane':
        R = get_rotation_matrix(dim, theta, plane=(0, 1))
    else:
        R = get_multi_plane_rotation_matrix(dim, theta)
    
    # Verify orthogonality
    ortho_error = np.linalg.norm(R @ R.T - np.eye(dim))
    det_R = np.linalg.det(R)
    print(f"  Rotation matrix orthogonality error: {ortho_error:.2e}")
    print(f"  det(R) = {det_R:.6f} (should be +1 for SO(n))")
    
    # Eigenvalue analysis
    eigvals = np.linalg.eigvals(R)
    closest_to_minus1 = eigvals[np.argmin(np.abs(eigvals + 1))]
    n_near_minus1 = np.sum(np.abs(eigvals + 1) < 0.1)
    print(f"  Eigenvalue closest to -1: {closest_to_minus1:.4f}")
    print(f"  Eigenvalues near -1 (|λ+1| < 0.1): {n_near_minus1}/{dim}")
    
    # Expected gate behavior
    if rotation_mode == 'single_plane':
        expected_gate = "γ → 1 (Cayley) - only 2 eigenvalues near -1"
    else:
        expected_gate = "γ → 0 (Householder) - ALL eigenvalues near -1"
    print(f"  Expected gate: {expected_gate}")
    
    # Generate trajectories
    print(f"\nGenerating trajectories...")
    train_x, train_y = generate_near_pi_trajectory_data(
        n_trajectories=n_train, dim=dim, seq_len=seq_len,
        theta=theta, rotation_mode=rotation_mode, seed=seed
    )
    val_x, val_y = generate_near_pi_trajectory_data(
        n_trajectories=n_val, dim=dim, seq_len=seq_len,
        theta=theta, rotation_mode=rotation_mode, seed=seed + 1000
    )
    
    # Verify isometry (norms should be preserved)
    train_norms = np.linalg.norm(train_y, axis=-1)
    norm_deviation = np.abs(train_norms - 1.0).mean()
    print(f"  Output norm deviation from 1.0: {norm_deviation:.2e}")
    
    # Save as numpy arrays
    np.save(os.path.join(output_dir, 'train_x.npy'), train_x)
    np.save(os.path.join(output_dir, 'train_y.npy'), train_y)
    np.save(os.path.join(output_dir, 'val_x.npy'), val_x)
    np.save(os.path.join(output_dir, 'val_y.npy'), val_y)
    
    # Save rotation matrix for analysis
    np.save(os.path.join(output_dir, 'rotation_matrix.npy'), R.astype(np.float32))
    
    # Save metadata
    metadata = {
        'dim': dim,
        'seq_len': seq_len,
        'n_train': n_train,
        'n_val': n_val,
        'theta': theta,
        'theta_degrees': np.degrees(theta),
        'rotation_mode': rotation_mode,
        'task': 'near_pi_rotation_trajectory',
        'description': f'v_{{t+1}} = R_θ·v_t where θ = {theta:.4f} rad (near π)',
        'expected_gate': expected_gate,
        'eigenvalues_near_minus1': int(n_near_minus1),
    }
    np.save(os.path.join(output_dir, 'metadata.npy'), metadata)
    
    print(f"\nNear-π rotation trajectory dataset saved to {output_dir}/")
    print(f"  Train: {train_x.shape} -> {train_y.shape}")
    print(f"  Val: {val_x.shape} -> {val_y.shape}")
    print(f"  Storage: {(train_x.nbytes + train_y.nbytes + val_x.nbytes + val_y.nbytes) / 1e6:.2f} MB")


def generate_theta_sweep_datasets(
    base_output_dir: str = 'data/near_pi_rotation_sweep',
    dim: int = 64,
    n_train: int = 500,
    n_val: int = 200,
    thetas: list = None,
    rotation_mode: str = 'single_plane',
    seed: int = 42
):
    """
    Generate multiple datasets with varying θ values for systematic analysis.
    
    This allows studying how model performance degrades as θ → π.
    
    Args:
        base_output_dir: Base directory for datasets
        dim: Vector dimension
        n_train: Training samples per theta
        n_val: Validation samples per theta
        thetas: List of theta values to test (default: [2.5, 2.8, 3.0, 3.1, 3.13, 3.14])
        rotation_mode: 'single_plane' or 'multi_plane'
        seed: Random seed
    """
    if thetas is None:
        # Default: test increasingly extreme angles approaching π ≈ 3.14159
        thetas = [2.5, 2.8, 3.0, 3.1, 3.13, 3.14]
    
    os.makedirs(base_output_dir, exist_ok=True)
    
    print(f"Generating theta sweep datasets...")
    print(f"  Thetas: {thetas}")
    print(f"  π = {np.pi:.6f}")
    print()
    
    results = []
    for theta in thetas:
        theta_str = f"theta_{theta:.2f}".replace('.', '_')
        output_dir = os.path.join(base_output_dir, theta_str)
        
        generate_near_pi_rotation_dataset(
            output_dir=output_dir,
            dim=dim,
            n_train=n_train,
            n_val=n_val,
            theta=theta,
            rotation_mode=rotation_mode,
            seed=seed
        )
        
        results.append({
            'theta': theta,
            'theta_degrees': np.degrees(theta),
            'distance_from_pi': np.pi - theta,
            'output_dir': output_dir
        })
        print()
    
    # Save sweep metadata
    sweep_metadata = {
        'thetas': thetas,
        'dim': dim,
        'n_train': n_train,
        'n_val': n_val,
        'rotation_mode': rotation_mode,
        'results': results
    }
    np.save(os.path.join(base_output_dir, 'sweep_metadata.npy'), sweep_metadata)
    
    print(f"Theta sweep complete. Metadata saved to {base_output_dir}/sweep_metadata.npy")


def load_near_pi_rotation_dataset(data_dir: str = 'data/near_pi_rotation', device: str = 'cuda') -> dict:
    """
    Load near-π rotation dataset from disk.
    
    Args:
        data_dir: Directory containing the dataset
        device: torch device to load data to
        
    Returns:
        Dictionary with train_x, train_y, val_x, val_y tensors and metadata
    """
    data = {
        'train_x': torch.from_numpy(np.load(os.path.join(data_dir, 'train_x.npy'))).to(device),
        'train_y': torch.from_numpy(np.load(os.path.join(data_dir, 'train_y.npy'))).to(device),
        'val_x': torch.from_numpy(np.load(os.path.join(data_dir, 'val_x.npy'))).to(device),
        'val_y': torch.from_numpy(np.load(os.path.join(data_dir, 'val_y.npy'))).to(device),
    }
    
    # Load rotation matrix if available
    R_path = os.path.join(data_dir, 'rotation_matrix.npy')
    if os.path.exists(R_path):
        data['rotation_matrix'] = torch.from_numpy(np.load(R_path)).to(device)
    
    # Load metadata if available
    metadata_path = os.path.join(data_dir, 'metadata.npy')
    if os.path.exists(metadata_path):
        data['metadata'] = np.load(metadata_path, allow_pickle=True).item()
    
    print(f"Loaded near-π rotation dataset from {data_dir}/")
    print(f"  Train: {data['train_x'].shape} -> {data['train_y'].shape}")
    print(f"  Val: {data['val_x'].shape} -> {data['val_y'].shape}")
    if 'metadata' in data:
        print(f"  Theta: {data['metadata'].get('theta', 'N/A'):.4f} rad")
    
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate Near-π Rotation dataset')
    parser.add_argument('--output_dir', type=str, default='data/near_pi_rotation')
    parser.add_argument('--dim', type=int, default=64)
    parser.add_argument('--seq_len', type=int, default=128,
                        help='Trajectory length (sequence length)')
    parser.add_argument('--n_train', type=int, default=1000)
    parser.add_argument('--n_val', type=int, default=500)
    parser.add_argument('--theta', type=float, default=3.1, 
                        help='Rotation angle in radians (default: 3.1, π ≈ 3.14159)')
    parser.add_argument('--rotation_mode', type=str, default='single_plane',
                        choices=['single_plane', 'multi_plane'],
                        help='single_plane: rotate in (0,1) plane; multi_plane: rotate in all dim//2 planes')
    parser.add_argument('--sweep', action='store_true',
                        help='Generate theta sweep datasets for systematic analysis')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    if args.sweep:
        generate_theta_sweep_datasets(
            base_output_dir=args.output_dir + '_sweep',
            dim=args.dim,
            n_train=args.n_train,
            n_val=args.n_val,
            rotation_mode=args.rotation_mode,
            seed=args.seed
        )
    else:
        generate_near_pi_rotation_dataset(
            output_dir=args.output_dir,
            dim=args.dim,
            seq_len=args.seq_len,
            n_train=args.n_train,
            n_val=args.n_val,
            theta=args.theta,
            rotation_mode=args.rotation_mode,
            seed=args.seed
        )
