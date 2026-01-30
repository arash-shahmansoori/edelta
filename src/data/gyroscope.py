"""
Dataset 1: The "Continuous Gyroscope" (Geometric Regression)

Target: Proves MANIFOLD PRECISION vs. DDL's LINEAR APPROXIMATION.

Why this is the "Kill-Shot":
- Forces the model to predict continuous rotations in high-dimensional space.
- DDL uses a linear step (x + δ). To rotate 90° on a circle, a linear step must go 
  "through" the circle (shrinking norm) or "tangent" to it (increasing norm).
- E∆-MHC-Geo creates a perfect arc via Cayley transform. Can rotate 179° in a 
  single step with zero error.

Expected Results:
- DDL's error spikes exponentially as θ → π
- E∆-MHC-Geo's error remains flat (near zero) across all angles
"""

import numpy as np
import os
import argparse


def get_rotation_matrix(dim: int, theta: float, plane: tuple = (0, 1)) -> np.ndarray:
    """
    Generates a rotation matrix in SO(n) that rotates in the specified plane.
    
    Args:
        dim: Dimension of the space
        theta: Rotation angle in radians
        plane: Tuple (i, j) specifying the plane of rotation
        
    Returns:
        R: (dim, dim) rotation matrix
    """
    R = np.eye(dim)
    i, j = plane
    c, s = np.cos(theta), np.sin(theta)
    R[i, i] = c
    R[i, j] = -s
    R[j, i] = s
    R[j, j] = c
    return R


def generate_gyroscope_data(
    output_dir: str = 'data/gyroscope',
    dim: int = 16,
    seq_len: int = 256,
    n_train: int = 9000,
    n_val: int = 1000,
    theta_min: float = 0.1,
    theta_max: float = 2.5,
    seed: int = 42
):
    """
    Generate gyroscope dataset for continuous rotation prediction.
    
    The task: Given v_t, predict v_{t+1} = R @ v_t where R is a rotation matrix.
    
    Args:
        output_dir: Directory to save the dataset
        dim: Vector dimension (default 16)
        seq_len: Trajectory length (default 256)
        n_train: Number of training trajectories (default 9000)
        n_val: Number of validation trajectories (default 1000)
        theta_min: Minimum rotation angle in radians
        theta_max: Maximum rotation angle in radians (DDL breaks at θ > 0.5)
        seed: Random seed for reproducibility
    """
    os.makedirs(output_dir, exist_ok=True)
    np.random.seed(seed)
    
    n_total = n_train + n_val
    
    print(f"Generating {n_total} gyroscope trajectories...")
    print(f"  Dimension: {dim}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Rotation angle range: [{theta_min:.2f}, {theta_max:.2f}] radians")
    print(f"  (DDL typically breaks at θ > 0.5 radians)")
    
    data_x = []
    data_y = []
    thetas = []
    
    for i in range(n_total):
        # 1. Start with random unit vector
        v = np.random.randn(dim)
        v = v / np.linalg.norm(v)
        
        # 2. Choose random angular velocity
        theta = np.random.uniform(theta_min, theta_max)
        thetas.append(theta)
        
        # 3. Choose random rotation plane (for variety)
        plane_i = np.random.randint(0, dim - 1)
        plane_j = np.random.randint(plane_i + 1, dim)
        R = get_rotation_matrix(dim, theta, (plane_i, plane_j))
        
        # 4. Generate trajectory
        traj = [v.copy()]
        for _ in range(seq_len - 1):
            v = R @ v
            traj.append(v.copy())
            
        traj = np.stack(traj)  # (seq_len, dim)
        
        # Input: v_t, Output: v_{t+1}
        data_x.append(traj[:-1])
        data_y.append(traj[1:])
        
        if (i + 1) % 1000 == 0:
            print(f"  Generated {i + 1}/{n_total} trajectories")
    
    # Convert to Float32 Tensor
    X = np.stack(data_x).astype(np.float32)
    Y = np.stack(data_y).astype(np.float32)
    thetas = np.array(thetas).astype(np.float32)
    
    # Split into train/val
    train_x, val_x = X[:n_train], X[n_train:]
    train_y, val_y = Y[:n_train], Y[n_train:]
    train_theta, val_theta = thetas[:n_train], thetas[n_train:]
    
    # Save
    np.save(os.path.join(output_dir, 'train_x.npy'), train_x)
    np.save(os.path.join(output_dir, 'train_y.npy'), train_y)
    np.save(os.path.join(output_dir, 'train_theta.npy'), train_theta)
    np.save(os.path.join(output_dir, 'val_x.npy'), val_x)
    np.save(os.path.join(output_dir, 'val_y.npy'), val_y)
    np.save(os.path.join(output_dir, 'val_theta.npy'), val_theta)
    
    print(f"\nGyroscope dataset saved to {output_dir}/")
    print(f"  Train: {train_x.shape} -> {train_y.shape}")
    print(f"  Val: {val_x.shape} -> {val_y.shape}")
    print(f"  Storage: {(train_x.nbytes + train_y.nbytes + val_x.nbytes + val_y.nbytes) / 1e6:.1f} MB")
    
    # Create stratified validation sets for analysis
    # Split validation by rotation angle: slow (θ < 0.5) vs fast (θ > 1.5)
    slow_mask = val_theta < 0.5
    fast_mask = val_theta > 1.5
    
    if slow_mask.sum() > 0:
        np.save(os.path.join(output_dir, 'val_slow_x.npy'), val_x[slow_mask])
        np.save(os.path.join(output_dir, 'val_slow_y.npy'), val_y[slow_mask])
        print(f"  Val (slow, θ < 0.5): {slow_mask.sum()} trajectories")
    
    if fast_mask.sum() > 0:
        np.save(os.path.join(output_dir, 'val_fast_x.npy'), val_x[fast_mask])
        np.save(os.path.join(output_dir, 'val_fast_y.npy'), val_y[fast_mask])
        print(f"  Val (fast, θ > 1.5): {fast_mask.sum()} trajectories")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate Gyroscope dataset')
    parser.add_argument('--output_dir', type=str, default='data/gyroscope')
    parser.add_argument('--dim', type=int, default=16)
    parser.add_argument('--seq_len', type=int, default=256)
    parser.add_argument('--n_train', type=int, default=9000)
    parser.add_argument('--n_val', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    generate_gyroscope_data(
        output_dir=args.output_dir,
        dim=args.dim,
        seq_len=args.seq_len,
        n_train=args.n_train,
        n_val=args.n_val,
        seed=args.seed
    )
