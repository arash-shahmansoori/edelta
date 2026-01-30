"""
Reflection Task Dataset: The "Negation Kill-Shot"

============================================================================
Purpose: Test geometric reflection capability - the PUREST test of whether
models can learn exact negation y = -x.
============================================================================

Mathematical Insight:
- Householder with β=2: H·x = x - 2(k·x)k = -x when k = x/||x|| (perfect!)
- Standard networks: Must learn f(x) ≈ -2x, then x + f(x) = -x (approximation)
- DDL: Can achieve this when β→2 and k→x/||x||

Why This Matters:
1. Tests the fundamental capability of geometric operators
2. Sample efficiency reveals inductive bias strength
3. Parameter convergence (β→2, gate→0) proves geometric learning

Expected Results:
| Model      | Samples Needed | β/Gate | Mechanism              |
|------------|---------------|--------|------------------------|
| GPT        | >200          | N/A    | Must approximate       |
| DDL        | ~50           | β→2    | Learns reflection      |
| mHC        | >200          | N/A    | Sinkhorn can't negate  |
| E∆-MHC-Geo | ~25           | γ→0    | Uses Householder       |

Author: Arash Shahmansoori (2026)
"""

import os
import argparse
import numpy as np
import torch


def generate_negation_data(
    n_samples: int,
    dim: int,
    device: str = 'cuda',
    seed: int = None
) -> tuple:
    """
    Generate reflection task data: y = -x.
    
    This is the PUREST test of reflection capability:
    - Householder with β=2: H·x = x - 2x = -x (exact)
    - Standard networks: Must learn f(x) ≈ -2x, then x + f(x) = -x
    
    Args:
        n_samples: Number of samples to generate
        dim: Vector dimension
        device: torch device ('cuda' or 'cpu')
        seed: Random seed for reproducibility
        
    Returns:
        x: Input tensor of shape (n_samples, dim)
        y: Target tensor of shape (n_samples, dim), where y = -x
    """
    if seed is not None:
        torch.manual_seed(seed)
        
    x = torch.randn(n_samples, dim, device=device)
    y = -x  # Target is negation
    
    return x, y


def generate_reflection_dataset(
    output_dir: str = 'data/reflection',
    dim: int = 64,
    n_train: int = 1000,
    n_val: int = 500,
    seed: int = 42
):
    """
    Generate and save reflection dataset to disk.
    
    Args:
        output_dir: Directory to save the dataset
        dim: Vector dimension
        n_train: Number of training samples
        n_val: Number of validation samples
        seed: Random seed for reproducibility
    """
    os.makedirs(output_dir, exist_ok=True)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    print(f"Generating reflection dataset...")
    print(f"  Dimension: {dim}")
    print(f"  Train samples: {n_train}")
    print(f"  Val samples: {n_val}")
    
    # Generate data (on CPU for saving)
    train_x = torch.randn(n_train, dim)
    train_y = -train_x
    
    val_x = torch.randn(n_val, dim)
    val_y = -val_x
    
    # Save as numpy arrays
    np.save(os.path.join(output_dir, 'train_x.npy'), train_x.numpy().astype(np.float32))
    np.save(os.path.join(output_dir, 'train_y.npy'), train_y.numpy().astype(np.float32))
    np.save(os.path.join(output_dir, 'val_x.npy'), val_x.numpy().astype(np.float32))
    np.save(os.path.join(output_dir, 'val_y.npy'), val_y.numpy().astype(np.float32))
    
    # Save metadata
    metadata = {
        'dim': dim,
        'n_train': n_train,
        'n_val': n_val,
        'task': 'negation',
        'description': 'y = -x (pure reflection task)'
    }
    np.save(os.path.join(output_dir, 'metadata.npy'), metadata)
    
    print(f"\nReflection dataset saved to {output_dir}/")
    print(f"  Train: ({n_train}, {dim}) -> ({n_train}, {dim})")
    print(f"  Val: ({n_val}, {dim}) -> ({n_val}, {dim})")
    print(f"  Storage: {(train_x.nbytes + train_y.nbytes + val_x.nbytes + val_y.nbytes) / 1e6:.2f} MB")


def load_reflection_dataset(data_dir: str = 'data/reflection', device: str = 'cuda') -> dict:
    """
    Load reflection dataset from disk.
    
    Args:
        data_dir: Directory containing the dataset
        device: torch device to load data to
        
    Returns:
        Dictionary with train_x, train_y, val_x, val_y tensors
    """
    data = {
        'train_x': torch.from_numpy(np.load(os.path.join(data_dir, 'train_x.npy'))).to(device),
        'train_y': torch.from_numpy(np.load(os.path.join(data_dir, 'train_y.npy'))).to(device),
        'val_x': torch.from_numpy(np.load(os.path.join(data_dir, 'val_x.npy'))).to(device),
        'val_y': torch.from_numpy(np.load(os.path.join(data_dir, 'val_y.npy'))).to(device),
    }
    
    # Load metadata if available
    metadata_path = os.path.join(data_dir, 'metadata.npy')
    if os.path.exists(metadata_path):
        data['metadata'] = np.load(metadata_path, allow_pickle=True).item()
    
    print(f"Loaded reflection dataset from {data_dir}/")
    print(f"  Train: {data['train_x'].shape} -> {data['train_y'].shape}")
    print(f"  Val: {data['val_x'].shape} -> {data['val_y'].shape}")
    
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate Reflection dataset')
    parser.add_argument('--output_dir', type=str, default='data/reflection')
    parser.add_argument('--dim', type=int, default=64)
    parser.add_argument('--n_train', type=int, default=1000)
    parser.add_argument('--n_val', type=int, default=500)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    generate_reflection_dataset(
        output_dir=args.output_dir,
        dim=args.dim,
        n_train=args.n_train,
        n_val=args.n_val,
        seed=args.seed
    )
