"""
Continuous Vector Rotation Dataset

This dataset tests TRUE geometric rotation capability:
- Input: N-dimensional vectors (continuous, not tokenized)
- Task: Predict the next rotated vector
- Success metric: MSE loss, norm preservation, angular accuracy

Why this tests DDC specifically:
- DDL's linear approximation causes norm explosion over time
- DDC's Cayley transform maintains ||Qx|| = ||x|| exactly
- No tokenization means the model must learn true geometric operations

Dataset structure:
- Each sample is a sequence of vectors: [v0, v1, v2, ..., vT]
- v_{t+1} = R * v_t where R is a rotation matrix
- Model predicts v_{t+1} given v_t
"""

import os
import numpy as np
import torch
import pickle

# Configuration
DIM = 16  # Vector dimension
SEQ_LEN = 32  # Number of rotation steps per sequence
N_TRAIN = 10000
N_VAL = 1000
ANGLE_RANGE = (5, 60)  # Rotation angle range in degrees

def random_rotation_matrix(dim, angle_deg):
    """Generate a random rotation matrix in SO(dim)."""
    # Generate random skew-symmetric matrix
    A = np.random.randn(dim, dim)
    A = (A - A.T) / 2  # Make skew-symmetric
    A = A / np.linalg.norm(A) * np.radians(angle_deg)  # Scale to desired angle
    
    # Cayley transform to get rotation matrix
    I = np.eye(dim)
    R = np.linalg.solve(I + A/2, I - A/2)
    return R

def generate_sequence(dim, seq_len, angle_deg):
    """Generate a sequence of rotated vectors."""
    # Random initial unit vector
    v0 = np.random.randn(dim)
    v0 = v0 / np.linalg.norm(v0)
    
    # Random rotation matrix
    R = random_rotation_matrix(dim, angle_deg)
    
    # Generate sequence
    sequence = [v0]
    v = v0
    for _ in range(seq_len - 1):
        v = R @ v
        sequence.append(v)
    
    return np.array(sequence), R, angle_deg

def create_dataset(n_samples, dim, seq_len, angle_range):
    """Create dataset of rotation sequences."""
    sequences = []
    metadata = []
    
    for i in range(n_samples):
        angle = np.random.uniform(*angle_range)
        seq, R, angle = generate_sequence(dim, seq_len, angle)
        sequences.append(seq)
        metadata.append({
            'angle': angle,
            'rotation_matrix': R,
            'initial_norm': np.linalg.norm(seq[0]),
            'final_norm': np.linalg.norm(seq[-1]),
            'norm_drift': abs(np.linalg.norm(seq[-1]) - np.linalg.norm(seq[0]))
        })
    
    return np.array(sequences), metadata

def main():
    print("Creating Continuous Vector Rotation Dataset")
    print(f"  Dimension: {DIM}")
    print(f"  Sequence length: {SEQ_LEN}")
    print(f"  Angle range: {ANGLE_RANGE} degrees")
    print(f"  Train samples: {N_TRAIN}")
    print(f"  Val samples: {N_VAL}")
    
    os.makedirs(os.path.dirname(os.path.abspath(__file__)), exist_ok=True)
    
    # Generate datasets
    print("\nGenerating training data...")
    train_data, train_meta = create_dataset(N_TRAIN, DIM, SEQ_LEN, ANGLE_RANGE)
    
    print("Generating validation data...")
    val_data, val_meta = create_dataset(N_VAL, DIM, SEQ_LEN, ANGLE_RANGE)
    
    # Verify norm preservation in ground truth
    train_norm_drift = np.mean([m['norm_drift'] for m in train_meta])
    print(f"\nGround truth norm drift (should be ~0): {train_norm_drift:.6f}")
    
    # Save as numpy arrays (continuous, not tokenized!)
    data_dir = os.path.dirname(os.path.abspath(__file__))
    
    np.save(os.path.join(data_dir, 'train.npy'), train_data.astype(np.float32))
    np.save(os.path.join(data_dir, 'val.npy'), val_data.astype(np.float32))
    
    # Save metadata
    meta = {
        'dim': DIM,
        'seq_len': SEQ_LEN,
        'angle_range': ANGLE_RANGE,
        'n_train': N_TRAIN,
        'n_val': N_VAL,
        'train_metadata': train_meta,
        'val_metadata': val_meta,
    }
    with open(os.path.join(data_dir, 'meta.pkl'), 'wb') as f:
        pickle.dump(meta, f)
    
    print(f"\nSaved to {data_dir}:")
    print(f"  train.npy: {train_data.shape}")
    print(f"  val.npy: {val_data.shape}")
    print(f"  meta.pkl")
    
    # Create stability test set with extreme angles
    print("\nCreating stability test set (large angles)...")
    stability_angles = [30, 45, 60, 90, 120]
    stability_data = {}
    for angle in stability_angles:
        seqs, _ = create_dataset(100, DIM, SEQ_LEN * 2, (angle, angle))  # Longer sequences
        stability_data[angle] = seqs
    np.savez(os.path.join(data_dir, 'stability_test.npz'), **{f'angle_{a}': d for a, d in stability_data.items()})
    print(f"  stability_test.npz: angles {stability_angles}")

if __name__ == '__main__':
    main()
