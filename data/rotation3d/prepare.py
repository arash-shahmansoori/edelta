"""
3D Rotation Prediction Dataset

Purpose: Test DDL's claim of geometric expressivity at higher dimensions.

Task: Predict the 3D rotation given original and rotated points.
Input:  "P1:(1.0,0.0,0.0) P2:(0.0,1.0,0.0) P3:(0.0,0.0,1.0) -> 
         P1':(0.0,1.0,0.0) P2':(-1.0,0.0,0.0) P3':(0.0,0.0,1.0) = ?"
Output: "Rz90" (rotation 90° around Z-axis)

This tests if the model can learn SO(3) structure.
E∆-MHC-Geo should excel because Cayley naturally represents SO(n).
"""

import os
import random
import numpy as np
from typing import Tuple

# Configuration
NUM_TRAIN = 50000
NUM_VAL = 5000

# Discrete rotation angles (degrees)
ANGLES = [0, 45, 90, 135, 180, 225, 270, 315]
AXES = ['x', 'y', 'z']

def rotation_matrix_3d(axis: str, angle_deg: float) -> np.ndarray:
    """Create a 3D rotation matrix."""
    angle = np.radians(angle_deg)
    c, s = np.cos(angle), np.sin(angle)
    
    if axis == 'x':
        return np.array([
            [1, 0, 0],
            [0, c, -s],
            [0, s, c]
        ])
    elif axis == 'y':
        return np.array([
            [c, 0, s],
            [0, 1, 0],
            [-s, 0, c]
        ])
    elif axis == 'z':
        return np.array([
            [c, -s, 0],
            [s, c, 0],
            [0, 0, 1]
        ])
    else:
        raise ValueError(f"Unknown axis: {axis}")

def format_point(p: np.ndarray) -> str:
    """Format a 3D point as string."""
    return f"({p[0]:.1f},{p[1]:.1f},{p[2]:.1f})"

def generate_sample() -> str:
    """Generate a single 3D rotation sample."""
    # Random axis and angle
    axis = random.choice(AXES)
    angle = random.choice(ANGLES)
    
    # Skip identity rotation sometimes
    if angle == 0 and random.random() < 0.5:
        angle = random.choice([a for a in ANGLES if a != 0])
    
    # Standard basis points (or random points)
    if random.random() < 0.7:
        # Use standard basis (easier)
        points = np.eye(3)
    else:
        # Use random points (harder)
        points = np.random.randn(3, 3)
        points = points / np.linalg.norm(points, axis=1, keepdims=True)
    
    # Apply rotation
    R = rotation_matrix_3d(axis, angle)
    rotated = (R @ points.T).T
    
    # Format input
    orig_strs = [f"P{i+1}:{format_point(points[i])}" for i in range(3)]
    rot_strs = [f"P{i+1}':{format_point(rotated[i])}" for i in range(3)]
    
    input_str = " ".join(orig_strs) + " -> " + " ".join(rot_strs) + " = "
    
    # Output: axis and angle
    output_str = f"R{axis}{angle}"
    
    full_seq = input_str + output_str + "\n"
    
    return full_seq

def generate_composition_sample() -> str:
    """Generate a composition of two rotations (harder)."""
    # Two random rotations
    axis1 = random.choice(AXES)
    angle1 = random.choice([90, 180, 270])
    axis2 = random.choice(AXES)
    angle2 = random.choice([90, 180, 270])
    
    # Standard basis
    points = np.eye(3)
    
    # Apply both rotations
    R1 = rotation_matrix_3d(axis1, angle1)
    R2 = rotation_matrix_3d(axis2, angle2)
    R_combined = R2 @ R1
    rotated = (R_combined @ points.T).T
    
    # Format
    orig_strs = [f"P{i+1}:{format_point(points[i])}" for i in range(3)]
    rot_strs = [f"P{i+1}':{format_point(rotated[i])}" for i in range(3)]
    
    input_str = " ".join(orig_strs) + " -> " + " ".join(rot_strs) + " = "
    output_str = f"R{axis1}{angle1}_R{axis2}{angle2}"
    
    return input_str + output_str + "\n"

def main():
    random.seed(42)
    np.random.seed(42)
    
    print("Generating 3D Rotation Prediction dataset...")
    print(f"  Train samples: {NUM_TRAIN}")
    print(f"  Val samples: {NUM_VAL}")
    print(f"  Axes: {AXES}")
    print(f"  Angles: {ANGLES}")
    
    # Generate training data (80% simple, 20% composition)
    train_data = []
    for i in range(NUM_TRAIN):
        if random.random() < 0.8:
            train_data.append(generate_sample())
        else:
            train_data.append(generate_composition_sample())
    
    # Validation data
    random.seed(123)
    np.random.seed(123)
    val_data = []
    for i in range(NUM_VAL):
        if random.random() < 0.8:
            val_data.append(generate_sample())
        else:
            val_data.append(generate_composition_sample())
    
    # Combine
    train_text = "".join(train_data)
    val_text = "".join(val_data)
    
    print(f"\nSample training examples:")
    print(f"  Simple: {train_data[0].strip()}")
    # Find a composition example
    for d in train_data:
        if "_R" in d:
            print(f"  Composition: {d.strip()}")
            break
    
    # Encode
    train_ids = np.array([ord(c) for c in train_text], dtype=np.uint16)
    val_ids = np.array([ord(c) for c in val_text], dtype=np.uint16)
    
    print(f"\nDataset sizes:")
    print(f"  Train: {len(train_ids):,} tokens")
    print(f"  Val: {len(val_ids):,} tokens")
    
    # Save
    train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
    val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))
    
    with open(os.path.join(os.path.dirname(__file__), 'meta.txt'), 'w') as f:
        f.write(f"3D Rotation Prediction Dataset\n")
        f.write(f"Train samples: {NUM_TRAIN}\n")
        f.write(f"Val samples: {NUM_VAL}\n")
        f.write(f"Purpose: Test SO(3) geometric reasoning\n")
    
    print("\nDone! Files saved: train.bin, val.bin, meta.txt")

if __name__ == "__main__":
    main()
