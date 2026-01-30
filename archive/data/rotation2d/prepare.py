"""
2D Rotation Prediction Task - A TRUE geometric task for testing rotation inductive bias.

Task: Given a sequence of 2D points and their rotated versions, predict the rotation angle.

Format: "x1,y1 x2,y2 ... -> x1',y1' x2',y2' ... = R{angle}"

This task has ACTUAL geometric structure:
- Points are rotated by SO(2) transformations
- The model must learn to recognize rotation patterns
- Cayley rotation SHOULD help here (same mathematical structure)

Example:
  Input: "1.0,0.0 0.0,1.0 -> 0.0,1.0 -1.0,0.0 = R90"
  (Points rotated 90 degrees counterclockwise)
"""

import os
import random
import numpy as np
import math


def encode(s):
    """ASCII encoding."""
    return [ord(c) for c in s]


def rotate_point(x, y, angle_deg):
    """Rotate a 2D point by angle (degrees)."""
    angle_rad = math.radians(angle_deg)
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    x_new = cos_a * x - sin_a * y
    y_new = sin_a * x + cos_a * y
    return x_new, y_new


def format_coord(val):
    """Format coordinate to 1 decimal place."""
    return f"{val:.1f}"


def generate_rotation_sample(n_points=3, discrete_angles=True):
    """
    Generate a rotation prediction sample.
    
    Args:
        n_points: Number of 2D points
        discrete_angles: If True, use discrete angles (0, 45, 90, ..., 315)
                        If False, use continuous angles
    """
    # Generate random points in [-2, 2] x [-2, 2]
    points = [(random.uniform(-2, 2), random.uniform(-2, 2)) for _ in range(n_points)]
    
    # Choose rotation angle
    if discrete_angles:
        # 8 discrete angles for easier classification
        angle = random.choice([0, 45, 90, 135, 180, 225, 270, 315])
    else:
        angle = random.randint(0, 359)
    
    # Rotate all points
    rotated_points = [rotate_point(x, y, angle) for x, y in points]
    
    # Format as string
    original_str = " ".join([f"{format_coord(x)},{format_coord(y)}" for x, y in points])
    rotated_str = " ".join([f"{format_coord(x)},{format_coord(y)}" for x, y in rotated_points])
    
    # The answer is the rotation angle
    sample = f"{original_str} -> {rotated_str} = R{angle}\n"
    
    return sample


def generate_composition_sample(n_rotations=2):
    """
    Generate a rotation COMPOSITION sample.
    
    This tests if the model can compose rotations (R1 * R2 = R3).
    
    Example: "R90 R90 = R180"
    """
    angles = [random.choice([0, 45, 90, 135, 180, 225, 270, 315]) for _ in range(n_rotations)]
    total_angle = sum(angles) % 360
    
    rotations_str = " ".join([f"R{a}" for a in angles])
    sample = f"{rotations_str} = R{total_angle}\n"
    
    return sample


def generate_inverse_sample():
    """
    Generate a rotation INVERSE sample.
    
    This tests if the model understands R * R^{-1} = I.
    
    Example: "R90 inv = R270"
    """
    angle = random.choice([0, 45, 90, 135, 180, 225, 270, 315])
    inverse_angle = (360 - angle) % 360
    
    sample = f"R{angle} inv = R{inverse_angle}\n"
    
    return sample


def generate_mixed_sample():
    """Generate a mix of task types."""
    task_type = random.choice(['rotation', 'composition', 'inverse'])
    
    if task_type == 'rotation':
        n_points = random.randint(2, 4)
        return generate_rotation_sample(n_points=n_points, discrete_angles=True)
    elif task_type == 'composition':
        n_rotations = random.randint(2, 4)
        return generate_composition_sample(n_rotations=n_rotations)
    else:
        return generate_inverse_sample()


if __name__ == '__main__':
    os.makedirs('data/rotation2d', exist_ok=True)
    
    print("Generating 2D rotation dataset...")
    
    # Generate samples
    data = []
    
    # Mix of different task types
    for _ in range(15000):
        data.append(generate_rotation_sample(n_points=3, discrete_angles=True))
    
    for _ in range(10000):
        data.append(generate_composition_sample(n_rotations=2))
    
    for _ in range(5000):
        data.append(generate_composition_sample(n_rotations=3))
    
    for _ in range(5000):
        data.append(generate_inverse_sample())
    
    # Shuffle
    random.seed(42)
    random.shuffle(data)
    
    # Print examples
    print("\nExample samples:")
    for i in range(5):
        print(f"  {data[i].strip()}")
    
    # Split train/val
    n = len(data)
    train_data = "".join(data[:int(n*0.9)])
    val_data = "".join(data[int(n*0.9):])
    
    # Save
    train_ids = np.array(encode(train_data), dtype=np.uint16)
    val_ids = np.array(encode(val_data), dtype=np.uint16)
    
    train_ids.tofile('data/rotation2d/train.bin')
    val_ids.tofile('data/rotation2d/val.bin')
    
    print(f"\nSaved {len(train_ids):,} train tokens, {len(val_ids):,} val tokens")
    print(f"Total samples: {n:,}")
