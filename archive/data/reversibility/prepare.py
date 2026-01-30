# data/reversibility/prepare.py
"""
Reversibility Task: Path Cancellation

This task tests whether a model can learn that operations cancel each other.

Format: "N N E S W -> (1, 1)"
- N = North (+1 in y)
- S = South (-1 in y)  [cancels N]
- E = East (+1 in x)
- W = West (-1 in x)   [cancels E]

The model must track cumulative position and output final (x, y).

This task should benefit from rotation because:
1. N and S are "inverses" (cancel each other)
2. E and W are "inverses" (cancel each other)
3. The 2D space has rotational symmetry (90° rotation swaps axes)
4. A unitary rotation in embedding space could naturally represent these cancellations
"""

import os
import random
import numpy as np

# Direction mappings
DIRECTIONS = ['N', 'S', 'E', 'W']
DIR_TO_DELTA = {
    'N': (0, 1),
    'S': (0, -1),
    'E': (1, 0),
    'W': (-1, 0)
}

def generate_path(min_steps=3, max_steps=12):
    """Generate a random path and compute final position."""
    n_steps = random.randint(min_steps, max_steps)
    
    # Generate random directions
    path = [random.choice(DIRECTIONS) for _ in range(n_steps)]
    
    # Compute final position
    x, y = 0, 0
    for d in path:
        dx, dy = DIR_TO_DELTA[d]
        x += dx
        y += dy
    
    # Format: "N E S W -> (x, y)\n"
    path_str = ' '.join(path)
    result_str = f"({x},{y})"
    
    return f"{path_str} -> {result_str}\n"

def generate_cancellation_heavy_path(min_steps=4, max_steps=12):
    """Generate paths with intentional cancellations to test reversibility."""
    n_steps = random.randint(min_steps, max_steps)
    
    path = []
    x, y = 0, 0
    
    for i in range(n_steps):
        if random.random() < 0.4 and len(path) > 0:
            # 40% chance to add a cancelling direction
            last = path[-1]
            if last == 'N':
                d = 'S'
            elif last == 'S':
                d = 'N'
            elif last == 'E':
                d = 'W'
            else:
                d = 'E'
        else:
            d = random.choice(DIRECTIONS)
        
        path.append(d)
        dx, dy = DIR_TO_DELTA[d]
        x += dx
        y += dy
    
    path_str = ' '.join(path)
    result_str = f"({x},{y})"
    
    return f"{path_str} -> {result_str}\n"

def generate_undo_task():
    """
    Alternative task: Explicit undo operations
    Format: "A B C undo D undo -> A D"
    """
    tokens = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
    
    # Generate a sequence with undos
    n_ops = random.randint(3, 8)
    sequence = []
    stack = []
    
    for _ in range(n_ops):
        if random.random() < 0.3 and len(stack) > 0:
            # Add undo
            sequence.append('undo')
            stack.pop()
        else:
            # Add token
            token = random.choice(tokens)
            sequence.append(token)
            stack.append(token)
    
    input_str = ' '.join(sequence)
    output_str = ' '.join(stack) if stack else 'empty'
    
    return f"{input_str} -> {output_str}\n"

def generate_rotation_composition():
    """
    Rotation composition task
    Format: "R90 R90 R180 R-90 -> R180"
    Rotations add: R90 + R90 = R180, R180 + R180 = R0
    """
    rotations = [0, 90, 180, 270]  # or equivalently -90
    
    n_ops = random.randint(2, 6)
    ops = [random.choice(rotations) for _ in range(n_ops)]
    
    # Compute total rotation mod 360
    total = sum(ops) % 360
    
    def rot_to_str(r):
        if r == 0:
            return 'R0'
        elif r == 90:
            return 'R90'
        elif r == 180:
            return 'R180'
        elif r == 270:
            return 'R270'
    
    input_str = ' '.join([rot_to_str(r) for r in ops])
    output_str = rot_to_str(total)
    
    return f"{input_str} -> {output_str}\n"

def encode(s):
    """Simple ASCII encoding."""
    return [ord(c) for c in s]

if __name__ == '__main__':
    os.makedirs('data/reversibility', exist_ok=True)
    
    random.seed(42)
    
    data = []
    
    # Mix of different reversibility tasks
    print("Generating reversibility dataset...")
    
    # 40% path cancellation (random)
    for _ in range(20000):
        data.append(generate_path(min_steps=3, max_steps=10))
    
    # 30% path cancellation (heavy cancellation)
    for _ in range(15000):
        data.append(generate_cancellation_heavy_path(min_steps=4, max_steps=10))
    
    # 20% explicit undo task
    for _ in range(10000):
        data.append(generate_undo_task())
    
    # 10% rotation composition
    for _ in range(5000):
        data.append(generate_rotation_composition())
    
    # Shuffle
    random.shuffle(data)
    
    # Split 90/10
    n = len(data)
    train_data = "".join(data[:int(n*0.9)])
    val_data = "".join(data[int(n*0.9):])
    
    # Encode and save
    train_ids = np.array(encode(train_data), dtype=np.uint16)
    val_ids = np.array(encode(val_data), dtype=np.uint16)
    
    train_ids.tofile('data/reversibility/train.bin')
    val_ids.tofile('data/reversibility/val.bin')
    
    print(f"Saved {len(train_ids):,} train tokens, {len(val_ids):,} val tokens")
    
    # Print examples
    print("\n=== Sample Data ===")
    print("Path cancellation:")
    for i in range(3):
        print(f"  {data[i].strip()}")
    print("\nHeavy cancellation:")
    for i in range(20000, 20003):
        print(f"  {data[i].strip()}")
    print("\nUndo task:")
    for i in range(35000, 35003):
        print(f"  {data[i].strip()}")
    print("\nRotation composition:")
    for i in range(45000, 45003):
        print(f"  {data[i].strip()}")
