"""
Viewpoint Transformation Task (Pure Rotation)

Tests TRUE geometric rotation: Transform spatial relationships between viewpoints.

Key properties:
- NO suppression/negation needed
- Information is PRESERVED and ROTATED
- Requires understanding SO(2) rotations (turning perspectives)
- Ground truth is deterministic geometric transformation

Example:
  Input:  "Alice faces NORTH. The tree is to her LEFT. From the tree's view, Alice is to the ="
  Output: "RIGHT"
  
  Explanation: If tree is to Alice's left, then from tree's perspective, Alice is to tree's right.
  This is a 180° viewpoint rotation.

Expected results:
- Cayley rotation: Should excel (true SO(2) structure)
- Pure DDL: May struggle (no negation/suppression)
- E∆-Hybrid: Should work (has rotation component)
"""

import os
import pickle
import random
import numpy as np

random.seed(42)
np.random.seed(42)

# ============================================================================
# SPATIAL DIRECTION SYSTEM
# ============================================================================

# Directions in clockwise order (for rotation)
DIRECTIONS = ['NORTH', 'EAST', 'SOUTH', 'WEST']  # 0, 90, 180, 270 degrees

# Relative positions
RELATIVES = ['FRONT', 'RIGHT', 'BACK', 'LEFT']  # 0, 90, 180, 270 degrees

def direction_to_idx(d):
    return DIRECTIONS.index(d)

def relative_to_idx(r):
    return RELATIVES.index(r)

def idx_to_direction(i):
    return DIRECTIONS[i % 4]

def idx_to_relative(i):
    return RELATIVES[i % 4]

def rotate_relative(relative, rotation_steps):
    """Rotate a relative position by rotation_steps * 90 degrees clockwise."""
    idx = relative_to_idx(relative)
    return idx_to_relative(idx + rotation_steps)

def opposite_relative(relative):
    """Get the opposite relative direction."""
    return rotate_relative(relative, 2)  # 180 degrees

# ============================================================================
# TASK TYPES
# ============================================================================

def generate_viewpoint_swap():
    """
    Task: Given A's view of B, determine B's view of A.
    
    This is always a 180° rotation (opposite).
    
    Example:
    "Alice faces NORTH. Bob is to her LEFT. From Bob's view, Alice is to the ="
    Answer: "RIGHT" (opposite of LEFT)
    """
    name_a = random.choice(['Alice', 'Bob', 'Carol', 'Dave', 'Eve', 'Frank'])
    name_b = random.choice(['the tree', 'the house', 'the car', 'the rock', 'the lamp', 'the bench'])
    while name_a == name_b:
        name_b = random.choice(['the tree', 'the house', 'the car', 'the rock', 'the lamp', 'the bench'])
    
    facing = random.choice(DIRECTIONS)
    relative = random.choice(RELATIVES)
    
    # From B's view, A is in the opposite direction
    answer = opposite_relative(relative)
    
    input_text = f"{name_a} faces {facing}. {name_b.capitalize()} is to {name_a}'s {relative}. From {name_b}'s view, {name_a} is to the ="
    
    return input_text, answer

def generate_facing_change():
    """
    Task: Person turns, where is object now relative to them?
    
    Example:
    "Alice faces NORTH. The tree is to her LEFT. She turns to face EAST. The tree is now to her ="
    Answer: "FRONT" (tree was WEST when facing NORTH, now facing EAST so tree is in FRONT)
    """
    name = random.choice(['Alice', 'Bob', 'Carol', 'Dave', 'Eve', 'Frank'])
    obj = random.choice(['tree', 'house', 'car', 'rock', 'lamp', 'bench', 'statue', 'fountain'])
    
    facing_start = random.choice(DIRECTIONS)
    relative_start = random.choice(RELATIVES)
    facing_end = random.choice(DIRECTIONS)
    
    # Calculate absolute direction of object
    # If facing NORTH and object is to LEFT, object is at WEST
    facing_idx = direction_to_idx(facing_start)
    relative_idx = relative_to_idx(relative_start)
    object_absolute_idx = (facing_idx + relative_idx) % 4  # Object's absolute direction
    
    # After turning, relative position changes
    new_facing_idx = direction_to_idx(facing_end)
    new_relative_idx = (object_absolute_idx - new_facing_idx) % 4
    answer = idx_to_relative(new_relative_idx)
    
    input_text = f"{name} faces {facing_start}. The {obj} is to {name}'s {relative_start}. {name} turns to face {facing_end}. The {obj} is now to {name}'s ="
    
    return input_text, answer

def generate_chain_perspective():
    """
    Task: Chain of perspective transformations.
    
    Example:
    "Alice faces NORTH. Bob is to her RIGHT. Carol is to Bob's LEFT. From Alice's view, Carol is to the ="
    
    Requires tracking multiple rotations.
    """
    names = random.sample(['Alice', 'Bob', 'Carol', 'Dave', 'Eve'], 3)
    
    facing = random.choice(DIRECTIONS)
    rel_ab = random.choice(RELATIVES)  # B relative to A
    rel_bc = random.choice(RELATIVES)  # C relative to B
    
    # Calculate A's facing direction
    a_facing_idx = direction_to_idx(facing)
    
    # B's position relative to A in absolute terms
    b_absolute_idx = (a_facing_idx + relative_to_idx(rel_ab)) % 4
    
    # B faces toward A (opposite direction from A to B)
    b_facing_idx = (b_absolute_idx + 2) % 4  # B faces opposite to the direction from A to B
    
    # Actually, let's simplify: B faces the same direction as A
    b_facing_idx = a_facing_idx
    
    # C's position relative to B in absolute terms
    c_absolute_idx = (b_facing_idx + relative_to_idx(rel_bc)) % 4
    
    # C's position relative to A
    c_relative_to_a_idx = (c_absolute_idx - a_facing_idx) % 4
    answer = idx_to_relative(c_relative_to_a_idx)
    
    input_text = f"{names[0]} faces {facing}. {names[1]} is to {names[0]}'s {rel_ab}. {names[1]} also faces {facing}. {names[2]} is to {names[1]}'s {rel_bc}. From {names[0]}'s view, {names[2]} is to the ="
    
    return input_text, answer

def generate_mirror_perspective():
    """
    Task: Mirror transformation (left-right swap).
    
    Example:
    "Alice looks in a mirror. Her RIGHT hand appears on the ="
    Answer: "LEFT"
    
    This tests if the model understands reflection vs rotation.
    (DDL should excel here, rotation should struggle)
    """
    name = random.choice(['Alice', 'Bob', 'Carol', 'Dave'])
    
    # Mirror only swaps LEFT-RIGHT, not FRONT-BACK
    original = random.choice(['RIGHT', 'LEFT'])
    answer = 'LEFT' if original == 'RIGHT' else 'RIGHT'
    
    input_text = f"{name} looks in a mirror. {name}'s {original} hand appears on the ="
    
    return input_text, answer

def generate_rotation_count():
    """
    Task: Count rotations.
    
    Example:
    "Alice faces NORTH. She turns RIGHT once. She turns RIGHT again. She now faces ="
    Answer: "SOUTH" (two 90° right turns = 180°)
    """
    name = random.choice(['Alice', 'Bob', 'Carol', 'Dave'])
    start_facing = random.choice(DIRECTIONS)
    
    # Generate 1-3 turns
    n_turns = random.randint(1, 3)
    directions = [random.choice(['RIGHT', 'LEFT']) for _ in range(n_turns)]
    
    # Calculate final facing
    facing_idx = direction_to_idx(start_facing)
    for d in directions:
        if d == 'RIGHT':
            facing_idx = (facing_idx + 1) % 4
        else:  # LEFT
            facing_idx = (facing_idx - 1) % 4
    
    answer = idx_to_direction(facing_idx)
    
    # Build description
    turn_descriptions = []
    for i, d in enumerate(directions):
        if i == 0:
            turn_descriptions.append(f"{name} turns {d}")
        else:
            turn_descriptions.append(f"then turns {d}")
    
    turns_text = ". ".join(turn_descriptions)
    input_text = f"{name} faces {start_facing}. {turns_text}. {name} now faces ="
    
    return input_text, answer

# ============================================================================
# DATASET GENERATION
# ============================================================================

TASK_GENERATORS = [
    (generate_viewpoint_swap, 0.25),
    (generate_facing_change, 0.30),
    (generate_chain_perspective, 0.20),
    (generate_mirror_perspective, 0.10),  # Included to test DDL vs rotation
    (generate_rotation_count, 0.15),
]

def generate_sample():
    """Generate a single sample based on weighted task distribution."""
    r = random.random()
    cumulative = 0
    for generator, weight in TASK_GENERATORS:
        cumulative += weight
        if r < cumulative:
            return generator()
    return TASK_GENERATORS[-1][0]()

def generate_dataset(n_samples):
    samples = []
    for _ in range(n_samples):
        inp, out = generate_sample()
        samples.append((inp, out))
    return samples

def encode_sample(input_text, output_text):
    full_text = input_text + output_text
    return [ord(c) for c in full_text]

def main():
    print("=" * 60)
    print("Generating Viewpoint Transformation Dataset")
    print("=" * 60)
    
    n_train = 50000
    n_val = 5000
    
    print(f"\nGenerating {n_train} training samples...")
    train_samples = generate_dataset(n_train)
    
    print(f"Generating {n_val} validation samples...")
    val_samples = generate_dataset(n_val)
    
    # Show examples
    print("\n" + "=" * 60)
    print("EXAMPLE SAMPLES:")
    print("=" * 60)
    
    # Show one of each type
    for gen, _ in TASK_GENERATORS:
        inp, out = gen()
        print(f"\nTask: {gen.__name__}")
        print(f"Input:  {inp}")
        print(f"Output: {out}")
    
    # Encode
    print("\n" + "=" * 60)
    print("Encoding samples...")
    print("=" * 60)
    
    train_tokens = []
    for inp, out in train_samples:
        train_tokens.extend(encode_sample(inp, out))
    
    val_tokens = []
    for inp, out in val_samples:
        val_tokens.extend(encode_sample(inp, out))
    
    train_data = np.array(train_tokens, dtype=np.uint16)
    val_data = np.array(val_tokens, dtype=np.uint16)
    
    print(f"\nTrain tokens: {len(train_data):,}")
    print(f"Val tokens:   {len(val_data):,}")
    
    # Save
    train_data.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
    val_data.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))
    
    meta = {
        'vocab_size': 256,
        'task': 'viewpoint_transformation',
        'description': 'Pure rotation task - transform spatial relationships between viewpoints',
        'task_types': [gen.__name__ for gen, _ in TASK_GENERATORS],
    }
    with open(os.path.join(os.path.dirname(__file__), 'meta.pkl'), 'wb') as f:
        pickle.dump(meta, f)
    
    print("\n" + "=" * 60)
    print("Dataset saved!")
    print("=" * 60)
    
    # Output distribution
    print("\n" + "=" * 60)
    print("OUTPUT DISTRIBUTION:")
    print("=" * 60)
    from collections import Counter
    outputs = Counter([out for _, out in train_samples])
    for out, count in sorted(outputs.items(), key=lambda x: -x[1]):
        print(f"  {out}: {count} ({100*count/n_train:.1f}%)")

if __name__ == "__main__":
    main()
