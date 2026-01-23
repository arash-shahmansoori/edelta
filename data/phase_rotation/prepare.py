"""
Continuous Phase Rotation Task

Tests SMOOTH rotation capabilities - where DDL's discrete reflections should struggle.

Task: Given a phase value and rotation amount, compute the final phase.
      All values are continuous (not discrete like 90° multiples).

Examples:
  Input:  "phase:0.25 rotate:0.30 final:"
  Output: "0.55"
  
  Input:  "phase:0.80 rotate:0.35 final:"
  Output: "0.15"  (wraps around: 0.80 + 0.35 = 1.15 → 0.15)

Why DDL should struggle:
- Discrete reflections can't smoothly interpolate
- Composition of reflections = discrete rotation angles
- Continuous output requires smooth manifold traversal

Why Cayley should excel:
- β continuously controls rotation angle
- Smooth interpolation on SO(n) manifold
- No discrete approximation errors

Additional task types for robustness:
1. Single rotation: phase + rotation → final
2. Double rotation: phase + rot1 + rot2 → final  
3. Inverse rotation: phase - rotation → final (CCW)
4. Composition: Apply same rotation N times
"""

import os
import pickle
import random
import numpy as np

random.seed(42)
np.random.seed(42)

def format_phase(value):
    """Format phase value to 2 decimal places."""
    # Ensure in [0, 1) range
    value = value % 1.0
    return f"{value:.2f}"

def generate_single_rotation():
    """
    Simple rotation: phase + rotation = final (mod 1)
    
    Example: "phase:0.25 rotate:0.30 final:0.55"
    """
    phase = random.uniform(0, 1)
    rotation = random.uniform(0, 0.5)  # Max half rotation to avoid ambiguity
    
    final = (phase + rotation) % 1.0
    
    input_text = f"phase:{format_phase(phase)} rotate:{format_phase(rotation)} final:"
    output_text = format_phase(final)
    
    return input_text, output_text

def generate_double_rotation():
    """
    Two rotations: phase + rot1 + rot2 = final (mod 1)
    
    Example: "phase:0.10 rot1:0.25 rot2:0.30 final:0.65"
    """
    phase = random.uniform(0, 1)
    rot1 = random.uniform(0, 0.3)
    rot2 = random.uniform(0, 0.3)
    
    final = (phase + rot1 + rot2) % 1.0
    
    input_text = f"phase:{format_phase(phase)} rot1:{format_phase(rot1)} rot2:{format_phase(rot2)} final:"
    output_text = format_phase(final)
    
    return input_text, output_text

def generate_inverse_rotation():
    """
    Counter-clockwise (subtract): phase - rotation = final (mod 1)
    
    Example: "phase:0.30 rotate_ccw:0.25 final:0.05"
    """
    phase = random.uniform(0, 1)
    rotation = random.uniform(0, 0.5)
    
    final = (phase - rotation) % 1.0
    
    input_text = f"phase:{format_phase(phase)} rotate_ccw:{format_phase(rotation)} final:"
    output_text = format_phase(final)
    
    return input_text, output_text

def generate_repeated_rotation():
    """
    Apply same rotation N times: phase + N*rotation = final (mod 1)
    
    Example: "phase:0.10 rotate:0.15 times:3 final:0.55" (0.10 + 3*0.15 = 0.55)
    """
    phase = random.uniform(0, 1)
    rotation = random.uniform(0.05, 0.15)  # Small rotation to avoid too much wrapping
    times = random.randint(2, 5)
    
    final = (phase + times * rotation) % 1.0
    
    input_text = f"phase:{format_phase(phase)} rotate:{format_phase(rotation)} times:{times} final:"
    output_text = format_phase(final)
    
    return input_text, output_text

def generate_interpolation():
    """
    Interpolate between two phases: lerp(phase1, phase2, t) along circle
    
    Example: "from:0.20 to:0.80 at:0.50 phase:0.50" (halfway between)
    
    This requires understanding circular interpolation!
    """
    phase1 = random.uniform(0, 1)
    phase2 = random.uniform(0, 1)
    t = random.choice([0.25, 0.50, 0.75])  # Discrete t for easier learning
    
    # Circular interpolation (shortest path)
    diff = phase2 - phase1
    if abs(diff) > 0.5:
        # Wrap around
        if diff > 0:
            diff = diff - 1
        else:
            diff = diff + 1
    
    final = (phase1 + t * diff) % 1.0
    
    input_text = f"from:{format_phase(phase1)} to:{format_phase(phase2)} at:{format_phase(t)} phase:"
    output_text = format_phase(final)
    
    return input_text, output_text

def generate_angle_to_phase():
    """
    Convert angle in degrees to phase (0-360 → 0-1)
    
    Example: "angle:90 phase:0.25"
    """
    angle = random.randint(0, 359)
    phase = angle / 360.0
    
    input_text = f"angle:{angle} phase:"
    output_text = format_phase(phase)
    
    return input_text, output_text

def generate_quadrant():
    """
    Determine which quadrant a phase is in (tests understanding of phase space)
    
    Example: "phase:0.30 quadrant:2" (0.25-0.50 = Q2)
    """
    phase = random.uniform(0, 1)
    
    if phase < 0.25:
        quadrant = 1
    elif phase < 0.50:
        quadrant = 2
    elif phase < 0.75:
        quadrant = 3
    else:
        quadrant = 4
    
    input_text = f"phase:{format_phase(phase)} quadrant:"
    output_text = str(quadrant)
    
    return input_text, output_text

# Task distribution - emphasize continuous rotation
TASK_GENERATORS = [
    (generate_single_rotation, 0.30),      # Core task
    (generate_double_rotation, 0.20),      # Composition
    (generate_inverse_rotation, 0.15),     # CCW rotation
    (generate_repeated_rotation, 0.15),    # Iterative application
    (generate_interpolation, 0.10),        # Smooth interpolation
    (generate_angle_to_phase, 0.05),       # Unit conversion
    (generate_quadrant, 0.05),             # Discretization (baseline)
]

def generate_sample():
    r = random.random()
    cumulative = 0
    for generator, weight in TASK_GENERATORS:
        cumulative += weight
        if r < cumulative:
            return generator()
    return TASK_GENERATORS[0][0]()

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
    print("Generating Continuous Phase Rotation Dataset")
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
        'task': 'continuous_phase_rotation',
        'description': 'Smooth rotation on circular phase space - tests continuous interpolation',
        'task_types': [gen.__name__ for gen, _ in TASK_GENERATORS],
    }
    with open(os.path.join(os.path.dirname(__file__), 'meta.pkl'), 'wb') as f:
        pickle.dump(meta, f)
    
    print("\n" + "=" * 60)
    print("Dataset saved!")
    print("=" * 60)
    
    # Task distribution check
    print("\n" + "=" * 60)
    print("TASK DISTRIBUTION (from training set):")
    print("=" * 60)
    from collections import Counter
    
    # Categorize by prefix
    task_counts = Counter()
    for inp, out in train_samples:
        if inp.startswith("phase:") and "rot1:" in inp:
            task_counts["double_rotation"] += 1
        elif inp.startswith("phase:") and "rotate_ccw:" in inp:
            task_counts["inverse_rotation"] += 1
        elif inp.startswith("phase:") and "times:" in inp:
            task_counts["repeated_rotation"] += 1
        elif inp.startswith("phase:") and "quadrant:" in inp:
            task_counts["quadrant"] += 1
        elif inp.startswith("phase:") and "rotate:" in inp:
            task_counts["single_rotation"] += 1
        elif inp.startswith("from:"):
            task_counts["interpolation"] += 1
        elif inp.startswith("angle:"):
            task_counts["angle_to_phase"] += 1
    
    for task, count in sorted(task_counts.items(), key=lambda x: -x[1]):
        print(f"  {task}: {count} ({100*count/n_train:.1f}%)")

if __name__ == "__main__":
    main()
