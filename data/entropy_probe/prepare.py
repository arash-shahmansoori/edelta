"""
Entropy Probe Dataset

Purpose: Directly test thermodynamic gating.
β should correlate with prediction entropy:
- High entropy (uncertain) → β should be HIGH
- Low entropy (confident) → β should be LOW

This validates E∆-MHC-Geo's core mechanism: the purity proxy (Φ)
correctly measures uncertainty and gates geometric restructuring.
"""

import os
import random
import numpy as np

# Configuration
NUM_TRAIN = 40000
NUM_VAL = 4000

# High entropy templates (multiple valid answers)
HIGH_ENTROPY_TEMPLATES = [
    # Multiple valid continuations
    ("The animal could be a cat, dog, or bird. It is a", ["cat", "dog", "bird"]),
    ("Pick any number from 1, 2, 3, 4, 5. I choose", ["1", "2", "3", "4", "5"]),
    ("The color might be red, blue, or green. It's", ["red", "blue", "green"]),
    ("Choose heads or tails:", ["heads", "tails"]),
    ("Say yes or no:", ["yes", "no"]),
    ("Left or right?", ["left", "right"]),
    ("True or false (your choice):", ["true", "false"]),
    ("Pick A, B, C, or D:", ["A", "B", "C", "D"]),
    ("Any day of the week:", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]),
    ("Name any planet:", ["Mercury", "Venus", "Earth", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune"]),
]

# Low entropy templates (only one correct answer)
LOW_ENTROPY_TEMPLATES = [
    # Deterministic facts
    ("2 + 2 =", "4"),
    ("The capital of France is", "Paris"),
    ("Water freezes at 0 degrees", "Celsius"),
    ("The sun rises in the", "East"),
    ("There are 24 hours in a", "day"),
    ("The opposite of up is", "down"),
    ("H2O is the formula for", "water"),
    ("A triangle has 3", "sides"),
    ("The first letter of the alphabet is", "A"),
    ("100 divided by 10 equals", "10"),
    ("Red and blue make", "purple"),
    ("A year has 12", "months"),
    ("The largest planet is", "Jupiter"),
    ("Ice is frozen", "water"),
    ("A decade is 10", "years"),
]

def generate_high_entropy_sample() -> tuple:
    """Generate a high-entropy (uncertain) sample."""
    template, options = random.choice(HIGH_ENTROPY_TEMPLATES)
    answer = random.choice(options)
    return template, answer, "HIGH"

def generate_low_entropy_sample() -> tuple:
    """Generate a low-entropy (certain) sample."""
    template, answer = random.choice(LOW_ENTROPY_TEMPLATES)
    return template, answer, "LOW"

def generate_sample() -> str:
    """Generate a random sample with entropy label."""
    if random.random() < 0.5:
        template, answer, entropy = generate_high_entropy_sample()
    else:
        template, answer, entropy = generate_low_entropy_sample()
    
    # Include entropy marker in training data (for analysis)
    # Format: [ENTROPY:HIGH] or [ENTROPY:LOW] at the start
    full_seq = f"[{entropy}] {template} {answer}\n"
    
    return full_seq

def generate_sample_no_label() -> str:
    """Generate sample without entropy label (for testing)."""
    if random.random() < 0.5:
        template, answer, _ = generate_high_entropy_sample()
    else:
        template, answer, _ = generate_low_entropy_sample()
    
    return f"{template} {answer}\n"

def main():
    random.seed(42)
    np.random.seed(42)
    
    print("Generating Entropy Probe dataset...")
    print(f"  Train samples: {NUM_TRAIN}")
    print(f"  Val samples: {NUM_VAL}")
    print(f"  High entropy templates: {len(HIGH_ENTROPY_TEMPLATES)}")
    print(f"  Low entropy templates: {len(LOW_ENTROPY_TEMPLATES)}")
    
    # Generate training data (with labels for analysis)
    train_data = [generate_sample() for _ in range(NUM_TRAIN)]
    
    # Validation data
    random.seed(123)
    val_data = [generate_sample() for _ in range(NUM_VAL)]
    
    # Combine
    train_text = "".join(train_data)
    val_text = "".join(val_data)
    
    print(f"\nSample training examples:")
    high_examples = [d for d in train_data if d.startswith("[HIGH]")][:2]
    low_examples = [d for d in train_data if d.startswith("[LOW]")][:2]
    
    print("  High entropy:")
    for e in high_examples:
        print(f"    {e.strip()}")
    print("  Low entropy:")
    for e in low_examples:
        print(f"    {e.strip()}")
    
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
        f.write(f"Entropy Probe Dataset\n")
        f.write(f"Train samples: {NUM_TRAIN}\n")
        f.write(f"Val samples: {NUM_VAL}\n")
        f.write(f"Purpose: Test β-entropy correlation\n")
        f.write(f"Labels: [HIGH] for uncertain, [LOW] for certain\n")
    
    print("\nDone! Files saved: train.bin, val.bin, meta.txt")

if __name__ == "__main__":
    main()
