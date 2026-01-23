"""
Correction Task Dataset (Aha! Moment Detection)

Purpose: Test "Aha!" moment detection — does β spike when the model 
should "change its mind"?

Inspired by the Illusion of Insight paper (arXiv:2601.00514v1).

Task: Follow instructions with mid-sequence corrections.
Input:  "Task: Add 5 to X. X=10. Wait, I meant subtract 5. Answer:"
Output: "5"

The model must:
1. Initially compute 10+5=15
2. Recognize "Wait, I meant" as correction signal
3. Abandon first strategy, compute 10-5=5

We expect β to SPIKE at "Wait" token in E∆-MHC-Geo!
"""

import os
import random
import numpy as np

# Configuration
NUM_TRAIN = 50000
NUM_VAL = 5000

# Correction phrases (from Illusion of Insight paper patterns)
CORRECTION_PHRASES = [
    "Wait, I meant",
    "Actually, no,",
    "Scratch that,",
    "On second thought,",
    "However, instead",
    "No wait,",
    "Sorry, I meant",
    "Correction:",
    "Let me rephrase:",
    "Ignore that,",
]

# Operations
OPERATIONS = {
    "add": lambda x, y: x + y,
    "subtract": lambda x, y: x - y,
    "multiply": lambda x, y: x * y,
}

OP_WORDS = {
    "add": ["add", "plus", "increase by"],
    "subtract": ["subtract", "minus", "decrease by"],
    "multiply": ["multiply by", "times"],
}

def get_op_word(op: str) -> str:
    """Get a random word for an operation."""
    return random.choice(OP_WORDS[op])

def generate_simple_correction() -> str:
    """Generate a simple arithmetic correction sample."""
    x = random.randint(1, 50)
    y = random.randint(1, 20)
    
    # Original operation
    op1 = random.choice(list(OPERATIONS.keys()))
    # Corrected operation (must be different)
    op2 = random.choice([o for o in OPERATIONS.keys() if o != op1])
    
    correction = random.choice(CORRECTION_PHRASES)
    
    # Format input
    input_str = f"Task: {get_op_word(op1)} {y} to X. X={x}. {correction} {get_op_word(op2)} {y}. Answer:"
    
    # Compute corrected result
    result = OPERATIONS[op2](x, y)
    
    return input_str + str(result) + "\n"

def generate_direction_correction() -> str:
    """Generate a direction correction sample (like the paper's "Go North" example)."""
    directions = ["North", "South", "East", "West", "Up", "Down", "Left", "Right"]
    
    dir1 = random.choice(directions)
    dir2 = random.choice([d for d in directions if d != dir1])
    
    correction = random.choice(CORRECTION_PHRASES)
    
    input_str = f"Instruction: Go {dir1}. {correction} go {dir2}. Final direction:"
    
    return input_str + dir2 + "\n"

def generate_value_correction() -> str:
    """Generate a value assignment correction."""
    vars = ["X", "Y", "Z", "A", "B"]
    var = random.choice(vars)
    
    val1 = random.randint(1, 100)
    val2 = random.randint(1, 100)
    while val2 == val1:
        val2 = random.randint(1, 100)
    
    correction = random.choice(CORRECTION_PHRASES)
    
    input_str = f"Set {var}={val1}. {correction} set {var}={val2}. Value of {var}:"
    
    return input_str + str(val2) + "\n"

def generate_negation_correction() -> str:
    """Generate a true/false negation correction."""
    statements = [
        ("The sky is blue", True),
        ("Water is wet", True),
        ("Fire is cold", False),
        ("Ice is hot", False),
        ("Grass is green", True),
        ("Snow is black", False),
    ]
    
    stmt, truth = random.choice(statements)
    correction = random.choice(CORRECTION_PHRASES)
    
    # First claim the opposite, then correct
    first_claim = "True" if not truth else "False"
    correct_claim = "True" if truth else "False"
    
    input_str = f"Statement: '{stmt}' is {first_claim}. {correction} it's {correct_claim}. Final answer:"
    
    return input_str + correct_claim + "\n"

def generate_sample() -> str:
    """Generate a random correction sample."""
    generators = [
        generate_simple_correction,
        generate_direction_correction,
        generate_value_correction,
        generate_negation_correction,
    ]
    
    weights = [0.4, 0.2, 0.2, 0.2]  # Weight towards arithmetic
    
    generator = random.choices(generators, weights=weights)[0]
    return generator()

def main():
    random.seed(42)
    np.random.seed(42)
    
    print("Generating Correction Task dataset (Aha! moments)...")
    print(f"  Train samples: {NUM_TRAIN}")
    print(f"  Val samples: {NUM_VAL}")
    print(f"  Correction phrases: {len(CORRECTION_PHRASES)}")
    
    # Generate training data
    train_data = [generate_sample() for _ in range(NUM_TRAIN)]
    
    # Validation data
    random.seed(123)
    val_data = [generate_sample() for _ in range(NUM_VAL)]
    
    # Combine
    train_text = "".join(train_data)
    val_text = "".join(val_data)
    
    print(f"\nSample training examples:")
    for i, d in enumerate(train_data[:5]):
        print(f"  {i+1}. {d.strip()}")
    
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
        f.write(f"Correction Task Dataset (Aha! Moments)\n")
        f.write(f"Train samples: {NUM_TRAIN}\n")
        f.write(f"Val samples: {NUM_VAL}\n")
        f.write(f"Purpose: Test if β spikes at correction tokens\n")
        f.write(f"Correction phrases: {CORRECTION_PHRASES}\n")
    
    print("\nDone! Files saved: train.bin, val.bin, meta.txt")

if __name__ == "__main__":
    main()
