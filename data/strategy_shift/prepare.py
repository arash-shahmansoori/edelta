"""
Strategy Shift Dataset (Math Reasoning)

Purpose: Replicate Illusion of Insight paper's math experiments.
Test if the model can detect when a naive approach fails and 
pivot to a correct strategy.

Inspired by examples from arXiv:2601.00514v1:
- "However, this approach does not yield a finite minimum. Instead..."
- "This does not simplify easily. Let's try another approach..."

We create math-like problems where:
1. A naive approach leads to a dead end
2. A strategy shift is required
3. The correct answer follows the shift
"""

import os
import random
import numpy as np

# Configuration
NUM_TRAIN = 30000
NUM_VAL = 3000

# Strategy shift markers (from Illusion of Insight paper)
SHIFT_MARKERS = [
    "However, this doesn't work.",
    "This approach fails.",
    "That leads nowhere.",
    "This gives an invalid result.",
    "Hmm, that's not right.",
]

PIVOT_PHRASES = [
    "Instead, let's try",
    "A better approach is",
    "Let me reconsider:",
    "Actually, we should",
    "The correct method is",
]

def generate_arithmetic_with_shift() -> str:
    """Generate arithmetic that requires order-of-operations awareness."""
    a = random.randint(2, 10)
    b = random.randint(2, 10)
    c = random.randint(2, 10)
    
    # Problem: a + b * c
    # Naive: (a + b) * c (wrong)
    # Correct: a + (b * c)
    
    naive_result = (a + b) * c
    correct_result = a + b * c
    
    shift = random.choice(SHIFT_MARKERS)
    pivot = random.choice(PIVOT_PHRASES)
    
    input_str = f"""Problem: Calculate {a} + {b} * {c}
Naive approach: Add first, then multiply... {a}+{b}={a+b}, then {a+b}*{c}={naive_result}
{shift} {pivot} follow order of operations.
Multiply first: {b}*{c}={b*c}, then add {a}: {a}+{b*c}="""
    
    return input_str + str(correct_result) + "\n"

def generate_modular_with_shift() -> str:
    """Generate modular arithmetic with potential overflow thinking."""
    a = random.randint(10, 50)
    b = random.randint(10, 50)
    m = random.randint(5, 15)
    
    # Problem: (a + b) mod m
    # Naive: might try to think of it wrong way
    # Correct: direct computation
    
    naive_wrong = (a + b) % (m + 1)  # Intentionally wrong
    correct = (a + b) % m
    
    shift = random.choice(SHIFT_MARKERS)
    pivot = random.choice(PIVOT_PHRASES)
    
    input_str = f"""Problem: ({a} + {b}) mod {m}
Attempt: {a}+{b}={a+b}, mod... wait, is it mod {m} or {m+1}?
Trying mod {m+1}: {naive_wrong}
{shift} {pivot} use mod {m} directly.
{a+b} mod {m} = """
    
    return input_str + str(correct) + "\n"

def generate_fraction_with_shift() -> str:
    """Generate fraction simplification with shift."""
    # Create a fraction that simplifies
    gcd = random.randint(2, 10)
    a = gcd * random.randint(1, 10)
    b = gcd * random.randint(1, 10)
    
    shift = random.choice(SHIFT_MARKERS)
    pivot = random.choice(PIVOT_PHRASES)
    
    simplified_a = a // gcd
    simplified_b = b // gcd
    
    input_str = f"""Problem: Simplify {a}/{b}
Naive: Just divide... {a}/{b} = {a/b:.4f}
{shift} {pivot} find GCD and simplify.
GCD({a},{b}) = {gcd}
{a}/{gcd} / {b}/{gcd} = """
    
    return input_str + f"{simplified_a}/{simplified_b}\n"

def generate_equation_with_shift() -> str:
    """Generate simple equation solving with a twist."""
    # 2x + 3 = 11 -> x = 4
    a = random.randint(2, 5)
    x_true = random.randint(1, 10)
    b = random.randint(1, 10)
    c = a * x_true + b
    
    shift = random.choice(SHIFT_MARKERS)
    pivot = random.choice(PIVOT_PHRASES)
    
    # Naive: divide first (wrong)
    naive_wrong = c // a
    
    input_str = f"""Problem: Solve {a}x + {b} = {c}
Naive attempt: Divide by {a}... x = {c}/{a} = {naive_wrong}?
Check: {a}*{naive_wrong}+{b} = {a*naive_wrong+b} ≠ {c}
{shift} {pivot} subtract {b} first, then divide.
{a}x = {c}-{b} = {c-b}
x = {c-b}/{a} = """
    
    return input_str + str(x_true) + "\n"

def generate_sample() -> str:
    """Generate a random strategy shift sample."""
    generators = [
        generate_arithmetic_with_shift,
        generate_modular_with_shift,
        generate_fraction_with_shift,
        generate_equation_with_shift,
    ]
    
    return random.choice(generators)()

def main():
    random.seed(42)
    np.random.seed(42)
    
    print("Generating Strategy Shift dataset (math reasoning)...")
    print(f"  Train samples: {NUM_TRAIN}")
    print(f"  Val samples: {NUM_VAL}")
    
    # Generate data
    train_data = [generate_sample() for _ in range(NUM_TRAIN)]
    
    random.seed(123)
    val_data = [generate_sample() for _ in range(NUM_VAL)]
    
    # Combine
    train_text = "".join(train_data)
    val_text = "".join(val_data)
    
    print(f"\nSample training example:")
    print(train_data[0])
    
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
        f.write(f"Strategy Shift Dataset (Math Reasoning)\n")
        f.write(f"Train samples: {NUM_TRAIN}\n")
        f.write(f"Val samples: {NUM_VAL}\n")
        f.write(f"Purpose: Test genuine reasoning pivots\n")
    
    print("\nDone! Files saved: train.bin, val.bin, meta.txt")

if __name__ == "__main__":
    main()
