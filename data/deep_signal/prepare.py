"""
Deep Signal Preservation Dataset

Purpose: Test mHC's claim that signal energy is preserved across depth.

Task: Copy a signal through noise.
Input:  "SIGNAL:42 noise noise noise ... noise RECALL:"
Output: "42"

The model must preserve the signal through many layers of noise.
If energy is not conserved, the signal will degrade.

This tests the core mHC claim: doubly stochastic (or orthogonal) mixing
preserves signal energy, while unconstrained mixing causes degradation.
"""

import os
import random
import numpy as np

# Configuration
NUM_TRAIN = 50000
NUM_VAL = 5000
SIGNAL_RANGE = (0, 99)  # 2-digit signals
NOISE_LENGTH_RANGE = (20, 100)  # Variable distraction length
NOISE_TOKEN_RANGE = (100, 999)  # 3-digit noise tokens

def generate_sample():
    """Generate a single deep signal sample."""
    # The signal to remember
    signal = random.randint(*SIGNAL_RANGE)
    
    # Noise tokens to filter through
    noise_length = random.randint(*NOISE_LENGTH_RANGE)
    noise_tokens = [random.randint(*NOISE_TOKEN_RANGE) for _ in range(noise_length)]
    
    # Format input
    noise_str = " ".join(map(str, noise_tokens))
    input_str = f"SIGNAL:{signal} {noise_str} RECALL:"
    
    # Output is just the signal
    output_str = str(signal)
    
    # Full sequence for training
    full_seq = input_str + output_str + "\n"
    
    return full_seq

def main():
    # Set seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    print("Generating Deep Signal Preservation dataset...")
    print(f"  Train samples: {NUM_TRAIN}")
    print(f"  Val samples: {NUM_VAL}")
    print(f"  Signal range: {SIGNAL_RANGE}")
    print(f"  Noise length: {NOISE_LENGTH_RANGE}")
    
    # Generate training data
    train_data = []
    for _ in range(NUM_TRAIN):
        train_data.append(generate_sample())
    
    # Generate validation data (different seed)
    random.seed(123)
    val_data = []
    for _ in range(NUM_VAL):
        val_data.append(generate_sample())
    
    # Combine to text
    train_text = "".join(train_data)
    val_text = "".join(val_data)
    
    print(f"\nSample training example:")
    print(f"  {train_data[0][:100]}...")
    
    # Encode to bytes
    train_ids = np.array([ord(c) for c in train_text], dtype=np.uint16)
    val_ids = np.array([ord(c) for c in val_text], dtype=np.uint16)
    
    print(f"\nDataset sizes:")
    print(f"  Train: {len(train_ids):,} tokens")
    print(f"  Val: {len(val_ids):,} tokens")
    
    # Save
    train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
    val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))
    
    # Save metadata
    with open(os.path.join(os.path.dirname(__file__), 'meta.txt'), 'w') as f:
        f.write(f"Deep Signal Preservation Dataset\n")
        f.write(f"Train samples: {NUM_TRAIN}\n")
        f.write(f"Val samples: {NUM_VAL}\n")
        f.write(f"Signal range: {SIGNAL_RANGE}\n")
        f.write(f"Noise length range: {NOISE_LENGTH_RANGE}\n")
        f.write(f"Purpose: Test signal energy conservation across depth\n")
    
    print("\nDone! Files saved: train.bin, val.bin, meta.txt")

if __name__ == "__main__":
    main()
