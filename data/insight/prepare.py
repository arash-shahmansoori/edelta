"""
Insight Disambiguation Task

Tests Type A "Aha!" moments: Uncertainty → Clarity (via rotation, NOT correction)

Task: Given an ambiguous word in initial context, then disambiguating context,
      predict which meaning applies.

Example:
  Input:  "The bank was notable. The fisherman cast his line. MEANING="
  Output: "river"
  
  Input:  "The bank was notable. The teller opened the vault. MEANING="  
  Output: "money"

Key properties:
- Creates genuine uncertainty (high Φ) at ambiguous word
- Resolution requires restructuring (rotation), not negation (reflection)
- No "correction" - just clarity emerging from ambiguity

Expected results:
- E∆-Hybrid: Should excel (entropy-driven rotation activates)
- Rotation-only: Should also work well
- Pure DDL: Should NOT help (no negation needed)
- Baseline: Moderate performance
"""

import os
import pickle
import random
import numpy as np

# Seed for reproducibility
random.seed(42)
np.random.seed(42)

# ============================================================================
# AMBIGUOUS WORDS AND THEIR MEANINGS WITH CONTEXTS
# ============================================================================

AMBIGUOUS_DATA = {
    'bank': {
        'meanings': ['river', 'money'],
        'contexts': {
            'river': [
                "The fisherman cast his line into the water.",
                "Water flowed downstream past the rocks.",
                "Ducks swam along the edge near the reeds.",
                "The canoe was tied to a tree by the water.",
                "Mud covered the slope down to the stream.",
                "Fish jumped near the grassy edge.",
            ],
            'money': [
                "The teller opened the vault carefully.",
                "Interest rates were discussed at length.",
                "The loan officer reviewed the application.",
                "ATM machines lined the entrance hall.",
                "Customers waited in line for service.",
                "The safe deposit boxes were downstairs.",
            ],
        },
    },
    'bat': {
        'meanings': ['animal', 'sports'],
        'contexts': {
            'animal': [
                "It flew through the cave at night.",
                "Echolocation helped it find insects.",
                "The creature hung upside down to rest.",
                "Wings flapped silently in the darkness.",
                "It roosted in the attic during daytime.",
                "Small eyes gleamed in the moonlight.",
            ],
            'sports': [
                "The player swung at the fastball.",
                "Wood cracked as it hit the ball.",
                "The slugger stepped up to home plate.",
                "Coach handed it to the next batter.",
                "The grip tape was worn from practice.",
                "It was made of solid maple wood.",
            ],
        },
    },
    'spring': {
        'meanings': ['season', 'water', 'coil'],
        'contexts': {
            'season': [
                "Flowers bloomed across the meadow.",
                "Birds returned from their migration.",
                "The weather grew warmer each day.",
                "Trees budded with new green leaves.",
                "Allergies flared up from the pollen.",
                "Days grew longer toward summer.",
            ],
            'water': [
                "Fresh water bubbled up from underground.",
                "The hiking trail led to a natural source.",
                "Cool liquid flowed from between rocks.",
                "Animals gathered to drink from it.",
                "The geologist studied the aquifer below.",
                "Pure water collected in a small pool.",
            ],
            'coil': [
                "Metal bounced back when compressed.",
                "The mattress felt too firm underneath.",
                "Tension built up in the mechanism.",
                "The engineer calculated the elasticity.",
                "It snapped back into its original shape.",
                "The clock mechanism relied on it.",
            ],
        },
    },
    'bark': {
        'meanings': ['tree', 'sound'],
        'contexts': {
            'tree': [
                "The rough texture protected the trunk.",
                "Beetles burrowed beneath the surface.",
                "Patterns of grooves covered the oak.",
                "The forester examined the outer layer.",
                "Moss grew on the north side.",
                "Woodpeckers searched for insects inside.",
            ],
            'sound': [
                "The noise echoed through the night.",
                "Neighbors complained about the racket.",
                "The guard dog alerted to intruders.",
                "Sharp yelps came from the backyard.",
                "The puppy made excited noises.",
                "Howling accompanied the loud sounds.",
            ],
        },
    },
    'light': {
        'meanings': ['bright', 'weight'],
        'contexts': {
            'bright': [
                "The lamp illuminated the dark room.",
                "Sunshine streamed through the window.",
                "The beacon guided ships to shore.",
                "Shadows disappeared when it turned on.",
                "The photographer adjusted the exposure.",
                "Candles flickered on the birthday cake.",
            ],
            'weight': [
                "The package was easy to carry.",
                "Aluminum made it less heavy.",
                "She lifted it with one hand easily.",
                "The scale showed only two pounds.",
                "Feathers filled the soft pillow.",
                "The backpack felt almost empty.",
            ],
        },
    },
    'match': {
        'meanings': ['fire', 'game', 'pair'],
        'contexts': {
            'fire': [
                "She struck it against the rough surface.",
                "A small flame ignited the kindling.",
                "The box contained fifty of them.",
                "Sulfur smell filled the air briefly.",
                "The camper lit the campfire carefully.",
                "The wooden stick burned down quickly.",
            ],
            'game': [
                "The tennis players competed fiercely.",
                "The final score was three to two.",
                "Spectators cheered from the stands.",
                "The referee made a controversial call.",
                "Both teams wanted to win badly.",
                "The championship was decided today.",
            ],
            'pair': [
                "The socks were the same color.",
                "She found its partner in the drawer.",
                "The puzzle pieces fit together perfectly.",
                "The colors complemented each other well.",
                "The decorator coordinated the fabrics.",
                "They were a perfect combination together.",
            ],
        },
    },
    'wave': {
        'meanings': ['ocean', 'gesture'],
        'contexts': {
            'ocean': [
                "Surfers waited for the perfect swell.",
                "Water crashed against the rocky shore.",
                "The tide brought larger swells today.",
                "Foam sprayed as it hit the beach.",
                "The sailor watched the rolling sea.",
                "Seagulls rode the wind above the water.",
            ],
            'gesture': [
                "She raised her hand in greeting.",
                "The crowd acknowledged the performer.",
                "He said goodbye from the departing train.",
                "Children signaled to their parents excitedly.",
                "The motion caught her attention immediately.",
                "Arms moved back and forth happily.",
            ],
        },
    },
    'date': {
        'meanings': ['fruit', 'time', 'romantic'],
        'contexts': {
            'fruit': [
                "The sweet flavor came from the desert.",
                "Palm trees produced the sticky treats.",
                "Dried pieces were added to the recipe.",
                "Middle Eastern cuisine featured them often.",
                "The chewy texture was quite pleasant.",
                "Natural sugar made them taste sweet.",
            ],
            'time': [
                "The calendar showed tomorrow circled.",
                "She checked what day it was today.",
                "The appointment was scheduled for Friday.",
                "Years passed since that memorable moment.",
                "The deadline was clearly marked.",
                "Historical records showed the exact year.",
            ],
            'romantic': [
                "They met for dinner at the restaurant.",
                "The couple held hands across the table.",
                "Flowers were presented at the doorstep.",
                "Butterflies filled her stomach nervously.",
                "The movie was perfect for two people.",
                "He asked her out for Saturday night.",
            ],
        },
    },
}

# Ambiguous sentence templates
AMBIGUOUS_TEMPLATES = [
    "The {word} was notable.",
    "The {word} was remarkable.",
    "The {word} was significant.",
    "Consider the {word} carefully.",
    "The {word} was quite interesting.",
    "Notice the {word} here.",
    "The {word} stood out clearly.",
    "The {word} was hard to ignore.",
]

def generate_sample():
    """Generate a single disambiguation sample."""
    # Pick random word
    word = random.choice(list(AMBIGUOUS_DATA.keys()))
    data = AMBIGUOUS_DATA[word]
    
    # Pick random meaning
    meaning = random.choice(data['meanings'])
    
    # Pick random disambiguating context
    context = random.choice(data['contexts'][meaning])
    
    # Pick random ambiguous template
    ambiguous = random.choice(AMBIGUOUS_TEMPLATES).format(word=word)
    
    # Construct input/output
    input_text = f"{ambiguous} {context} MEANING="
    output_text = meaning
    
    return input_text, output_text

def generate_dataset(n_samples):
    """Generate dataset with balanced meanings."""
    samples = []
    
    # Generate samples ensuring some balance
    for _ in range(n_samples):
        inp, out = generate_sample()
        samples.append((inp, out))
    
    return samples

def encode_sample(input_text, output_text):
    """Encode input+output as token sequence."""
    # Simple character-level encoding
    full_text = input_text + output_text
    return [ord(c) for c in full_text]

def main():
    print("=" * 60)
    print("Generating Insight Disambiguation Dataset")
    print("=" * 60)
    
    # Generate datasets
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
    for i in range(5):
        inp, out = train_samples[i]
        print(f"\nInput:  {inp}")
        print(f"Output: {out}")
    
    # Encode to tokens
    print("\n" + "=" * 60)
    print("Encoding samples...")
    print("=" * 60)
    
    train_tokens = []
    for inp, out in train_samples:
        train_tokens.extend(encode_sample(inp, out))
    
    val_tokens = []
    for inp, out in val_samples:
        val_tokens.extend(encode_sample(inp, out))
    
    # Convert to numpy arrays
    train_data = np.array(train_tokens, dtype=np.uint16)
    val_data = np.array(val_tokens, dtype=np.uint16)
    
    print(f"\nTrain tokens: {len(train_data):,}")
    print(f"Val tokens:   {len(val_data):,}")
    
    # Save binary files
    train_data.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
    val_data.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))
    
    # Save metadata
    meta = {
        'vocab_size': 256,  # ASCII
        'task': 'insight_disambiguation',
        'description': 'Uncertainty-driven Aha! moments (Type A)',
    }
    with open(os.path.join(os.path.dirname(__file__), 'meta.pkl'), 'wb') as f:
        pickle.dump(meta, f)
    
    print("\n" + "=" * 60)
    print("Dataset saved!")
    print("  - train.bin")
    print("  - val.bin") 
    print("  - meta.pkl")
    print("=" * 60)
    
    # Statistics
    print("\n" + "=" * 60)
    print("MEANING DISTRIBUTION:")
    print("=" * 60)
    from collections import Counter
    meanings = Counter([out for _, out in train_samples])
    for meaning, count in sorted(meanings.items(), key=lambda x: -x[1]):
        print(f"  {meaning}: {count} ({100*count/n_train:.1f}%)")

if __name__ == "__main__":
    main()
