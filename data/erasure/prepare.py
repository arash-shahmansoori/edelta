import os
import random
import numpy as np

# A simple vocabulary of actions
ACTIONS = ["North", "South", "East", "West", "Red", "Blue", "Green", "Yellow"]
NEGATIONS = ["Wait", "No", "Ignore", "Correction", "Stop"]

def encode(s):
    return [ord(c) for c in s]

def generate_sample():
    # 50% chance of a standard command: "Go North."
    # 50% chance of a negated command: "Go North. Wait, go South."
    
    target_action = random.choice(ACTIONS)
    
    if random.random() < 0.5:
        # Standard
        prompt = f"Go {target_action}."
        target = target_action
    else:
        # Negated (The Erasure Task)
        wrong_action = random.choice([a for a in ACTIONS if a != target_action])
        negation = random.choice(NEGATIONS)
        prompt = f"Go {wrong_action}. {negation}, go {target_action}."
        target = target_action
        
    return f"{prompt} -> {target}\n"

if __name__ == '__main__':
    os.makedirs('data/erasure', exist_ok=True)
    data = [generate_sample() for _ in range(50000)]
    
    # Save standard train/val split
    n = len(data)
    train_data = "".join(data[:int(n*0.9)])
    val_data = "".join(data[int(n*0.9):])
    
    np.array(encode(train_data), dtype=np.uint16).tofile('data/erasure/train.bin')
    np.array(encode(val_data), dtype=np.uint16).tofile('data/erasure/val.bin')
    
    print(f"Saved {len(encode(train_data))} train tokens, {len(encode(val_data))} val tokens")
