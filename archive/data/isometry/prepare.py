import os
import random
import numpy as np

def generate_passkey(length=1024):
    # Needle in a Haystack
    key = random.randint(1000, 9999)
    needle = f"The magic number is {key}."
    
    # Generate noise
    noise_tokens = [random.randint(32, 126) for _ in range(length)]
    noise = "".join([chr(t) for t in noise_tokens])
    
    # Insert needle at random depth
    insert_idx = random.randint(0, len(noise) - len(needle))
    context = noise[:insert_idx] + needle + noise[insert_idx:]
    
    # The query
    text = f"{context}\nWhat is the magic number? {key}\n"
    return text

if __name__ == '__main__':
    os.makedirs('data/isometry', exist_ok=True)
    # Generate curriculum of lengths
    data = []
    for length in [512, 1024, 2048, 4096]: # curriculum
        for _ in range(500):
            data.append(generate_passkey(length))
            
    random.shuffle(data)
    train_data = "".join(data[:int(len(data)*0.9)])
    val_data = "".join(data[int(len(data)*0.9):])
    
    np.array([ord(c) for c in train_data], dtype=np.uint16).tofile('data/isometry/train.bin')
    np.array([ord(c) for c in val_data], dtype=np.uint16).tofile('data/isometry/val.bin')
    
    print(f"Saved {len(train_data)} train tokens, {len(val_data)} val tokens")
