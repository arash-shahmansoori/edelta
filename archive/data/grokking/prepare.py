# data/grokking/prepare.py
import os
import random
import numpy as np

def encode(s):
    return [ord(c) for c in s] # Simple ASCII encoding for math

if __name__ == '__main__':
    os.makedirs('data/grokking', exist_ok=True)
    
    # Generate Modulo Arithmetic: a + b = c (mod 97)
    p = 97
    data = []
    for a in range(p):
        for b in range(p):
            res = (a + b) % p
            # Format: "a+b=c\n"
            text = f"{a}+{b}={res}\n"
            data.append(text)
            
    # Shuffle
    random.seed(42)
    random.shuffle(data)
    
    # Split 90/10
    n = len(data)
    train_data = "".join(data[:int(n*0.9)])
    val_data = "".join(data[int(n*0.9):])
    
    # Encode
    train_ids = encode(train_data)
    val_ids = encode(val_data)
    
    # Save as bin
    train_ids = np.array(train_ids, dtype=np.uint16)
    val_ids = np.array(val_ids, dtype=np.uint16)
    train_ids.tofile(os.path.join('data/grokking', 'train.bin'))
    val_ids.tofile(os.path.join('data/grokking', 'val.bin'))
    
    print(f"Saved {len(train_ids)} tokens to train.bin")
