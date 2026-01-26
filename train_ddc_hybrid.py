"""
Training script for Data-Dependent Cayley Hybrid (DDC-Hybrid) Transformer

Combines:
- Data-Dependent Cayley for input-adaptive rotation (guaranteed orthogonal)
- Householder reflection for negation capability
- Learned gate for adaptive selection
"""
import os
import sys
import time
import math
import pickle
from contextlib import nullcontext

import numpy as np
import torch

from proposed_model_ddc_hybrid import GPTConfig, GPT

# -----------------------------------------------------------------------------
# Configuration with defaults
out_dir = 'out-ddc-hybrid'
eval_interval = 500
log_interval = 100
eval_iters = 50
eval_only = False
always_save_checkpoint = False
init_from = 'scratch'

# Data
dataset = 'correction'
gradient_accumulation_steps = 1
batch_size = 64
block_size = 128

# Model
n_layer = 4
n_head = 4
n_embd = 128
n_streams = 4
dropout = 0.0
bias = False

# Training
learning_rate = 1e-3
max_iters = 10000
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0
decay_lr = True
warmup_iters = 500
lr_decay_iters = 10000
min_lr = 1e-4

# System
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
compile = False

# Parse command line
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, str, bool))]
exec(open('configurator.py').read())

# Update output directory
out_dir = f'out-ddc-hybrid-{dataset}'
os.makedirs(out_dir, exist_ok=True)

# Setup
tokens_per_iter = gradient_accumulation_steps * batch_size * block_size
print(f"DDC-Hybrid training on {dataset}")
print(f"Tokens per iteration: {tokens_per_iter:,}")

torch.manual_seed(1337)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# Data loading
data_dir = os.path.join('data', dataset)
train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    if device_type == 'cuda':
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

# Get vocab size
meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    print(f"Found vocab_size = {meta_vocab_size} from {meta_path}")

# Initialize model
model_args = dict(
    n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
    bias=bias, vocab_size=None, dropout=dropout, n_streams=n_streams
)
if init_from == 'scratch':
    if meta_vocab_size is not None:
        model_args['vocab_size'] = meta_vocab_size
    else:
        model_args['vocab_size'] = 50304
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)

model.to(device)
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type) if hasattr(model, 'configure_optimizers') else torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=(beta1, beta2), weight_decay=weight_decay)

if compile:
    print("Compiling model...")
    model = torch.compile(model)

# Learning rate scheduler
def get_lr(it):
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# Training loop
X, Y = get_batch('train')
t0 = time.time()
best_val_loss = float('inf')

for iter_num in range(max_iters + 1):
    lr = get_lr(iter_num)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    if iter_num % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        if losses['val'] < best_val_loss:
            best_val_loss = losses['val']
            if iter_num > 0:
                checkpoint = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                }
                torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
                print(f"Saved checkpoint with val_loss={best_val_loss:.4f}")

    if eval_only:
        break

    for micro_step in range(gradient_accumulation_steps):
        with ctx:
            logits, loss = model(X, Y)
            loss = loss / gradient_accumulation_steps
        X, Y = get_batch('train')
        scaler.scale(loss).backward()

    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)

    if iter_num % log_interval == 0:
        t1 = time.time()
        dt = t1 - t0
        t0 = t1
        lossf = loss.item() * gradient_accumulation_steps
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, lr {lr:.2e}")

print(f"\nDDC-Hybrid Training complete. Best val loss: {best_val_loss:.4f}")

# Print gate analysis if available
if hasattr(model, 'get_gate_values'):
    print("\n=== Gate Analysis ===")
    gates = model.get_gate_values()
    for g in gates:
        print(f"Layer {g['layer']}: attn_gate_bias={g['attn_gate_bias']:.4f}, mlp_gate_bias={g['mlp_gate_bias']:.4f}")
