# config/train_grok_tiny_baseline.py
# TINY baseline model - should FAIL to grok (not enough capacity)

out_dir = 'out-grok-tiny-baseline'
eval_interval = 500
eval_iters = 50
log_interval = 50

always_save_checkpoint = True

dataset = 'grokking'
gradient_accumulation_steps = 1
batch_size = 64
block_size = 128

# Model Architecture - TINY (same as geodesic)
n_layer = 1              # Single layer!
n_head = 2
n_embd = 32              # Tiny embedding
dropout = 0.0
bias = False

# Optimization (same as geodesic)
learning_rate = 1e-3
max_iters = 10000
lr_decay_iters = 10000
min_lr = 1e-4
beta1 = 0.9
beta2 = 0.99
weight_decay = 0.1
grad_clip = 1.0

# Hardware
device = 'cuda'
compile = True
