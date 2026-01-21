# config/train_grok.py

out_dir = 'out-grok-geodesic'
eval_interval = 250
eval_iters = 20
log_interval = 10

always_save_checkpoint = False

dataset = 'grokking'
gradient_accumulation_steps = 1  # Single GPU
batch_size = 64
block_size = 128  # Short sequences for math

# Model Architecture
n_layer = 4
n_head = 4
n_embd = 128  # Must be divisible by n_streams=4 (128/4=32 ✓)
dropout = 0.0
bias = False

# Optimization (per hyperparameter table)
learning_rate = 1e-3
max_iters = 3000
lr_decay_iters = 3000
min_lr = 1e-4
beta1 = 0.9
beta2 = 0.99
weight_decay = 0.1  # Apply to mixing matrices to prevent rotation drift
grad_clip = 1.0  # Prevents instability if beta spikes

# Hardware
device = 'cuda'
compile = True
