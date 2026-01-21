# config/train_erasure_baseline.py
# Baseline GPT on Erasure task ("Did I Stutter?")

out_dir = 'out-erasure-baseline'
eval_interval = 500
eval_iters = 50
log_interval = 10

always_save_checkpoint = False

dataset = 'erasure'
gradient_accumulation_steps = 1
batch_size = 64
block_size = 128  # Short sequences for commands

# Model Architecture (same as geodesic for fair comparison)
n_layer = 4
n_head = 4
n_embd = 128
dropout = 0.0
bias = False

# Optimization (same hyperparameters)
learning_rate = 1e-3
max_iters = 5000
lr_decay_iters = 5000
min_lr = 1e-4
beta1 = 0.9
beta2 = 0.99
weight_decay = 0.1
grad_clip = 1.0

# Hardware
device = 'cuda'
compile = True
