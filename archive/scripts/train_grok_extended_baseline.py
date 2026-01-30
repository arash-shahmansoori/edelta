# config/train_grok_extended_baseline.py
# Extended grokking experiment - BASELINE for comparison

out_dir = 'out-grok-extended-baseline'
eval_interval = 500
eval_iters = 50
log_interval = 50

always_save_checkpoint = True

dataset = 'grokking'
gradient_accumulation_steps = 1
batch_size = 64
block_size = 128

# Model Architecture (same as geodesic)
n_layer = 4
n_head = 4
n_embd = 128
dropout = 0.0
bias = False

# Optimization - Extended training
learning_rate = 1e-3
max_iters = 15000
lr_decay_iters = 15000
min_lr = 1e-5
beta1 = 0.9
beta2 = 0.99
weight_decay = 0.1
grad_clip = 1.0

# Hardware
device = 'cuda'
compile = True
