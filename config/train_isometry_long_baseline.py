# config/train_isometry_long_baseline.py
# Long-context isometry - BASELINE for comparison

out_dir = 'out-isometry-long-baseline'
eval_interval = 500
eval_iters = 50
log_interval = 50

always_save_checkpoint = True

dataset = 'isometry'
gradient_accumulation_steps = 4
batch_size = 4
block_size = 2048

# Model Architecture (same as geodesic)
n_layer = 6
n_head = 8
n_embd = 256
dropout = 0.0
bias = False

# Optimization
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
