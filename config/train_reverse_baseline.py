# config/train_reverse_baseline.py
# Baseline model on Reversibility Task

out_dir = 'out-reverse-baseline'
eval_interval = 500
eval_iters = 50
log_interval = 50

always_save_checkpoint = True

dataset = 'reversibility'
gradient_accumulation_steps = 1
batch_size = 64
block_size = 64  # Short sequences for path tasks

# Model Architecture (same as geodesic)
n_layer = 4
n_head = 4
n_embd = 128
dropout = 0.0
bias = False

# Optimization (same hyperparameters)
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
