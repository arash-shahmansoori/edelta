# config/train_reverse_geodesic.py
# Geodesic model on Reversibility Task
# This task SHOULD benefit from rotation due to inherent cancellation structure

out_dir = 'out-reverse-geodesic'
eval_interval = 500
eval_iters = 50
log_interval = 50

always_save_checkpoint = True

dataset = 'reversibility'
gradient_accumulation_steps = 1
batch_size = 64
block_size = 64  # Short sequences for path tasks

# Model Architecture
n_layer = 4
n_head = 4
n_embd = 128  # Must be divisible by n_streams=4
dropout = 0.0
bias = False

# Geodesic settings - BOOSTED
use_damper = True        # Keep damper
use_static_gate = False  # Full thermodynamic gate
geo_lr_mult = 50.0       # 50x LR boost

# Optimization
learning_rate = 1e-3
max_iters = 10000        # Longer training
lr_decay_iters = 10000
min_lr = 1e-4
beta1 = 0.9
beta2 = 0.99
weight_decay = 0.1
grad_clip = 1.0

# Hardware
device = 'cuda'
compile = True
