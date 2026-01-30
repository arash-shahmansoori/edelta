# config/train_reverse_v2.py
# Reversibility task with V2 model (fixed rotation collapse)

out_dir = 'out-reverse-v2'
eval_interval = 500
eval_iters = 20
log_interval = 50

always_save_checkpoint = False

dataset = 'reversibility'
batch_size = 64
block_size = 128

# Model - 4 layer for this task
n_layer = 4
n_head = 4
n_embd = 128
dropout = 0.0

# V2 Geodesic settings (rotation collapse fixed)
use_damper = True
use_static_gate = False
geo_lr_mult = 50.0

# Optimization
learning_rate = 1e-3
max_iters = 10000
lr_decay_iters = 10000
min_lr = 1e-4
beta2 = 0.99

# Hardware
device = 'cuda'
compile = True
