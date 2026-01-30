# config/train_grok_tiny_geodesic.py
# TINY model to starve baseline - Geodesic should still solve via rotation

out_dir = 'out-grok-tiny-geodesic'
eval_interval = 500
eval_iters = 50
log_interval = 50

always_save_checkpoint = True

dataset = 'grokking'
gradient_accumulation_steps = 1
batch_size = 64
block_size = 128

# Model Architecture - TINY (starve the baseline)
n_layer = 1              # Single layer!
n_head = 2
n_embd = 32              # Tiny embedding (must be divisible by n_streams=4)
dropout = 0.0
bias = False

# Geodesic ablation settings
use_damper = True        # Keep damper for now
use_static_gate = False  # Use full thermodynamic gate
geo_lr_mult = 50.0       # 50x LR boost for geodesic params

# Optimization
learning_rate = 1e-3
max_iters = 10000        # Longer training for tiny model
lr_decay_iters = 10000
min_lr = 1e-4
beta1 = 0.9
beta2 = 0.99
weight_decay = 0.1
grad_clip = 1.0

# Hardware
device = 'cuda'
compile = True
