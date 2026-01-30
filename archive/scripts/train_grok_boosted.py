# config/train_grok_boosted.py
# Full rescue: 50x LR boost + keep damper + full gate
# This is the "kickstarted" version of the original model

out_dir = 'out-grok-boosted'
eval_interval = 250
eval_iters = 20
log_interval = 10

always_save_checkpoint = False

dataset = 'grokking'
gradient_accumulation_steps = 1
batch_size = 64
block_size = 128

# Model Architecture
n_layer = 4
n_head = 4
n_embd = 128
dropout = 0.0
bias = False

# Geodesic settings - BOOSTED
use_damper = True        # Keep damper
use_static_gate = False  # Full thermodynamic gate
geo_lr_mult = 50.0       # 50x LR BOOST - force gate to open!

# Optimization
learning_rate = 1e-3
max_iters = 3000
lr_decay_iters = 3000
min_lr = 1e-4
beta1 = 0.9
beta2 = 0.99
weight_decay = 0.1
grad_clip = 1.0

# Hardware
device = 'cuda'
compile = True
