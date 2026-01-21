# config/train_isometry_long.py
# Long-context isometry to test information preservation over distance

out_dir = 'out-isometry-long-geodesic'
eval_interval = 500
eval_iters = 50
log_interval = 50

always_save_checkpoint = True

dataset = 'isometry'
gradient_accumulation_steps = 4  # Accumulate for memory
batch_size = 4                   # Small batch for long sequences
block_size = 2048                # 4x longer context

# Model Architecture - Slightly larger for long context
n_layer = 6
n_head = 8
n_embd = 256                     # Must be divisible by n_streams=4
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
