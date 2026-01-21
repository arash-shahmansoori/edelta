# config/train_grok_large.py
# Scaled-up model to test if Geodesic benefits emerge with depth

out_dir = 'out-grok-large-geodesic'
eval_interval = 500
eval_iters = 50
log_interval = 50

always_save_checkpoint = True

dataset = 'grokking'
gradient_accumulation_steps = 2  # Accumulate for effective larger batch
batch_size = 32                  # Smaller batch for memory
block_size = 128

# Model Architecture - Scaled up
n_layer = 12                     # 3x more layers
n_head = 8
n_embd = 512                     # 4x larger embedding (must be divisible by n_streams=4)
dropout = 0.0
bias = False

# Optimization
learning_rate = 3e-4             # Lower LR for larger model
max_iters = 10000
lr_decay_iters = 10000
min_lr = 3e-5
beta1 = 0.9
beta2 = 0.99
weight_decay = 0.1
grad_clip = 1.0

# Hardware
device = 'cuda'
compile = True
