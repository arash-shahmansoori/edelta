# config/train_grok_mhc.py
# Grokking task with mHC-only model (mixing without rotation)

out_dir = 'out-grok-mhc'
eval_interval = 250
eval_iters = 20
log_interval = 50

always_save_checkpoint = False

dataset = 'grokking'
batch_size = 64
block_size = 128

# Model
n_layer = 4
n_head = 4
n_embd = 128
dropout = 0.0

# Optimization
learning_rate = 1e-3
max_iters = 5000
lr_decay_iters = 5000
min_lr = 1e-4
beta2 = 0.99

# Hardware
device = 'cuda'
compile = True
