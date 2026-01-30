# config/train_rot2d_baseline_nocompile.py
# 2D Rotation task - Baseline (NO compile for speed)

out_dir = 'out-rot2d-baseline'
eval_interval = 500
eval_iters = 20
log_interval = 50

always_save_checkpoint = False

dataset = 'rotation2d'
batch_size = 64
block_size = 128

n_layer = 4
n_head = 4
n_embd = 128
dropout = 0.0

learning_rate = 1e-3
max_iters = 10000
lr_decay_iters = 10000
min_lr = 1e-4
beta2 = 0.99

device = 'cuda'
compile = False  # Disabled for speed
