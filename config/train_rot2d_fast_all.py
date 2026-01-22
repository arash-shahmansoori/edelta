# Shared fast config for all models
# gradient_accumulation_steps = 1 for speed

out_dir = 'out-rot2d-fast'
eval_interval = 1000
eval_iters = 20
log_interval = 100

always_save_checkpoint = False

dataset = 'rotation2d'
gradient_accumulation_steps = 1  # FAST - same for all models
batch_size = 64
block_size = 128

n_layer = 4
n_head = 4
n_embd = 128
dropout = 0.0
bias = False

learning_rate = 1e-3
max_iters = 20000  # More iters since less tokens/iter
lr_decay_iters = 20000
min_lr = 1e-4
beta2 = 0.99
weight_decay = 0.1
grad_clip = 1.0

device = 'cuda'
compile = True
