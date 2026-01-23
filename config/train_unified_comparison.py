"""
Unified Configuration for Fair Model Comparison

This config ensures ALL models use IDENTICAL hyperparameters
for fair comparison across experiments.

Usage:
    python train.py config/train_unified_comparison.py --dataset=rotation3d
    python train_geodesic.py config/train_unified_comparison.py --dataset=rotation3d
    python train_mhc_real.py config/train_unified_comparison.py --dataset=rotation3d
    python train_ddl.py config/train_unified_comparison.py --dataset=rotation3d
"""

# ============================================================
# OUTPUT DIRECTORY (will be overridden by run script)
# ============================================================
out_dir = 'out-unified'

# ============================================================
# DATA CONFIG
# ============================================================
dataset = 'rotation2d'  # Default, override via command line
gradient_accumulation_steps = 1
batch_size = 64
block_size = 128

# ============================================================
# MODEL ARCHITECTURE (IDENTICAL FOR ALL MODELS)
# ============================================================
n_layer = 4
n_head = 4
n_embd = 128  # Must be divisible by n_streams=4 for E∆ and mHC
dropout = 0.0
bias = False

# ============================================================
# OPTIMIZER (IDENTICAL FOR ALL MODELS)
# ============================================================
learning_rate = 1e-3
weight_decay = 0.1
beta1 = 0.9
beta2 = 0.99
grad_clip = 1.0

# ============================================================
# LEARNING RATE SCHEDULE (IDENTICAL FOR ALL MODELS)
# ============================================================
decay_lr = True
warmup_iters = 500
lr_decay_iters = 10000
min_lr = 1e-4

# ============================================================
# TRAINING DURATION
# ============================================================
max_iters = 10000
eval_interval = 500
log_interval = 100
eval_iters = 50

# ============================================================
# SYSTEM
# ============================================================
device = 'cuda'
dtype = 'bfloat16'
compile = True  # Use torch.compile for speed

# ============================================================
# E∆-MHC-GEO SPECIFIC (only affects E∆ model)
# ============================================================
use_damper = True
use_static_gate = False
geo_lr_mult = 10.0  # 10x learning rate for geodesic params

# ============================================================
# PURE mHC SPECIFIC (only affects mHC model)
# ============================================================
n_streams = 4
n_sinkhorn_iters = 20

# ============================================================
# SUMMARY OF FAIR COMPARISON SETTINGS
# ============================================================
# All models will have:
#   - Same architecture: 4 layers, 4 heads, 128 embedding
#   - Same optimizer: AdamW with lr=1e-3, weight_decay=0.1
#   - Same schedule: 500 warmup, cosine decay to 1e-4
#   - Same training: 10000 iters, batch_size=64, block_size=128
#   - Same evaluation: every 500 steps, 50 eval batches
# ============================================================
