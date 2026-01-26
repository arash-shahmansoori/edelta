#!/bin/bash
# Comprehensive DDC Comparison Experiment
# Compares: Baseline, DDL, DDC (pure), DDC-Hybrid on Correction Task

set -e

echo "=============================================="
echo "DDC COMPARISON EXPERIMENT"
echo "Dataset: Correction (tests negation + rotation)"
echo "=============================================="

# Unified hyperparameters for fair comparison
N_LAYER=4
N_HEAD=4
N_EMBD=128
BATCH_SIZE=64
BLOCK_SIZE=128
MAX_ITERS=5000
LR=1e-3
WARMUP=500
EVAL_INTERVAL=500
LOG_INTERVAL=100
DATASET="correction"

# Check if correction dataset exists
if [ ! -f "data/correction/train.bin" ]; then
    echo "Preparing correction dataset..."
    python data/correction/prepare.py
fi

echo ""
echo "=== Configuration ==="
echo "Layers: $N_LAYER, Heads: $N_HEAD, Embed: $N_EMBD"
echo "Batch: $BATCH_SIZE, Block: $BLOCK_SIZE, Iters: $MAX_ITERS"
echo ""

# 1. Baseline Transformer
echo "=============================================="
echo "1/4: Training BASELINE Transformer..."
echo "=============================================="
python train.py \
    --dataset=$DATASET \
    --n_layer=$N_LAYER \
    --n_head=$N_HEAD \
    --n_embd=$N_EMBD \
    --batch_size=$BATCH_SIZE \
    --block_size=$BLOCK_SIZE \
    --max_iters=$MAX_ITERS \
    --learning_rate=$LR \
    --warmup_iters=$WARMUP \
    --eval_interval=$EVAL_INTERVAL \
    --log_interval=$LOG_INTERVAL \
    --gradient_accumulation_steps=1 \
    --dropout=0.0 \
    --bias=False \
    --compile=False

# 2. DDL (Deep Delta Learning - Householder)
echo ""
echo "=============================================="
echo "2/4: Training DDL (Householder Reflection)..."
echo "=============================================="
python train_ddl.py \
    --dataset=$DATASET \
    --n_layer=$N_LAYER \
    --n_head=$N_HEAD \
    --n_embd=$N_EMBD \
    --batch_size=$BATCH_SIZE \
    --block_size=$BLOCK_SIZE \
    --max_iters=$MAX_ITERS \
    --learning_rate=$LR \
    --warmup_iters=$WARMUP \
    --eval_interval=$EVAL_INTERVAL \
    --log_interval=$LOG_INTERVAL \
    --gradient_accumulation_steps=1 \
    --dropout=0.0 \
    --bias=False \
    --compile=False

# 3. DDC (Data-Dependent Cayley - pure rotation)
echo ""
echo "=============================================="
echo "3/4: Training DDC (Data-Dependent Cayley)..."
echo "=============================================="
python train_ddc.py \
    --dataset=$DATASET \
    --n_layer=$N_LAYER \
    --n_head=$N_HEAD \
    --n_embd=$N_EMBD \
    --batch_size=$BATCH_SIZE \
    --block_size=$BLOCK_SIZE \
    --max_iters=$MAX_ITERS \
    --learning_rate=$LR \
    --warmup_iters=$WARMUP \
    --eval_interval=$EVAL_INTERVAL \
    --log_interval=$LOG_INTERVAL \
    --gradient_accumulation_steps=1 \
    --dropout=0.0 \
    --bias=False \
    --compile=False

# 4. DDC-Hybrid (DDC + Householder with gate)
echo ""
echo "=============================================="
echo "4/4: Training DDC-Hybrid (Rotation + Reflection)..."
echo "=============================================="
python train_ddc_hybrid.py \
    --dataset=$DATASET \
    --n_layer=$N_LAYER \
    --n_head=$N_HEAD \
    --n_embd=$N_EMBD \
    --batch_size=$BATCH_SIZE \
    --block_size=$BLOCK_SIZE \
    --max_iters=$MAX_ITERS \
    --learning_rate=$LR \
    --warmup_iters=$WARMUP \
    --eval_interval=$EVAL_INTERVAL \
    --log_interval=$LOG_INTERVAL \
    --gradient_accumulation_steps=1 \
    --dropout=0.0 \
    --bias=False \
    --compile=False

echo ""
echo "=============================================="
echo "EXPERIMENT COMPLETE!"
echo "=============================================="
echo ""
echo "Results Summary:"
echo "----------------"

# Extract best validation losses from output directories
python << 'EOF'
import torch
import os

models = {
    "baseline": f"out-baseline-correction",
    "ddl": f"out-ddl-correction", 
    "ddc": f"out-ddc-correction",
    "ddc-hybrid": f"out-ddc-hybrid-correction"
}

results = []
for name, dir in models.items():
    ckpt_path = os.path.join(dir, "ckpt.pt")
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location='cpu')
        val_loss = ckpt.get('best_val_loss', 'N/A')
        results.append((name, val_loss))
        print(f"{name:15s}: val_loss = {val_loss:.4f}")
    else:
        print(f"{name:15s}: No checkpoint found")

print()
print("=" * 50)
print("RANKING (lower is better):")
print("=" * 50)
results.sort(key=lambda x: x[1] if isinstance(x[1], float) else float('inf'))
for rank, (name, loss) in enumerate(results, 1):
    if isinstance(loss, float):
        print(f"{rank}. {name}: {loss:.4f}")
EOF

echo ""
echo "Expected Results (based on theory):"
echo "- Baseline: High loss (no geometric operators)"
echo "- DDL: LOW loss (can negate via Householder)"
echo "- DDC: Medium-High loss (cannot negate, rotation only)"
echo "- DDC-Hybrid: LOW loss (can both rotate AND negate)"
