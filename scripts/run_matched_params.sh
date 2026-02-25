#!/bin/bash
# Run experiments with matched parameter counts (~1.79M for all models)
# 
# RECOMMENDED APPROACH: Scale UP baseline n_layer to match E∆-MHC-Geo
# All models keep n_embd=128 (same representation dimension)
# This tests: does geometric inductive bias beat additional depth?
#
# Parameter counts with --match_proposed_params (all n_embd=128):
#   E∆-MHC-Geo: n_layer=6 → 1.788M (reference)
#   GPT:        n_layer=9 → 1.780M (0.996x)
#   DDL:        n_layer=8 → 1.784M (0.998x)
#   mHC:        n_layer=9 → 1.838M (1.028x)
#   JPmHC:      n_layer=9 → 1.896M (1.061x)

set -e

echo "============================================================"
echo "MATCHED PARAMETER EXPERIMENTS (~1.79M params for all models)"
echo "============================================================"
echo "All models: n_embd=128 (same representation dimension)"
echo "E∆-MHC-Geo: n_layer=6 (with geometric operators)"
echo "Baselines:  n_layer scaled up (more depth to compensate)"
echo "  GPT:   9L  DDL: 8L  mHC: 9L  JPmHC: 9L"
echo "============================================================"

# Verify datasets exist
echo ""
echo "=== Checking datasets ==="
missing=0

for dataset in gyroscope stability; do
    if [ -f "data/${dataset}/train_x.npy" ] && [ -f "data/${dataset}/val_x.npy" ]; then
        echo "✓ ${dataset}: OK"
    else
        echo "✗ ${dataset}: MISSING"
        missing=1
    fi
done

if [ $missing -eq 1 ]; then
    echo ""
    echo "ERROR: Some datasets are missing!"
    echo "Please run: bash scripts/prepare_data.sh"
    exit 1
fi

# Create output directories
mkdir -p out-matched

# Common settings
MAX_ITERS=2000
N_LAYER=6
N_HEAD=4
N_EMBD=128  # Base value (E∆-MHC-Geo uses this, baselines get scaled up)
BATCH_SIZE=64

echo ""
echo "=== GYROSCOPE DATASET ==="
echo ""

# GPT Baseline (n_layer scaled to 9)
echo "Training GPT (n_layer=9, ~1.78M params) on gyroscope..."
uv run src/training/train_continuous.py \
    --model_type gpt2 \
    --dataset gyroscope \
    --out_dir out-matched/gyroscope-baseline \
    --max_iters $MAX_ITERS \
    --n_layer $N_LAYER \
    --n_head $N_HEAD \
    --n_embd $N_EMBD \
    --batch_size $BATCH_SIZE \
    --match_proposed_params

# DDL (n_layer scaled to 8)
echo "Training DDL (n_layer=8, ~1.78M params) on gyroscope..."
uv run src/training/train_continuous.py \
    --model_type ddl \
    --dataset gyroscope \
    --out_dir out-matched/gyroscope-ddl \
    --max_iters $MAX_ITERS \
    --n_layer $N_LAYER \
    --n_head $N_HEAD \
    --n_embd $N_EMBD \
    --batch_size $BATCH_SIZE \
    --match_proposed_params

# mHC (n_layer scaled to 9)
echo "Training mHC (n_layer=9, ~1.84M params) on gyroscope..."
uv run src/training/train_continuous.py \
    --model_type mhc \
    --dataset gyroscope \
    --out_dir out-matched/gyroscope-mhc \
    --max_iters $MAX_ITERS \
    --n_layer $N_LAYER \
    --n_head $N_HEAD \
    --n_embd $N_EMBD \
    --batch_size $BATCH_SIZE \
    --match_proposed_params

# JPmHC (n_layer scaled to 9)
echo "Training JPmHC (n_layer=9, ~1.89M params) on gyroscope..."
uv run src/training/train_continuous.py \
    --model_type jpmhc \
    --dataset gyroscope \
    --out_dir out-matched/gyroscope-jpmhc \
    --max_iters $MAX_ITERS \
    --n_layer $N_LAYER \
    --n_head $N_HEAD \
    --n_embd $N_EMBD \
    --batch_size $BATCH_SIZE \
    --match_proposed_params

# E∆-MHC-Geo (n_layer=6, full design, ~1.79M params)
echo "Training E∆-MHC-Geo (n_layer=6, ~1.79M params) on gyroscope..."
uv run src/training/train_continuous.py \
    --model_type edelta \
    --dataset gyroscope \
    --out_dir out-matched/gyroscope-proposed \
    --max_iters $MAX_ITERS \
    --n_layer $N_LAYER \
    --n_head $N_HEAD \
    --n_embd $N_EMBD \
    --batch_size $BATCH_SIZE

echo ""
echo "=== STABILITY DATASET ==="
echo ""

# GPT Baseline (n_layer scaled to 9)
echo "Training GPT (n_layer=9, ~1.78M params) on stability..."
uv run src/training/train_continuous.py \
    --model_type gpt2 \
    --dataset stability \
    --out_dir out-matched/stability-baseline \
    --max_iters $MAX_ITERS \
    --n_layer $N_LAYER \
    --n_head $N_HEAD \
    --n_embd $N_EMBD \
    --batch_size $BATCH_SIZE \
    --match_proposed_params

# DDL (n_layer scaled to 8)
echo "Training DDL (n_layer=8, ~1.78M params) on stability..."
uv run src/training/train_continuous.py \
    --model_type ddl \
    --dataset stability \
    --out_dir out-matched/stability-ddl \
    --max_iters $MAX_ITERS \
    --n_layer $N_LAYER \
    --n_head $N_HEAD \
    --n_embd $N_EMBD \
    --batch_size $BATCH_SIZE \
    --match_proposed_params

# mHC (n_layer scaled to 9)
echo "Training mHC (n_layer=9, ~1.84M params) on stability..."
uv run src/training/train_continuous.py \
    --model_type mhc \
    --dataset stability \
    --out_dir out-matched/stability-mhc \
    --max_iters $MAX_ITERS \
    --n_layer $N_LAYER \
    --n_head $N_HEAD \
    --n_embd $N_EMBD \
    --batch_size $BATCH_SIZE \
    --match_proposed_params

# JPmHC (n_layer scaled to 9)
echo "Training JPmHC (n_layer=9, ~1.89M params) on stability..."
uv run src/training/train_continuous.py \
    --model_type jpmhc \
    --dataset stability \
    --out_dir out-matched/stability-jpmhc \
    --max_iters $MAX_ITERS \
    --n_layer $N_LAYER \
    --n_head $N_HEAD \
    --n_embd $N_EMBD \
    --batch_size $BATCH_SIZE \
    --match_proposed_params

# E∆-MHC-Geo (n_layer=6, full design)
echo "Training E∆-MHC-Geo (n_layer=6, ~1.79M params) on stability..."
uv run src/training/train_continuous.py \
    --model_type edelta \
    --dataset stability \
    --out_dir out-matched/stability-proposed \
    --max_iters $MAX_ITERS \
    --n_layer $N_LAYER \
    --n_head $N_HEAD \
    --n_embd $N_EMBD \
    --batch_size $BATCH_SIZE

echo ""
echo "============================================================"
echo "ALL EXPERIMENTS COMPLETE"
echo "============================================================"
echo ""
echo "Results saved to out-matched/"
echo ""
echo "Generate figures with:"
echo "  uv run src/visualization/visualize_journal.py"
