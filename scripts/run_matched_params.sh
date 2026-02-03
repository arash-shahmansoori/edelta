#!/bin/bash
# Run experiments with matched parameter counts (~1.79M for all models)
# 
# RECOMMENDED APPROACH: Scale UP baselines to match E∆-MHC-Geo
# This keeps E∆-MHC-Geo at full capacity while giving baselines MORE parameters
# If baselines still lose, that's a stronger result!
#
# Parameter counts with --match_proposed_params:
#   E∆-MHC-Geo: n_embd=128, geo_hidden_ratio=4 → 1.788M (reference)
#   GPT:        n_embd=156 → 1.764M (0.987x)
#   DDL:        n_embd=148 → 1.789M (1.001x)
#   mHC:        n_embd=156 → 1.811M (1.013x)

set -e

echo "============================================================"
echo "MATCHED PARAMETER EXPERIMENTS (~1.79M params for all models)"
echo "============================================================"
echo "E∆-MHC-Geo: n_embd=128 (full design)"
echo "Baselines:  n_embd scaled up to match"
echo "============================================================"

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

# GPT Baseline (n_embd scaled to 156)
echo "Training GPT (n_embd=156, ~1.76M params) on gyroscope..."
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

# DDL (n_embd scaled to 148)
echo "Training DDL (n_embd=148, ~1.79M params) on gyroscope..."
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

# mHC (n_embd scaled to 156)
echo "Training mHC (n_embd=156, ~1.81M params) on gyroscope..."
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

# E∆-MHC-Geo (n_embd=128, full design, ~1.79M params)
echo "Training E∆-MHC-Geo (n_embd=128, ~1.79M params) on gyroscope..."
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

# GPT Baseline
echo "Training GPT (n_embd=156, ~1.76M params) on stability..."
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

# DDL
echo "Training DDL (n_embd=148, ~1.79M params) on stability..."
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

# mHC
echo "Training mHC (n_embd=156, ~1.81M params) on stability..."
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

# E∆-MHC-Geo
echo "Training E∆-MHC-Geo (n_embd=128, ~1.79M params) on stability..."
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
