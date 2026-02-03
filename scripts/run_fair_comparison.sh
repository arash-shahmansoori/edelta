#!/bin/bash
# Run fair comparison experiments
# E∆-MHC-Geo uses geo_hidden_ratio=8 (reduced from 4) for smaller geo networks

set -e

echo "============================================================"
echo "FAIR COMPARISON EXPERIMENTS"
echo "E∆-MHC-Geo: geo_hidden_ratio=8 (n_embd//8 instead of n_embd//4)"
echo "============================================================"

# Create output directories
mkdir -p out-gradnorm

# Common settings
MAX_ITERS=2000
N_LAYER=6
N_HEAD=4
N_EMBD=128
BATCH_SIZE=64

echo ""
echo "=== GYROSCOPE DATASET ==="
echo ""

# GPT Baseline
echo "Training GPT (baseline) on gyroscope..."
uv run src/training/train_continuous.py \
    --model_type gpt2 \
    --dataset gyroscope \
    --out_dir out-gradnorm/gyroscope-baseline \
    --max_iters $MAX_ITERS \
    --n_layer $N_LAYER \
    --n_head $N_HEAD \
    --n_embd $N_EMBD \
    --batch_size $BATCH_SIZE

# DDL
echo "Training DDL on gyroscope..."
uv run src/training/train_continuous.py \
    --model_type ddl \
    --dataset gyroscope \
    --out_dir out-gradnorm/gyroscope-ddl \
    --max_iters $MAX_ITERS \
    --n_layer $N_LAYER \
    --n_head $N_HEAD \
    --n_embd $N_EMBD \
    --batch_size $BATCH_SIZE

# mHC
echo "Training mHC on gyroscope..."
uv run src/training/train_continuous.py \
    --model_type mhc \
    --dataset gyroscope \
    --out_dir out-gradnorm/gyroscope-mhc \
    --max_iters $MAX_ITERS \
    --n_layer $N_LAYER \
    --n_head $N_HEAD \
    --n_embd $N_EMBD \
    --batch_size $BATCH_SIZE

# E∆-MHC-Geo (with fair_params - smaller geo networks: n_embd//8)
echo "Training E∆-MHC-Geo (fair params: geo_hidden_ratio=8) on gyroscope..."
uv run src/training/train_continuous.py \
    --model_type edelta \
    --dataset gyroscope \
    --out_dir out-gradnorm/gyroscope-proposed \
    --max_iters $MAX_ITERS \
    --n_layer $N_LAYER \
    --n_head $N_HEAD \
    --n_embd $N_EMBD \
    --batch_size $BATCH_SIZE \
    --fair_params

echo ""
echo "=== STABILITY DATASET ==="
echo ""

# GPT Baseline
echo "Training GPT (baseline) on stability..."
uv run src/training/train_continuous.py \
    --model_type gpt2 \
    --dataset stability \
    --out_dir out-gradnorm/stability-baseline \
    --max_iters $MAX_ITERS \
    --n_layer $N_LAYER \
    --n_head $N_HEAD \
    --n_embd $N_EMBD \
    --batch_size $BATCH_SIZE

# DDL
echo "Training DDL on stability..."
uv run src/training/train_continuous.py \
    --model_type ddl \
    --dataset stability \
    --out_dir out-gradnorm/stability-ddl \
    --max_iters $MAX_ITERS \
    --n_layer $N_LAYER \
    --n_head $N_HEAD \
    --n_embd $N_EMBD \
    --batch_size $BATCH_SIZE

# mHC
echo "Training mHC on stability..."
uv run src/training/train_continuous.py \
    --model_type mhc \
    --dataset stability \
    --out_dir out-gradnorm/stability-mhc \
    --max_iters $MAX_ITERS \
    --n_layer $N_LAYER \
    --n_head $N_HEAD \
    --n_embd $N_EMBD \
    --batch_size $BATCH_SIZE

# E∆-MHC-Geo (with fair_params - smaller geo networks: n_embd//8)
echo "Training E∆-MHC-Geo (fair params: geo_hidden_ratio=8) on stability..."
uv run src/training/train_continuous.py \
    --model_type edelta \
    --dataset stability \
    --out_dir out-gradnorm/stability-proposed \
    --max_iters $MAX_ITERS \
    --n_layer $N_LAYER \
    --n_head $N_HEAD \
    --n_embd $N_EMBD \
    --batch_size $BATCH_SIZE \
    --fair_params

echo ""
echo "=== GENERATING FIGURES ==="
echo ""

# Generate visualization
uv run src/visualization/visualize_journal.py

echo ""
echo "============================================================"
echo "FAIR COMPARISON COMPLETE!"
echo "Results saved to results/"
echo "============================================================"
