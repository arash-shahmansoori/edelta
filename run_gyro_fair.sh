#!/bin/bash
# FAIR Gyroscope Experiment - All configs MATCHED

echo "=============================================="
echo "FAIR GYROSCOPE EXPERIMENT"
echo "All models with IDENTICAL hyperparameters"
echo "=============================================="
echo "Start time: $(date)"
echo ""

# UNIFIED CONFIG - EXACTLY THE SAME FOR ALL MODELS
N_LAYER=6
N_HEAD=6
N_EMBD=384
BATCH_SIZE=64
BLOCK_SIZE=256
GRAD_ACCUM=1
MAX_ITERS=3000
EVAL_INTERVAL=500
LOG_INTERVAL=500
LR="6e-4"
DROPOUT=0.0

echo "Unified Config:"
echo "  n_layer=$N_LAYER, n_head=$N_HEAD, n_embd=$N_EMBD"
echo "  batch_size=$BATCH_SIZE, block_size=$BLOCK_SIZE"
echo "  gradient_accumulation_steps=$GRAD_ACCUM"
echo "  max_iters=$MAX_ITERS, lr=$LR"
echo "  Tokens per iter: $((BATCH_SIZE * BLOCK_SIZE * GRAD_ACCUM))"
echo ""

# Common args
COMMON="--dataset=gyroscope --n_layer=$N_LAYER --n_head=$N_HEAD --n_embd=$N_EMBD --batch_size=$BATCH_SIZE --block_size=$BLOCK_SIZE --gradient_accumulation_steps=$GRAD_ACCUM --max_iters=$MAX_ITERS --eval_interval=$EVAL_INTERVAL --log_interval=$LOG_INTERVAL --learning_rate=$LR --dropout=$DROPOUT --compile=False"

gpu_stats() {
    nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits 2>/dev/null || echo "N/A"
}

# Results array
declare -A RESULTS

# 1. BASELINE
echo "=============================================="
echo "[1/4] BASELINE Transformer"
echo "=============================================="
START=$(date +%s)
OUTPUT=$(python -u train.py $COMMON --out_dir=out-gyro-fair-baseline 2>&1)
echo "$OUTPUT" | grep -E "(step|parameters|val loss)"
VAL_LOSS=$(echo "$OUTPUT" | grep "val loss" | tail -1 | grep -oP "val loss \K[0-9.]+")
END=$(date +%s)
echo "Time: $((END-START))s | Final val_loss: $VAL_LOSS | GPU: $(gpu_stats)"
RESULTS[baseline]=$VAL_LOSS
echo ""

# 2. DDL (Householder)
echo "=============================================="
echo "[2/4] DDL (Householder Reflection)"
echo "=============================================="
START=$(date +%s)
OUTPUT=$(python -u train_ddl.py $COMMON --out_dir=out-gyro-fair-ddl 2>&1)
echo "$OUTPUT" | grep -E "(step|parameters|val loss)"
VAL_LOSS=$(echo "$OUTPUT" | grep "val loss" | tail -1 | grep -oP "val loss \K[0-9.]+")
END=$(date +%s)
echo "Time: $((END-START))s | Final val_loss: $VAL_LOSS | GPU: $(gpu_stats)"
RESULTS[ddl]=$VAL_LOSS
echo ""

# 3. CAYLEY (Pure Rotation)
echo "=============================================="
echo "[3/4] CAYLEY (Pure Rotation)"
echo "=============================================="
START=$(date +%s)
OUTPUT=$(python -u train_cayley.py $COMMON --out_dir=out-gyro-fair-cayley 2>&1)
echo "$OUTPUT" | grep -E "(step|parameters|val loss)"
VAL_LOSS=$(echo "$OUTPUT" | grep "val loss" | tail -1 | grep -oP "val loss \K[0-9.]+")
END=$(date +%s)
echo "Time: $((END-START))s | Final val_loss: $VAL_LOSS | GPU: $(gpu_stats)"
RESULTS[cayley]=$VAL_LOSS
echo ""

# 4. HYBRID (Cayley + Householder)
echo "=============================================="
echo "[4/4] HYBRID (Cayley + Householder)"
echo "=============================================="
START=$(date +%s)
OUTPUT=$(python -u train_hybrid.py $COMMON --out_dir=out-gyro-fair-hybrid 2>&1)
echo "$OUTPUT" | grep -E "(step|parameters|val loss)"
VAL_LOSS=$(echo "$OUTPUT" | grep "val loss" | tail -1 | grep -oP "val loss \K[0-9.]+")
END=$(date +%s)
echo "Time: $((END-START))s | Final val_loss: $VAL_LOSS | GPU: $(gpu_stats)"
RESULTS[hybrid]=$VAL_LOSS
echo ""

echo "=============================================="
echo "FAIR COMPARISON RESULTS"
echo "=============================================="
echo ""
echo "| Model      | Val Loss | Status |"
echo "|------------|----------|--------|"
echo "| Baseline   | ${RESULTS[baseline]} |        |"
echo "| DDL        | ${RESULTS[ddl]} |        |"
echo "| Cayley     | ${RESULTS[cayley]} |        |"
echo "| Hybrid     | ${RESULTS[hybrid]} |        |"
echo ""
echo "End time: $(date)"
