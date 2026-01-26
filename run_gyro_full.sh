#!/bin/bash
# Full Gyroscope Experiment with GPU Tracking

echo "=============================================="
echo "GYROSCOPE EXPERIMENT - DDL vs Cayley vs Hybrid"
echo "=============================================="
echo "Start time: $(date)"
echo ""

# Common settings
COMMON="--dataset=gyroscope --n_layer=6 --n_head=6 --n_embd=384 --batch_size=64 --block_size=256 --max_iters=2000 --eval_interval=400 --log_interval=400 --learning_rate=6e-4 --dropout=0.0 --compile=False"

# Function to show GPU stats
gpu_stats() {
    nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu --format=csv,noheader,nounits
}

echo "Initial GPU state:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
echo ""

# 1. BASELINE
echo "=============================================="
echo "[1/4] BASELINE Transformer"
echo "=============================================="
START=$(date +%s)
python -u train.py $COMMON --out_dir=out-gyro-baseline 2>&1 | grep -E "(step|iter|loss|parameters)"
END=$(date +%s)
echo "Time: $((END-START))s | GPU: $(gpu_stats)"
echo ""

# 2. DDL (Householder)
echo "=============================================="
echo "[2/4] DDL (Householder Reflection)"
echo "=============================================="
START=$(date +%s)
python -u train_ddl.py $COMMON --out_dir=out-gyro-ddl 2>&1 | grep -E "(step|iter|loss|parameters)"
END=$(date +%s)
echo "Time: $((END-START))s | GPU: $(gpu_stats)"
echo ""

# 3. CAYLEY (Pure Rotation)
echo "=============================================="
echo "[3/4] CAYLEY (Pure Rotation - Proposed)"
echo "=============================================="
START=$(date +%s)
python -u train_cayley.py $COMMON --out_dir=out-gyro-cayley 2>&1 | grep -E "(step|iter|loss|parameters)"
END=$(date +%s)
echo "Time: $((END-START))s | GPU: $(gpu_stats)"
echo ""

# 4. HYBRID (Cayley + Householder)
echo "=============================================="
echo "[4/4] HYBRID (Cayley + Householder Gate)"
echo "=============================================="
START=$(date +%s)
python -u train_hybrid.py $COMMON --out_dir=out-gyro-hybrid 2>&1 | grep -E "(step|iter|loss|parameters)"
END=$(date +%s)
echo "Time: $((END-START))s | GPU: $(gpu_stats)"
echo ""

echo "=============================================="
echo "EXPERIMENT COMPLETE"
echo "=============================================="
echo "End time: $(date)"

# Summary
echo ""
echo "RESULTS SUMMARY:"
echo "================"
for dir in out-gyro-baseline out-gyro-ddl out-gyro-cayley out-gyro-hybrid; do
    if [ -f "$dir/ckpt.pt" ]; then
        echo "$dir: checkpoint saved"
    else
        echo "$dir: NO checkpoint"
    fi
done
