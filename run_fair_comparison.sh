#!/bin/bash
# =============================================================================
# FAIR COMPARISON: DDC vs Baselines on Continuous Rotation
# =============================================================================
# 
# This script ensures ALL models use IDENTICAL hyperparameters.
# The continuous rotation task tests true geometric capabilities:
# - Norm preservation (DDC should excel due to orthogonality)
# - Rotation accuracy
# - Long-term stability
#
# Expected results based on theory:
# - Baseline: Medium (no geometric inductive bias)
# - DDL: May have norm drift (only orthogonal at β=2)
# - DDC: Best (unconditional orthogonality, perfect norm preservation)
# - DDC-Hybrid: Best (combines rotation + reflection)

set -e

echo "=============================================="
echo "FAIR COMPARISON: Continuous Vector Rotation"
echo "=============================================="

# =============================================================================
# UNIFIED HYPERPARAMETERS (identical for ALL models)
# =============================================================================
DIM=16              # Vector dimension
HIDDEN_DIM=128      # Hidden dimension
N_LAYERS=4          # Number of layers
BATCH_SIZE=64       # Batch size
MAX_ITERS=5000      # Training iterations
LR=0.001            # Learning rate
EVAL_INTERVAL=500   # Evaluation frequency
LOG_INTERVAL=100    # Logging frequency
EVAL_ITERS=50       # Evaluation iterations

echo ""
echo "=== UNIFIED CONFIGURATION ==="
echo "Vector dim:    $DIM"
echo "Hidden dim:    $HIDDEN_DIM"
echo "Layers:        $N_LAYERS"
echo "Batch size:    $BATCH_SIZE"
echo "Max iters:     $MAX_ITERS"
echo "Learning rate: $LR"
echo "=============================="
echo ""

# Step 1: Prepare dataset (if needed)
if [ ! -f "data/continuous_rotation/train.npy" ]; then
    echo "Preparing continuous rotation dataset..."
    python data/continuous_rotation/prepare.py
    echo ""
fi

# Step 2: Train all models with IDENTICAL configs
MODELS=("baseline" "ddl" "ddc" "ddc_hybrid")

for model in "${MODELS[@]}"; do
    echo "=============================================="
    echo "Training: $model"
    echo "=============================================="
    
    python train_continuous_rotation.py \
        --model_type=$model \
        --dim=$DIM \
        --hidden_dim=$HIDDEN_DIM \
        --n_layers=$N_LAYERS \
        --batch_size=$BATCH_SIZE \
        --max_iters=$MAX_ITERS \
        --lr=$LR \
        --eval_interval=$EVAL_INTERVAL \
        --log_interval=$LOG_INTERVAL \
        --eval_iters=$EVAL_ITERS
    
    echo ""
done

# Step 3: Analyze results
echo "=============================================="
echo "RESULTS SUMMARY"
echo "=============================================="

python << 'EOF'
import torch
import os

models = ["baseline", "ddl", "ddc", "ddc_hybrid"]
results = []

print("\n" + "=" * 60)
print("MODEL COMPARISON (Continuous Vector Rotation)")
print("=" * 60)
print(f"{'Model':<15} {'Val Loss':<12} {'Norm Error':<12} {'Status'}")
print("-" * 60)

for model in models:
    ckpt_path = f"out-continuous-{model}/ckpt.pt"
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location='cpu')
        val_loss = ckpt.get('best_val_loss', float('inf'))
        norm_error = ckpt.get('norm_error', float('inf'))
        results.append((model, val_loss, norm_error))
        print(f"{model:<15} {val_loss:<12.6f} {norm_error:<12.6f} ✓")
    else:
        print(f"{model:<15} {'N/A':<12} {'N/A':<12} ✗ (no checkpoint)")

if results:
    print("\n" + "=" * 60)
    print("RANKING BY VALIDATION LOSS (lower is better)")
    print("=" * 60)
    results.sort(key=lambda x: x[1])
    for rank, (model, loss, norm_err) in enumerate(results, 1):
        star = "⭐" if rank == 1 else "  "
        print(f"{star} {rank}. {model}: val_loss={loss:.6f}, norm_error={norm_err:.6f}")
    
    print("\n" + "=" * 60)
    print("RANKING BY NORM PRESERVATION (lower error is better)")
    print("=" * 60)
    results.sort(key=lambda x: x[2])
    for rank, (model, loss, norm_err) in enumerate(results, 1):
        star = "⭐" if rank == 1 else "  "
        print(f"{star} {rank}. {model}: norm_error={norm_err:.6f}")

print("\n" + "=" * 60)
print("THEORETICAL EXPECTATIONS:")
print("=" * 60)
print("• DDC should have BEST norm preservation (unconditional orthogonality)")
print("• DDL may have norm drift (orthogonal only at β=2)")
print("• Baseline has no geometric inductive bias")
print("• DDC-Hybrid should be competitive (combines both)")
print("=" * 60)
EOF

echo ""
echo "Experiment complete!"
