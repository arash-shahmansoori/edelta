#!/bin/bash
# Prepare all datasets for continuous benchmark experiments
#
# Usage:
#   bash scripts/prepare_data.sh          # Prepare all datasets
#   bash scripts/prepare_data.sh gyroscope  # Prepare only gyroscope
#   bash scripts/prepare_data.sh stability  # Prepare only stability

set -e

echo "============================================================"
echo "DATA PREPARATION FOR E∆-MHC-GEO EXPERIMENTS"
echo "============================================================"

# Function to prepare gyroscope dataset
prepare_gyroscope() {
    echo ""
    echo "=== Preparing Gyroscope Dataset ==="
    echo "Task: Continuous rotation prediction (tests manifold precision)"
    echo ""
    
    uv run src/data/gyroscope.py \
        --output_dir data/gyroscope \
        --dim 16 \
        --seq_len 256 \
        --n_train 9000 \
        --n_val 1000 \
        --seed 42
    
    echo ""
    echo "Gyroscope dataset ready at data/gyroscope/"
}

# Function to prepare stability dataset  
prepare_stability() {
    echo ""
    echo "=== Preparing Stability Dataset ==="
    echo "Task: Long-horizon identity (tests unconditional isometry)"
    echo ""
    
    uv run src/data/stability.py \
        --output_dir data/stability \
        --dim 64 \
        --n_train 900 \
        --n_val 100 \
        --train_seq_len 128 \
        --seed 42
    
    echo ""
    echo "Stability dataset ready at data/stability/"
}

# Function to verify datasets exist
verify_datasets() {
    echo ""
    echo "=== Verifying Datasets ==="
    
    missing=0
    
    for dataset in gyroscope stability; do
        if [ -f "data/${dataset}/train_x.npy" ] && [ -f "data/${dataset}/val_x.npy" ]; then
            echo "✓ ${dataset}: OK"
        else
            echo "✗ ${dataset}: MISSING"
            missing=1
        fi
    done
    
    return $missing
}

# Main logic
case "${1:-all}" in
    gyroscope)
        prepare_gyroscope
        ;;
    stability)
        prepare_stability
        ;;
    verify)
        verify_datasets
        ;;
    all)
        prepare_gyroscope
        prepare_stability
        echo ""
        echo "============================================================"
        echo "ALL DATASETS PREPARED SUCCESSFULLY"
        echo "============================================================"
        verify_datasets
        ;;
    *)
        echo "Usage: $0 [gyroscope|stability|verify|all]"
        exit 1
        ;;
esac
