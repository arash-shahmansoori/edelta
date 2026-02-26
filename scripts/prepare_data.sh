#!/bin/bash
# Prepare all datasets for E∆-MHC-Geo experiments
#
# Usage:
#   bash scripts/prepare_data.sh                    # All datasets
#   bash scripts/prepare_data.sh gyroscope          # Only gyroscope
#   bash scripts/prepare_data.sh stability          # Only stability
#   bash scripts/prepare_data.sh near_pi            # Only near-π (single + multi)
#   bash scripts/prepare_data.sh verify             # Check all exist

set -e

echo "============================================================"
echo "DATA PREPARATION FOR E∆-MHC-GEO EXPERIMENTS"
echo "============================================================"

prepare_gyroscope() {
    echo ""
    echo "=== Preparing Gyroscope Dataset ==="
    echo "Task: Continuous rotation prediction (d=16, seq=256, 9000 train)"
    uv run src/data/gyroscope.py \
        --output_dir data/gyroscope \
        --dim 16 --seq_len 256 --n_train 9000 --n_val 1000 --seed 42
    echo "Done: data/gyroscope/"
}

prepare_stability() {
    echo ""
    echo "=== Preparing Stability Dataset ==="
    echo "Task: Long-horizon isometry (d=64, seq=128, 900 train)"
    uv run src/data/stability.py \
        --output_dir data/stability \
        --dim 64 --n_train 900 --n_val 100 --train_seq_len 128 --seed 42
    echo "Done: data/stability/"
}

prepare_near_pi() {
    echo ""
    echo "=== Preparing Near-π Rotation Datasets ==="
    echo "Single-plane (θ=177.6°, d=64, seq=128)"
    uv run src/data/near_pi_rotation.py \
        --output_dir data/near_pi_rotation \
        --rotation_mode single_plane --seed 42
    echo "Multi-plane (θ=179.9°, d=64, seq=128)"
    uv run src/data/near_pi_rotation.py \
        --output_dir data/near_pi_rotation_multiplane \
        --rotation_mode multi_plane --seed 42
    echo "Done: data/near_pi_rotation/ and data/near_pi_rotation_multiplane/"
}

verify_datasets() {
    echo ""
    echo "=== Verifying Datasets ==="
    missing=0
    for dataset in gyroscope stability near_pi_rotation near_pi_rotation_multiplane; do
        if [ -f "data/${dataset}/train_x.npy" ] && [ -f "data/${dataset}/val_x.npy" ]; then
            echo "  ✓ ${dataset}"
        else
            echo "  ✗ ${dataset}: MISSING"
            missing=1
        fi
    done
    return $missing
}

case "${1:-all}" in
    gyroscope)    prepare_gyroscope ;;
    stability)    prepare_stability ;;
    near_pi)      prepare_near_pi ;;
    verify)       verify_datasets ;;
    all)
        prepare_gyroscope
        prepare_stability
        prepare_near_pi
        echo ""
        echo "============================================================"
        echo "ALL DATASETS PREPARED"
        echo "============================================================"
        verify_datasets
        ;;
    *)
        echo "Usage: $0 [gyroscope|stability|near_pi|verify|all]"
        exit 1
        ;;
esac
