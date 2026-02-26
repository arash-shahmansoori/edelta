#!/bin/bash
# Run ablation studies for the paper
#
# 1. Regularization weight ablation (Table tab:ablation_reg)
#    Tests λ ∈ {0.0, 0.05, 0.1, 0.2} on gyroscope
#
# 2. Computational efficiency profiling (Table tab:efficiency)
#    Measures forward time, memory, throughput for all models
#
# Usage:
#   bash scripts/run_ablation.sh              # All ablations
#   bash scripts/run_ablation.sh reg          # Regularization only
#   bash scripts/run_ablation.sh efficiency   # Efficiency profiling only

set -e

run_reg_ablation() {
    echo "============================================================"
    echo "REGULARIZATION WEIGHT ABLATION (E∆ on Gyroscope)"
    echo "============================================================"

    if [ ! -f "data/gyroscope/train_x.npy" ]; then
        echo "Missing gyroscope data. Run: bash scripts/prepare_data.sh gyroscope"
        exit 1
    fi

    for LAMBDA in 0.0 0.05 0.1 0.2; do
        echo ""
        echo "=== gate_reg_weight=${LAMBDA} ==="
        uv run src/training/train_continuous.py \
            --model_type edelta \
            --dataset gyroscope \
            --out_dir out-ablation/gyro-reg-${LAMBDA} \
            --gate_reg_weight $LAMBDA \
            --max_iters 2000 \
            --seed 42 2>&1 | tail -3
    done

    echo ""
    echo "Results in out-ablation/gyro-reg-*"
}

run_efficiency() {
    echo "============================================================"
    echo "COMPUTATIONAL EFFICIENCY PROFILING"
    echo "============================================================"

    if [ ! -f "data/gyroscope/train_x.npy" ]; then
        echo "Missing gyroscope data. Run: bash scripts/prepare_data.sh gyroscope"
        exit 1
    fi

    echo "Profiling forward pass time, memory, and throughput..."
    echo "(Runs 100 forward passes per model on gyroscope data)"

    for MODEL in gpt2 ddl mhc jpmhc edelta; do
        echo ""
        echo "=== ${MODEL} ==="
        uv run src/training/train_continuous.py \
            --model_type $MODEL \
            --dataset gyroscope \
            --out_dir out-ablation/efficiency-${MODEL} \
            --max_iters 10 \
            --match_proposed_params \
            --seed 42 2>&1 | grep -E "parameters|time|ms"
    done
}

case "${1:-all}" in
    reg)        run_reg_ablation ;;
    efficiency) run_efficiency ;;
    all)
        run_reg_ablation
        echo ""
        run_efficiency
        echo ""
        echo "All ablations complete."
        ;;
    *)
        echo "Usage: $0 [reg|efficiency|all]"
        exit 1
        ;;
esac
