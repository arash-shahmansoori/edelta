#!/bin/bash
# Run reflection experiments: test geometric operators on y = -x
#
# Models tested: DDL, JPmHC, E∆-MHC-Geo
# (GPT and mHC excluded — they use MLP approximation, not geometric operators)
#
# Following "Illusion of Insight" (arXiv:2601.00514) methodology:
# - DDL: β should converge to 2.0 (exact Householder)
# - JPmHC: should FAIL (SO(n)-only, no eigenvalue -1)
# - E∆-MHC-Geo: γ should converge to 0.0 (select Householder)
#
# Usage:
#   bash scripts/run_reflection.sh                    # Sample efficiency test, seed 42
#   bash scripts/run_reflection.sh trajectory         # Parameter trajectory analysis
#   bash scripts/run_reflection.sh single             # Quick single test
#   bash scripts/run_reflection.sh --seeds 3          # Multi-seed sample efficiency

set -e

MODE="sample_efficiency"
SEED=42
EXTRA_ARGS=""

# Parse arguments
while [ $# -gt 0 ]; do
    case "$1" in
        trajectory|single|sample_efficiency)
            MODE="$1"; shift ;;
        --seeds)
            shift
            if [ "$1" = "3" ]; then
                echo "Running 3 seeds: 42, 123, 456"
                for s in 42 123 456; do
                    echo ""
                    echo "========== SEED $s =========="
                    uv run src/training/train_reflection.py \
                        --mode sample_efficiency --dim 64 --max_iters 2000 \
                        --save_figures --output_dir results --seed $s
                done
                echo ""
                echo "All 3 seeds complete. Figures: results/"
                exit 0
            fi
            shift ;;
        *) shift ;;
    esac
done

echo "============================================================"
echo "REFLECTION EXPERIMENT: Geometric Operator Analysis"
echo "============================================================"
echo "Task: Learn y = -x (pure negation)"
echo "Mode: ${MODE}"
echo "Models: DDL (β→2), JPmHC (should fail), E∆ (γ→0)"
echo "============================================================"

mkdir -p results

case "${MODE}" in
    sample_efficiency)
        uv run src/training/train_reflection.py \
            --mode sample_efficiency --dim 64 --max_iters 2000 \
            --save_figures --output_dir results --seed $SEED
        ;;
    trajectory)
        uv run src/training/train_reflection.py \
            --mode trajectory --dim 64 --max_iters 2000 --n_samples 500 \
            --save_figures --output_dir results --seed $SEED
        ;;
    single)
        uv run src/training/train_reflection.py \
            --mode single --dim 64 --max_iters 1000 --n_samples 100 \
            --output_dir results --seed $SEED
        ;;
    *)
        echo "Usage: $0 [sample_efficiency|trajectory|single] [--seeds 3]"
        exit 1
        ;;
esac

echo ""
echo "============================================================"
echo "REFLECTION EXPERIMENT COMPLETE — results in results/"
echo "============================================================"
