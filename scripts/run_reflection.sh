#!/bin/bash
# Run reflection experiments: Direct test of geometric operators
#
# This experiment tests the CORE CLAIM of geometric models:
# - DDL: β should converge to 2 for exact Householder reflection
# - E∆-MHC-Geo: γ should converge to 0 to use Householder component
#
# Task: Learn y = -x (pure negation/reflection)
#
# Following "The Illusion of Insight in Reasoning Models" (arXiv:2601.00514v1):
# - Track parameter trajectories (β for DDL, γ for E∆-MHC-Geo)
# - Measure accuracy conditional on parameter convergence
# - Analyze "Aha!" moments (parameter shifts that improve performance)
#
# NOTE: GPT and mHC are excluded - they use MLP approximation rather than
# geometric operators, which would be "cheating" in this test.
#
# Usage:
#   bash scripts/run_reflection.sh                    # Full sample efficiency test
#   bash scripts/run_reflection.sh trajectory         # Parameter trajectory analysis
#   bash scripts/run_reflection.sh single             # Quick single test

set -e

MODE="${1:-sample_efficiency}"

echo "============================================================"
echo "REFLECTION EXPERIMENT: Geometric Operator Analysis"
echo "============================================================"
echo "Task: Learn y = -x (pure negation/reflection)"
echo "Mode: ${MODE}"
echo ""
echo "Models:"
echo "  - DDL: β should converge to 2.0"
echo "  - E∆-MHC-Geo: γ should converge to 0.0"
echo ""
echo "Following arXiv:2601.00514v1 methodology"
echo "============================================================"

# Create output directory
mkdir -p results

case "${MODE}" in
    sample_efficiency)
        echo ""
        echo "Running SAMPLE EFFICIENCY test..."
        echo "Tests: [10, 25, 50, 100, 200, 500] training samples"
        echo ""
        
        uv run src/training/train_reflection.py \
            --mode sample_efficiency \
            --dim 64 \
            --max_iters 2000 \
            --save_figures \
            --output_dir results
        ;;
        
    trajectory)
        echo ""
        echo "Running PARAMETER TRAJECTORY analysis..."
        echo "Tracks β and γ evolution during training"
        echo ""
        
        uv run src/training/train_reflection.py \
            --mode trajectory \
            --dim 64 \
            --max_iters 2000 \
            --n_samples 500 \
            --save_figures \
            --output_dir results
        ;;
        
    single)
        echo ""
        echo "Running QUICK SINGLE test..."
        echo "Fast sanity check with 100 samples"
        echo ""
        
        uv run src/training/train_reflection.py \
            --mode single \
            --dim 64 \
            --max_iters 1000 \
            --n_samples 100 \
            --output_dir results
        ;;
        
    *)
        echo "Usage: $0 [sample_efficiency|trajectory|single]"
        echo ""
        echo "Modes:"
        echo "  sample_efficiency  Full test across multiple sample sizes (default)"
        echo "  trajectory         Parameter trajectory analysis with detailed tracking"
        echo "  single             Quick sanity check"
        exit 1
        ;;
esac

echo ""
echo "============================================================"
echo "REFLECTION EXPERIMENT COMPLETE"
echo "============================================================"
echo ""
echo "Results saved to results/"
echo ""
echo "Generated figure:"
echo "  - reflection_aha_moment.png        (4-panel 'Aha!' moment visualization)"
