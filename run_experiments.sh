#!/bin/bash
# =============================================================================
# Comparative Study Experimental Pipeline
# =============================================================================
# This script runs the complete "Kill-Shot" benchmark experiments comparing:
#   - Standard GPT (model.py)
#   - Deep Delta Learning (proposed_model_ddl.py) - arXiv:2601.00417
#   - DeepSeek mHC (proposed_model_mhc_real.py) - arXiv:2512.24880
#   - E∆-MHC-Geo (proposed_model_hybrid.py) - Proposed
#
# Usage:
#   ./run_experiments.sh              # Run all experiments
#   ./run_experiments.sh gyroscope    # Run only gyroscope experiments
#   ./run_experiments.sh correction   # Run only correction experiments
#   ./run_experiments.sh stability    # Run only stability experiments
# =============================================================================

set -e  # Exit on error

DATASET=${1:-"all"}
MAX_ITERS=${2:-2000}

echo "=============================================================="
echo "E∆-MHC-Geo COMPARATIVE STUDY"
echo "=============================================================="
echo "Dataset: $DATASET"
echo "Max iterations: $MAX_ITERS"
echo ""
echo "Models:"
echo "  - gpt2: Standard GPT (model.py)"
echo "  - ddl: Deep Delta Learning (arXiv:2601.00417)"
echo "  - mhc: DeepSeek mHC (arXiv:2512.24880)"
echo "  - edelta: E∆-MHC-Geo (Proposed)"
echo ""

# Common parameters (from comparative_study.py)
COMMON_ARGS="--n_layer 6 --n_head 4 --n_embd 128 --n_streams 4 \
             --dropout 0.0 --learning_rate 1e-3 --max_iters $MAX_ITERS \
             --batch_size 64 --eval_interval 100 --log_interval 50 \
             --gate_reg_weight 0.01"

# =============================================================================
# Step 1: Generate Datasets (if not exist)
# =============================================================================
echo "=============================================================="
echo "Step 1: Generating Datasets"
echo "=============================================================="
python generate_all_datasets.py --skip_existing

# =============================================================================
# Step 2: Train Models
# =============================================================================
train_on_dataset() {
    local dataset=$1
    echo ""
    echo "=============================================================="
    echo "Training on $dataset dataset"
    echo "=============================================================="
    
    # Standard GPT (Baseline)
    echo ""
    echo ">>> [1/4] Training Standard GPT (Baseline)..."
    python train_continuous.py $COMMON_ARGS \
        --model_type gpt2 \
        --dataset $dataset \
        --out_dir out-$dataset-baseline \
        --always_save_checkpoint
    
    # DDL (Deep Delta Learning)
    echo ""
    echo ">>> [2/4] Training DDL (arXiv:2601.00417)..."
    python train_continuous.py $COMMON_ARGS \
        --model_type ddl \
        --dataset $dataset \
        --out_dir out-$dataset-ddl \
        --always_save_checkpoint
    
    # mHC (DeepSeek)
    echo ""
    echo ">>> [3/4] Training mHC (arXiv:2512.24880)..."
    python train_continuous.py $COMMON_ARGS \
        --model_type mhc \
        --dataset $dataset \
        --out_dir out-$dataset-mhc \
        --always_save_checkpoint
    
    # E∆-MHC-Geo (Proposed)
    echo ""
    echo ">>> [4/4] Training E∆-MHC-Geo (Proposed)..."
    python train_continuous.py $COMMON_ARGS \
        --model_type edelta \
        --dataset $dataset \
        --out_dir out-$dataset-proposed \
        --always_save_checkpoint
}

# =============================================================================
# Step 3: Run Training
# =============================================================================
if [ "$DATASET" == "all" ] || [ "$DATASET" == "gyroscope" ]; then
    train_on_dataset "gyroscope"
fi

if [ "$DATASET" == "all" ] || [ "$DATASET" == "correction" ]; then
    train_on_dataset "correction"
fi

if [ "$DATASET" == "all" ] || [ "$DATASET" == "stability" ]; then
    train_on_dataset "stability"
fi

# =============================================================================
# Step 4: Evaluate and Generate Plots
# =============================================================================
echo ""
echo "=============================================================="
echo "Step 4: Evaluation and Plotting"
echo "=============================================================="

evaluate_dataset() {
    local dataset=$1
    local extra_args=$2
    echo ""
    echo ">>> Evaluating $dataset..."
    python evaluate_models.py \
        --dataset $dataset \
        --model_dirs out-$dataset-baseline out-$dataset-ddl out-$dataset-mhc out-$dataset-proposed \
        --model_types gpt2 ddl mhc edelta \
        --plot $extra_args
}

if [ "$DATASET" == "all" ] || [ "$DATASET" == "gyroscope" ]; then
    evaluate_dataset "gyroscope" ""
fi

if [ "$DATASET" == "all" ] || [ "$DATASET" == "correction" ]; then
    evaluate_dataset "correction" ""
fi

if [ "$DATASET" == "all" ] || [ "$DATASET" == "stability" ]; then
    evaluate_dataset "stability" "--long_horizon --n_steps 10000"
fi

# =============================================================================
# Step 5: Summary
# =============================================================================
echo ""
echo "=============================================================="
echo "EXPERIMENT COMPLETE"
echo "=============================================================="
echo ""
echo "Results saved to:"
if [ "$DATASET" == "all" ] || [ "$DATASET" == "gyroscope" ]; then
    echo "  - gyroscope_results.npy"
    echo "  - gyroscope_comparison.png"
fi
if [ "$DATASET" == "all" ] || [ "$DATASET" == "correction" ]; then
    echo "  - correction_results.npy"
    echo "  - correction_comparison.png"
fi
if [ "$DATASET" == "all" ] || [ "$DATASET" == "stability" ]; then
    echo "  - stability_results.npy"
    echo "  - stability_comparison.png"
fi
echo ""
echo "Training logs in:"
echo "  - out-*/train_log.npy"
echo "  - out-*/results.npy"
echo ""
echo "=============================================================="
echo "EXPECTED RESULTS (Hypotheses)"
echo "=============================================================="
echo ""
echo "Gyroscope (Rotation Prediction):"
echo "  - Standard GPT: Loss plateaus ~0.1 (linear approximation)"
echo "  - DDL: Unstable for large angles (θ > 0.5), may diverge"
echo "  - mHC: Higher loss (spectral dampening)"
echo "  - E∆-MHC-Geo: Near-zero loss (Cayley isometry)"
echo ""
echo "Correction (Negation):"
echo "  - Standard GPT: Struggles with x → -x"
echo "  - DDL: Can negate but unstable path"
echo "  - mHC: Cannot flip instantly (no eigenvalue -1)"
echo "  - E∆-MHC-Geo: Instant flip via Householder (γ→0)"
echo ""
echo "Stability (Norm Preservation):"
echo "  - Standard GPT: Norm drifts after ~500 steps"
echo "  - DDL: Norm drifts (β ≠ 2)"
echo "  - mHC: Spectral collapse (streams converge)"
echo "  - E∆-MHC-Geo: ||v|| = 1 for 10,000+ steps"
echo ""
echo "=============================================================="
