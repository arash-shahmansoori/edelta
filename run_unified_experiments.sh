#!/bin/bash
# ============================================================
# Unified Comparison Experiment Runner
# ============================================================
# Runs ALL models on ALL datasets with IDENTICAL hyperparameters
# for fair scientific comparison.
#
# Usage:
#   chmod +x run_unified_experiments.sh
#   ./run_unified_experiments.sh
#
# Or run specific dataset:
#   ./run_unified_experiments.sh rotation3d
# ============================================================

set -e  # Exit on error

# Configuration
CONFIG="config/train_unified_comparison.py"
MAX_ITERS=10000
COMPILE="True"

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Datasets to test (override with argument)
if [ -n "$1" ]; then
    DATASETS=("$1")
else
    DATASETS=(
        "rotation2d"
        "rotation3d"
        "deep_signal"
        "correction"
        "strategy_shift"
        "entropy_probe"
    )
fi

# Models to compare
declare -A MODELS
MODELS["baseline"]="train.py"
MODELS["edelta"]="train_geodesic.py"
MODELS["mhc_real"]="train_mhc_real.py"
MODELS["ddl"]="train_ddl.py"

echo "============================================================"
echo "UNIFIED COMPARISON EXPERIMENTS"
echo "============================================================"
echo "Config: $CONFIG"
echo "Max iterations: $MAX_ITERS"
echo "Datasets: ${DATASETS[*]}"
echo "Models: ${!MODELS[*]}"
echo "============================================================"
echo ""

# Results file
RESULTS_FILE="unified_results_$(date +%Y%m%d_%H%M%S).txt"
echo "Results will be saved to: $RESULTS_FILE"
echo ""

echo "UNIFIED COMPARISON RESULTS" > $RESULTS_FILE
echo "Date: $(date)" >> $RESULTS_FILE
echo "Config: $CONFIG" >> $RESULTS_FILE
echo "" >> $RESULTS_FILE

# Run experiments
for DATASET in "${DATASETS[@]}"; do
    echo -e "${BLUE}============================================================${NC}"
    echo -e "${BLUE}DATASET: $DATASET${NC}"
    echo -e "${BLUE}============================================================${NC}"
    
    echo "" >> $RESULTS_FILE
    echo "=== DATASET: $DATASET ===" >> $RESULTS_FILE
    
    for MODEL_NAME in "${!MODELS[@]}"; do
        SCRIPT="${MODELS[$MODEL_NAME]}"
        OUT_DIR="out-unified-${MODEL_NAME}-${DATASET}"
        
        echo ""
        echo -e "${GREEN}Running: $MODEL_NAME on $DATASET${NC}"
        echo "  Script: $SCRIPT"
        echo "  Output: $OUT_DIR"
        
        # Run training
        python $SCRIPT $CONFIG \
            --dataset=$DATASET \
            --out_dir=$OUT_DIR \
            --max_iters=$MAX_ITERS \
            --compile=$COMPILE \
            2>&1 | tee "${OUT_DIR}.log"
        
        # Extract final val loss
        FINAL_VAL_LOSS=$(grep "val loss" "${OUT_DIR}.log" | tail -1 | awk '{print $NF}')
        
        echo "  Final val loss: $FINAL_VAL_LOSS"
        echo "$MODEL_NAME: $FINAL_VAL_LOSS" >> $RESULTS_FILE
    done
done

echo ""
echo "============================================================"
echo "ALL EXPERIMENTS COMPLETE"
echo "============================================================"
echo "Results saved to: $RESULTS_FILE"
echo ""
cat $RESULTS_FILE
