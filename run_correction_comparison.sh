#!/bin/bash
#
# Correction Task Comparison Benchmark
# 
# Based on "The Illusion of Insight in Reasoning Models" (arXiv:2601.00514v1)
#
# Key metrics from the paper:
# - %S: Shift prevalence (should be ~6.31% for spontaneous, higher for triggered)
# - P(✓|S=1): Accuracy with shift detected
# - P(✓|S=0): Accuracy without shift  
# - Δ(pp): Accuracy difference (shift effect)
# - Entropy-stratified performance (high uncertainty = correction helps more)
#
# Expected results:
# - GPT2 Baseline: Cannot detect correction signals, poor at instant flips
# - DDL: Incremental improvement, struggles with instant belief flips
# - mHC: May detect shifts but limited correction execution
# - E∆-MHC-Geo: Should excel via β-gating (detection) + Householder (flip execution)
#

set -e

echo "=============================================================="
echo "Correction Benchmark Suite"
echo "Based on: 'The Illusion of Insight' (arXiv:2601.00514v1)"
echo "=============================================================="

# Configuration
DEVICE=${DEVICE:-cuda}
OUT_DIR=${OUT_DIR:-out-correction-benchmark}
MAX_ITERS=${MAX_ITERS:-2000}
BATCH_SIZE=${BATCH_SIZE:-64}

mkdir -p $OUT_DIR

# Step 1: Generate datasets if needed
echo ""
echo "Step 1: Checking/Generating datasets..."
echo "--------------------------------------------------------------"

if [ ! -d "data/correction_insight" ]; then
    python data/correction_insight.py --dataset all
fi

if [ ! -f "data/correction_ultimate_rotation_reflection.npz" ]; then
    python data/correction_ultimate.py
fi

# Step 2: Run insight dataset comparison
echo ""
echo "Step 2: Running Insight Dataset Comparison"
echo "--------------------------------------------------------------"
echo "This tests intrinsic self-correction capability"
echo ""

python train_correction_insight.py \
    --compare \
    --dataset insight \
    --device $DEVICE \
    --out_dir $OUT_DIR/insight \
    --max_iters $MAX_ITERS \
    --batch_size $BATCH_SIZE

# Step 3: Run entropy-stratified comparison
echo ""
echo "Step 3: Running Entropy-Stratified Comparison"
echo "--------------------------------------------------------------"
echo "Following paper Table 26: high entropy (top 20%) vs low entropy (bottom 80%)"
echo ""

python train_correction_insight.py \
    --compare \
    --dataset entropy \
    --device $DEVICE \
    --out_dir $OUT_DIR/entropy \
    --max_iters $MAX_ITERS \
    --batch_size $BATCH_SIZE

# Step 4: Run shift detection comparison
echo ""
echo "Step 4: Running Shift Detection Comparison"
echo "--------------------------------------------------------------"
echo "Tests if model can detect AND execute corrections"
echo ""

python train_correction_insight.py \
    --compare \
    --dataset shift \
    --device $DEVICE \
    --out_dir $OUT_DIR/shift \
    --max_iters $MAX_ITERS \
    --batch_size $BATCH_SIZE

# Step 5: Run ultimate correction comparison
echo ""
echo "Step 5: Running Ultimate Correction Comparison (Rotation vs Reflection)"
echo "--------------------------------------------------------------"
echo "Direct test of Cayley (rotation, det=+1) vs Householder (reflection, det=-1)"
echo ""

python train_ultimate_correction.py \
    --compare \
    --task rotation_reflection \
    --device $DEVICE \
    --out_dir $OUT_DIR/ultimate \
    --max_iters $MAX_ITERS \
    --batch_size $BATCH_SIZE

# Step 6: Summary
echo ""
echo "=============================================================="
echo "BENCHMARK COMPLETE"
echo "=============================================================="
echo ""
echo "Results saved to: $OUT_DIR/"
echo ""
echo "Key findings to look for:"
echo "1. E∆-MHC-Geo should have highest correction quality"
echo "2. Baseline GPT2 should struggle with instant belief flips"
echo "3. Entropy-stratified: gains should be larger for high-uncertainty samples"
echo "4. Rotation vs Reflection: tests architectural capability directly"
echo ""
echo "To visualize results:"
echo "  python visualize_correction_results.py --results_dir $OUT_DIR"
echo ""
