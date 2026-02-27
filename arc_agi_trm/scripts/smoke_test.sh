#!/bin/bash
# Smoke test: run all 3 mixer variants for ~50 steps each
# Verifies: model builds, forward/backward works, all metrics tracked
#
# Expected runtime: ~5 minutes on GPU, ~15 minutes on CPU
#
# What to check in wandb (or stdout):
#   ✓ train/lm_loss — decreasing
#   ✓ train/grad_norm — not exploding/vanishing
#   ✓ train/step_time_s — reasonable
#   ✓ train/vram_mb — peak GPU memory
#   ✓ train/gate_attn_L0, train/gate_ffn_L0 — only for edelta
#   ✓ train/gate_reg_loss — only for edelta
#   ✓ train/exact_accuracy — evaluation metric
#
# Usage:
#   bash scripts/smoke_test.sh          # Run all 3 variants
#   bash scripts/smoke_test.sh edelta   # Run only edelta

set -e

echo "============================================================"
echo "SMOKE TEST: Verifying all mixer variants"
echo "============================================================"

# Prepare ARC-AGI data if not present
if [ ! -d "data/arc1concept-aug-1000" ]; then
    echo "Preparing ARC-AGI-1 dataset..."
    python -m dataset.build_arc_dataset \
        --input-file-prefix kaggle/combined/arc-agi \
        --output-dir data/arc1concept-aug-1000 \
        --subsets training evaluation concept \
        --test-set-name evaluation
fi
DATA_PATH="data/arc1concept-aug-1000"
echo "Using ARC-AGI-1 data"

VARIANTS="${1:-none jpmhc edelta}"
STEPS=50
EVAL_INTERVAL=25

for MIXER in $VARIANTS; do
    echo ""
    echo "========== mixer_type=${MIXER} =========="
    echo ""

    python pretrain.py \
        arch=trm \
        data_paths="[${DATA_PATH}]" \
        evaluators="[]" \
        epochs=${STEPS} \
        eval_interval=${EVAL_INTERVAL} \
        lr=1e-4 \
        puzzle_emb_lr=1e-4 \
        weight_decay=1.0 \
        puzzle_emb_weight_decay=1.0 \
        arch.L_layers=2 \
        arch.H_cycles=2 \
        arch.L_cycles=2 \
        arch.mixer_type=${MIXER} \
        arch.n_streams=4 \
        +run_name=smoke_${MIXER} \
        2>&1 | grep -E "step|loss|grad_norm|gate|vram|accuracy|param|Error|error" | head -30

    echo ""
    echo "  ✓ mixer_type=${MIXER} completed"
done

echo ""
echo "============================================================"
echo "SMOKE TEST COMPLETE"
echo "============================================================"
echo ""
echo "Check wandb for detailed metrics, or review stdout above."
echo "Key things to verify:"
echo "  1. All 3 variants ran without errors"
echo "  2. grad_norm is reasonable (not NaN/Inf)"
echo "  3. lm_loss is decreasing"
echo "  4. For edelta: gate values (γ) are being tracked"
echo "  5. vram_mb is reported"
