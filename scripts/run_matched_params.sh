#!/bin/bash
# Run experiments with matched parameter counts (~1.79M for all models)
#
# Parameter counts with --match_proposed_params:
#   E∆-MHC-Geo: n_layer=6, n_embd=128  → 1.788M (reference)
#   GPT:        n_layer=9, n_embd=128  → 1.780M (0.996x)
#   DDL:        n_layer=8, n_embd=128  → 1.784M (0.998x)
#   mHC:        n_layer=9, n_embd=128  → 1.838M (1.028x)
#   JPmHC:      n_layer=7, n_embd=512  → 1.771M (0.991x)
#
# JPmHC uses n_embd=512 because its sub-layer F operates at d_stream=128
# (per arXiv:2602.18308 Section 3.2), requiring wider embedding.
#
# Usage:
#   bash scripts/run_matched_params.sh              # All models, seed 42
#   bash scripts/run_matched_params.sh --seeds 3    # All models, 3 seeds (42,123,456)

set -e

SEEDS="42"
if [ "$1" = "--seeds" ] && [ "$2" = "3" ]; then
    SEEDS="42 123 456"
    echo "Running with 3 seeds: $SEEDS"
fi

echo "============================================================"
echo "MATCHED PARAMETER EXPERIMENTS (~1.79M params for all models)"
echo "============================================================"

# Verify datasets
for dataset in gyroscope stability near_pi_rotation near_pi_rotation_multiplane; do
    if [ ! -f "data/${dataset}/train_x.npy" ]; then
        echo "Missing: data/${dataset}/train_x.npy"
        echo "Run: bash scripts/prepare_data.sh"
        exit 1
    fi
done
echo "All datasets OK."

mkdir -p out-matched
MAX_ITERS=2000

for SEED in $SEEDS; do
    SUFFIX=""
    [ "$SEED" != "42" ] && SUFFIX="-s${SEED}"

    for DATASET in gyroscope stability near_pi_rotation near_pi_rotation_multiplane; do
        echo ""
        echo "=== ${DATASET} (seed=${SEED}) ==="

        for MODEL in gpt2 ddl mhc jpmhc edelta; do
            OUTDIR="out-matched/${DATASET}-${MODEL}${SUFFIX}"
            echo "  Training ${MODEL}..."
            uv run src/training/train_continuous.py \
                --model_type $MODEL \
                --dataset $DATASET \
                --out_dir $OUTDIR \
                --max_iters $MAX_ITERS \
                --seed $SEED \
                --match_proposed_params 2>&1 | tail -1
        done
    done
done

echo ""
echo "============================================================"
echo "ALL EXPERIMENTS COMPLETE — results in out-matched/"
echo "Generate figures: uv run src/visualization/visualize_journal.py"
echo "============================================================"
