#!/bin/bash
# Run near-π rotation experiments + initialization robustness
#
# Near-π tests: all 5 models on single-plane and multi-plane
# Init robustness: E∆ only, 3 gate biases × 2 datasets
#
# Usage:
#   bash scripts/run_near_pi.sh              # All experiments, seed 42
#   bash scripts/run_near_pi.sh --seeds 3    # 3 seeds (42,123,456)

set -e

SEEDS="42"
if [ "$1" = "--seeds" ] && [ "$2" = "3" ]; then
    SEEDS="42 123 456"
fi

echo "============================================================"
echo "NEAR-π ROTATION EXPERIMENTS"
echo "============================================================"

# Generate datasets if missing
for ds in near_pi_rotation near_pi_rotation_multiplane; do
    if [ ! -f "data/${ds}/train_x.npy" ]; then
        echo "Generating ${ds}..."
        if [ "$ds" = "near_pi_rotation" ]; then
            uv run src/data/near_pi_rotation.py --rotation_mode single_plane \
                --theta 3.10 --seed 42
        else
            uv run src/data/near_pi_rotation.py --rotation_mode multi_plane \
                --output_dir data/near_pi_rotation_multiplane --theta 3.14 --seed 42
        fi
    fi
done

MAX_ITERS=2000

# === Baseline comparison ===
for SEED in $SEEDS; do
    SUFFIX=""
    [ "$SEED" != "42" ] && SUFFIX="-s${SEED}"

    for DS in near_pi_rotation near_pi_rotation_multiplane; do
        echo "=== ${DS} (seed=${SEED}) ==="
        for MODEL in gpt2 ddl mhc jpmhc edelta; do
            echo "  ${MODEL}..."
            uv run src/training/train_continuous.py \
                --model_type $MODEL --dataset $DS \
                --out_dir out-matched/${DS}-${MODEL}${SUFFIX} \
                --max_iters $MAX_ITERS --seed $SEED \
                --match_proposed_params 2>&1 | tail -1
        done
    done
done

# === Init robustness (E∆ only) ===
echo ""
echo "=== INITIALIZATION ROBUSTNESS (E∆-MHC-Geo) ==="

for SEED in $SEEDS; do
    SUFFIX=""
    [ "$SEED" != "42" ] && SUFFIX="-s${SEED}"

    for DS in near_pi_rotation near_pi_rotation_multiplane; do
        TASK="single"
        [ "$DS" = "near_pi_rotation_multiplane" ] && TASK="multi"

        for BIAS in 1.5 0.0 -1.5; do
            if [ "$TASK" = "single" ]; then
                case "$BIAS" in
                    1.5)  TAG="bias-pos" ;;
                    0.0)  TAG="init" ;;
                    -1.5) TAG="bias-neg" ;;
                esac
                OUTDIR="out-near-pi-single-edelta-${TAG}${SUFFIX}"
            else
                OUTDIR="out-near-pi-multiplane-init-${BIAS}${SUFFIX}"
            fi

            echo "  ${TASK} bias=${BIAS} seed=${SEED}..."
            uv run src/training/train_continuous.py \
                --model_type edelta --dataset $DS \
                --out_dir $OUTDIR \
                --init_gate_bias $BIAS --gate_reg_weight 0.5 \
                --max_iters $MAX_ITERS --seed $SEED 2>&1 | tail -1
        done
    done
done

echo ""
echo "=== Generate figure ==="
uv run src/visualization/visualize_near_pi.py

echo "Done! Figure: results/near_pi_rotation_comparison.png"
