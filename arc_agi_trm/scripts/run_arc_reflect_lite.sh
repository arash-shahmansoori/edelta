#!/bin/bash
# ARC-Reflect-Lite: low-cost focused comparison for reflection value
#
# Runs TRM with selected mixer variants on two curated slices:
# - reflect: reflection-sensitive tasks
# - control: rotation/regular tasks (non-reflection dominant)
#
# Required prepared datasets:
#   data/arc_reflect_lite_reflect
#   data/arc_reflect_lite_control
#
# Usage:
#   bash scripts/run_arc_reflect_lite.sh edelta
#   bash scripts/run_arc_reflect_lite.sh jpmhc
#   bash scripts/run_arc_reflect_lite.sh all
#
# Optional env overrides:
#   DATA_ROOT=data
#   EPOCHS=80
#   EVAL_INTERVAL=10
#   GLOBAL_BATCH_SIZE=128
#   LR=1e-4
#   NUM_GPUS=1

set -e

DATA_ROOT="${DATA_ROOT:-data}"
EPOCHS="${EPOCHS:-80}"
EVAL_INTERVAL="${EVAL_INTERVAL:-10}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-128}"
LR="${LR:-1e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1.0}"
PUZZLE_EMB_LR="${PUZZLE_EMB_LR:-1e-4}"
PUZZLE_EMB_WD="${PUZZLE_EMB_WD:-1.0}"
HALT_MAX_STEPS="${HALT_MAX_STEPS:-2}"
N_STREAMS="${N_STREAMS:-4}"

run_variant_slice() {
    local MIXER=$1
    local SLICE=$2
    local DATA_PATH="${DATA_ROOT}/arc_reflect_lite_${SLICE}"
    local RUN_NAME="reflect_lite_${SLICE}_${MIXER}"

    if [ ! -d "${DATA_PATH}" ]; then
        echo "Missing dataset directory: ${DATA_PATH}"
        echo "Build it first (see ARC_REFLECT_LITE_PROTOCOL.md)."
        exit 1
    fi

    echo "============================================================"
    echo "ARC-Reflect-Lite | slice=${SLICE} | mixer=${MIXER}"
    echo "============================================================"

    if [ "${NUM_GPUS:-1}" -gt 1 ]; then
        CMD="torchrun --nproc-per-node ${NUM_GPUS} --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --nnodes=1"
    else
        CMD="python"
    fi

    $CMD pretrain.py \
        arch=trm \
        data_paths="[${DATA_PATH}]" \
        arch.L_layers=2 \
        arch.H_cycles=3 \
        arch.L_cycles=6 \
        arch.halt_max_steps=${HALT_MAX_STEPS} \
        arch.mixer_type=${MIXER} \
        arch.n_streams=${N_STREAMS} \
        epochs=${EPOCHS} \
        eval_interval=${EVAL_INTERVAL} \
        lr=${LR} \
        puzzle_emb_lr=${PUZZLE_EMB_LR} \
        weight_decay=${WEIGHT_DECAY} \
        puzzle_emb_weight_decay=${PUZZLE_EMB_WD} \
        global_batch_size=${GLOBAL_BATCH_SIZE} \
        +run_name=${RUN_NAME} \
        ema=True
}

run_variant() {
    local MIXER=$1
    run_variant_slice "${MIXER}" "reflect"
    run_variant_slice "${MIXER}" "control"
}

case "${1:-all}" in
    baseline) run_variant "none" ;;
    jpmhc)    run_variant "jpmhc" ;;
    edelta)   run_variant "edelta" ;;
    all)
        run_variant "none"
        run_variant "jpmhc"
        run_variant "edelta"
        echo ""
        echo "All ARC-Reflect-Lite runs completed."
        ;;
    *)
        echo "Usage: $0 [baseline|jpmhc|edelta|all]"
        exit 1
        ;;
esac
