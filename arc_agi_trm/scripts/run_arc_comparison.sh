#!/bin/bash
# Run E∆ vs JPmHC comparison on ARC-AGI-1
#
# Both use the same TRM backbone (7M params), same data, same optimizer.
# Only the mixer module differs.
#
# Prerequisites:
#   pip install -r requirements.txt
#   python -m dataset.build_arc_dataset \
#     --input-file-prefix kaggle/combined/arc-agi \
#     --output-dir data/arc1concept-aug-1000 \
#     --subsets training evaluation concept \
#     --test-set-name evaluation
#
# Usage:
#   bash scripts/run_arc_comparison.sh baseline   # Standard TRM (no mixer)
#   bash scripts/run_arc_comparison.sh jpmhc      # TRM + JPmHC Cayley mixer
#   bash scripts/run_arc_comparison.sh edelta      # TRM + E∆ (Cayley+Householder+gate)
#   bash scripts/run_arc_comparison.sh all         # Run all three

set -e

run_variant() {
    local MIXER=$1
    local RUN_NAME="arc1_${MIXER}"

    echo "============================================================"
    echo "ARC-AGI-1: mixer_type=${MIXER}"
    echo "============================================================"

    # Use torchrun for multi-GPU if available, else single GPU
    if [ "${NUM_GPUS:-1}" -gt 1 ]; then
        CMD="torchrun --nproc-per-node ${NUM_GPUS} --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --nnodes=1"
    else
        CMD="python"
    fi

    $CMD pretrain.py \
        arch=trm \
        data_paths="[data/arc1concept-aug-1000]" \
        arch.L_layers=2 \
        arch.H_cycles=3 arch.L_cycles=6 \
        arch.mixer_type=${MIXER} \
        arch.n_streams=4 \
        +run_name=${RUN_NAME} \
        ema=True
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
        echo "All three variants complete."
        echo "Compare results in wandb or output logs."
        ;;
    *)
        echo "Usage: $0 [baseline|jpmhc|edelta|all]"
        exit 1
        ;;
esac
