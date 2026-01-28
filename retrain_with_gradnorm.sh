#!/bin/bash
# Re-run training for all models with gradient norm logging
# This will capture gradient norms for publication-quality figures

set -e

echo "=============================================="
echo "Re-training all models with gradient norm logging"
echo "=============================================="

# Output directories
BASE_DIR="out-gradnorm"

# Model types
MODELS=("gpt2" "ddl" "mhc" "edelta")
MODEL_NAMES=("baseline" "ddl" "mhc" "proposed")

# Datasets
DATASETS=("gyroscope" "correction" "stability")

# Fair Fight hyperparameters
N_LAYER=6
N_EMBD=128
N_HEAD=4
N_STREAMS=4
DROPOUT=0.0
BATCH_SIZE=64
LR=1e-3
MAX_ITERS=2000
WEIGHT_DECAY=0.1
GRAD_CLIP=1.0
EVAL_INTERVAL=100
LOG_INTERVAL=50

mkdir -p $BASE_DIR

# Run training for each combination
for i in "${!MODELS[@]}"; do
    MODEL=${MODELS[$i]}
    MODEL_NAME=${MODEL_NAMES[$i]}
    
    for DATASET in "${DATASETS[@]}"; do
        OUT_DIR="${BASE_DIR}/${DATASET}-${MODEL_NAME}"
        
        echo ""
        echo "=============================================="
        echo "Training: $MODEL on $DATASET"
        echo "Output: $OUT_DIR"
        echo "=============================================="
        
        uv run python train_continuous.py \
            --model_type=$MODEL \
            --dataset=$DATASET \
            --out_dir=$OUT_DIR \
            --n_layer=$N_LAYER \
            --n_embd=$N_EMBD \
            --n_head=$N_HEAD \
            --n_streams=$N_STREAMS \
            --dropout=$DROPOUT \
            --batch_size=$BATCH_SIZE \
            --learning_rate=$LR \
            --max_iters=$MAX_ITERS \
            --weight_decay=$WEIGHT_DECAY \
            --grad_clip=$GRAD_CLIP \
            --eval_interval=$EVAL_INTERVAL \
            --log_interval=$LOG_INTERVAL \
            --gate_reg_weight=0.1 \
            --init_gate_bias=0.0
            
        echo "Completed: $MODEL on $DATASET"
    done
done

echo ""
echo "=============================================="
echo "All training complete!"
echo "Results saved to: $BASE_DIR"
echo "=============================================="
