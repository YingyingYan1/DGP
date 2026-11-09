#!/bin/bash

# ========== POPE eval + layer-wise router weight statistics (8 GPUs) ==========

MODEL_BASE="./llava-v1.5-7b"
MODEL_PATH="./checkpoint_newrouter/llava-v1.5-7b-pretrain/checkpoint-2100"

# Output files
ANSWERS_DIR="./playground/data/eval/pope/answers"
ROUTER_STATS_DIR="./playground/data/eval/pope/router_stats"
FINAL_ANSWERS_FILE="${ANSWERS_DIR}/llava-v1.5-7b-newrouter-PT2100-with-stats.jsonl"
FINAL_ROUTER_STATS_FILE="${ROUTER_STATS_DIR}/llava-v1.5-7b-newrouter-PT2100-router-weights.json"

# Number of GPUs
NUM_GPUS=8

echo "=========================================="
echo "POPE eval - ${NUM_GPUS} GPUs + router weight statistics"
echo "=========================================="
echo "Base model: $MODEL_BASE"
echo "PT weights: $MODEL_PATH"
echo "=========================================="

mkdir -p $ANSWERS_DIR
mkdir -p $ROUTER_STATS_DIR

# Step 1: parallel eval
echo "Starting ${NUM_GPUS} eval workers..."

for IDX in $(seq 0 $((NUM_GPUS-1))); do
    CUDA_VISIBLE_DEVICES=${IDX} python -m llava.eval.model_vqa_loader_with_router_stats \
        --model-path $MODEL_PATH \
        --model-base $MODEL_BASE \
        --question-file ./playground/data/eval/pope/llava_pope_test.jsonl \
        --image-folder ./playground/data/eval/pope/val2014 \
        --answers-file ${ANSWERS_DIR}/chunk_${IDX}.jsonl \
        --router-stats-file ${ROUTER_STATS_DIR}/chunk_${IDX}_stats.json \
        --num-chunks $NUM_GPUS \
        --chunk-idx $IDX \
        --temperature 0 \
        --conv-mode vicuna_v1 &
done

wait
echo "All GPU eval workers finished."

# Step 2: merge answer shards
echo "Merging answer shards..."
> $FINAL_ANSWERS_FILE
for IDX in $(seq 0 $((NUM_GPUS-1))); do
    cat ${ANSWERS_DIR}/chunk_${IDX}.jsonl >> $FINAL_ANSWERS_FILE
done
echo "Merged answers: $(wc -l < $FINAL_ANSWERS_FILE) lines"

# Step 3: merge router weight statistics
echo "Merging router weight statistics..."
python -m llava.eval.merge_router_stats \
    --stats-dir $ROUTER_STATS_DIR \
    --num-chunks $NUM_GPUS \
    --output-file $FINAL_ROUTER_STATS_FILE

# Step 4: POPE accuracy
echo ""
echo "=========================================="
echo "Computing POPE accuracy..."
echo "=========================================="
python llava/eval/eval_pope.py \
    --annotation-dir ./playground/data/eval/pope/coco \
    --question-file ./playground/data/eval/pope/llava_pope_test.jsonl \
    --result-file $FINAL_ANSWERS_FILE

echo ""
echo "=========================================="
echo "Done."
echo "=========================================="
echo "Answers: $FINAL_ANSWERS_FILE"
echo "Layer-wise router weight statistics: $FINAL_ROUTER_STATS_FILE"
echo "=========================================="

