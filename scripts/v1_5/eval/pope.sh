#!/bin/bash

# ========== POPE SFT eval (8 GPUs) ==========

# Full SFT checkpoint
MODEL_PATH="./checkpoint_crossattetion/llava-v1.5-7b-sft/checkpoint-5100/"

# Output file
ANSWERS_DIR="./playground/data/eval/pope/answers"
FINAL_ANSWERS_FILE="${ANSWERS_DIR}/llava-v1.5-7b-new-SFT5100.jsonl"

# Number of GPUs
NUM_GPUS=8

echo "=========================================="
echo "POPE eval - ${NUM_GPUS} GPUs"
echo "=========================================="
echo "SFT model: $MODEL_PATH"
echo "=========================================="

mkdir -p $ANSWERS_DIR

# Remove stale chunk outputs
rm -f ${ANSWERS_DIR}/chunk_*.jsonl

# Step 1: parallel eval
echo "Starting ${NUM_GPUS} eval workers..."

for IDX in $(seq 0 $((NUM_GPUS-1))); do
    CUDA_VISIBLE_DEVICES=${IDX} python -m llava.eval.model_vqa_loader \
        --model-path $MODEL_PATH \
        --question-file ./playground/data/eval/pope/llava_pope_test.jsonl \
        --image-folder ./playground/data/eval/pope/val2014 \
        --answers-file ${ANSWERS_DIR}/chunk_${IDX}.jsonl \
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

# Step 3: POPE accuracy
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
echo "=========================================="
