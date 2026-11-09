#!/bin/bash

# ========== POPE eval - pretrained (PT) model ==========

MODEL_BASE="./llava-v1.5-7b"
MODEL_PATH="./checkpoint_newrouter/llava-v1.5-7b-pretrain/checkpoint-2100"

ANSWERS_FILE="./playground/data/eval/pope/answers/llava-v1.5-7b-newrouter-PT2100.jsonl"

echo "=========================================="
echo "POPE eval - PT model"
echo "=========================================="
echo "Base model: $MODEL_BASE"
echo "PT weights: $MODEL_PATH"
echo "=========================================="

# Step 1: predictions
python -m llava.eval.model_vqa_loader \
    --model-path $MODEL_PATH \
    --model-base $MODEL_BASE \
    --question-file ./playground/data/eval/pope/llava_pope_test.jsonl \
    --image-folder ./playground/data/eval/pope/val2014 \
    --answers-file $ANSWERS_FILE \
    --temperature 0 \
    --conv-mode vicuna_v1

# Step 2: scoring
python llava/eval/eval_pope.py \
    --annotation-dir ./playground/data/eval/pope/coco \
    --question-file ./playground/data/eval/pope/llava_pope_test.jsonl \
    --result-file $ANSWERS_FILE

echo ""
echo "=========================================="
echo "Done."
echo "=========================================="

