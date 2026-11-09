#!/bin/bash

# ========== GQA eval (Cross Attention checkpoint) (multi-GPU) ==========

# GPUs (edit gpu_list as needed)
gpu_list="0,1,2,3,4,5,6,7"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

# Current SFT checkpoint
CKPT="llava-v1.5-7b-sft-glpse-linear-canchaloss-3500"
MODEL_PATH="./checkpoint_glimpse-linear-cancha-loss/llava-v1.5-7b-sft-glimpse-linear-cancha-loss/checkpoint-3500/"
SPLIT="llava_gqa_testdev_balanced"
GQADIR="./playground/data/eval/gqa/data"

echo "=========================================="
echo "GQA eval - ${CHUNKS} GPUs"
echo "=========================================="
echo "Model path: $MODEL_PATH"
echo "=========================================="

# Create output directory
mkdir -p ./playground/data/eval/gqa/answers/$SPLIT/$CKPT

# Step 1: parallel eval
echo "Starting ${CHUNKS} eval workers (GPUs: ${gpu_list})..."

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_vqa_loader \
        --model-path $MODEL_PATH \
        --question-file ./playground/data/eval/gqa/$SPLIT.jsonl \
        --image-folder ./playground/data/eval/gqa/images \
        --answers-file ./playground/data/eval/gqa/answers/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --conv-mode vicuna_v1 &
done

wait
echo "All GPU eval workers finished."

# Step 2: merge answers
output_file=./playground/data/eval/gqa/answers/$SPLIT/$CKPT/merge.jsonl

echo "Merging answer shards..."
# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ./playground/data/eval/gqa/answers/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

echo "Merged answers: $(wc -l < $output_file) lines"

# Step 3: convert to the official GQA prediction JSON
# convert_gqa_for_eval.py maps LLaVA jsonl -> GQA eval format
# Input:  merge.jsonl (LLaVA: {"question_id": "...", "text": "..."})
# Output: testdev_balanced_predictions.json (GQA: [{"questionId": "...", "prediction": "..."}])
echo "Converting answers to GQA format..."
python scripts/convert_gqa_for_eval.py --src $output_file --dst $GQADIR/testdev_balanced_predictions.json

# Step 4: official GQA eval.py
# Metrics include accuracy, validity, plausibility, consistency, etc.
# Place GQA release assets under playground/data/eval/gqa/:
#   - testdev_balanced_questions.json (required)
#   - testdev_balanced_sceneGraphs.json, testdev_balanced_choices.json (optional; validity/plausibility)
# Download: https://cs.stanford.edu/people/dorarad/gqa/download.html
echo ""
echo "=========================================="
echo "Running official GQA eval.py ..."
echo "=========================================="
cd "$GQADIR"
# Questions file: {tier}_questions.json -> testdev_balanced_questions.json
# (do not pass a path containing *_balanced_questions*, or names stack as testdev_balanced_balanced_questions.json).
# Predictions default: {tier}_predictions.json -> testdev_balanced_predictions.json (in this data/ directory)
python eval.py --tier testdev_balanced \
    --questions "../testdev_balanced_questions.json" \
    --scenes "../testdev_balanced_sceneGraphs.json" \
    --choices "../testdev_balanced_choices.json" \
    --predictions "testdev_balanced_predictions.json"

echo ""
echo "=========================================="
echo "Done."
echo "=========================================="
echo "Answers: $output_file"
echo "GQA-format predictions: $GQADIR/testdev_balanced_predictions.json"
echo "=========================================="

