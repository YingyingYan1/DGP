#!/bin/bash

# ========== POPE SFT eval + layer-wise router weight statistics (8 GPUs) ==========
# Collect two router weight tensors:
#   1. Pre-Router: before LLM layer 0, layerwise vision fusion over 24 ViT layers -> [1, 24] weights
#   2. LLM Router: inside 32 LLM layers, one layer-wise router per layer -> [32, 24] weights
# Output JSON includes separate pre_router and llm_router router weight statistics

# Project root (absolute path; script can be run from any cwd)
PROJECT_ROOT="."
cd "${PROJECT_ROOT}" || exit 1

# Full SFT checkpoint (no --model-base needed)
MODEL_PATH="./checkpoint_glimpse-linear-cancha-loss/llava-v1.5-7b-sft-glimpse-linear-cancha-loss/checkpoint-3300/"

# Data and outputs (absolute paths)
QUESTION_FILE="${PROJECT_ROOT}/playground/data/eval/pope/llava_pope_test.jsonl"
IMAGE_FOLDER="${PROJECT_ROOT}/playground/data/eval/pope/val2014"
ANNOTATION_DIR="${PROJECT_ROOT}/playground/data/eval/pope/coco"
ANSWERS_DIR="${PROJECT_ROOT}/playground/data/eval/pope/answers"
ROUTER_STATS_DIR="${PROJECT_ROOT}/playground/data/eval/pope/router_stats"
FINAL_ANSWERS_FILE="${ANSWERS_DIR}/llava-v1.5-7b-gls-linear-cancahloss-3300-with-stats.jsonl"
FINAL_ROUTER_STATS_FILE="${ROUTER_STATS_DIR}/llava-v1.5-7b-gls-linear-cancahloss-3300-router-weights.json"

# GPUs: devices 0-7
gpu_list="0,1,2,3,4,5,6,7"
IFS=',' read -ra GPULIST <<< "$gpu_list"
NUM_GPUS=${#GPULIST[@]}

echo "=========================================="
echo "POPE eval - ${NUM_GPUS} GPUs + router weight statistics"
echo "=========================================="
echo "SFT model: $MODEL_PATH"
echo "GPUs: ${gpu_list}"
echo "=========================================="

mkdir -p "${ANSWERS_DIR}"
mkdir -p "${ROUTER_STATS_DIR}"

# Remove stale chunk outputs
rm -f "${ANSWERS_DIR}"/chunk_*.jsonl
rm -f "${ROUTER_STATS_DIR}"/chunk_*_stats.json

# Step 1: parallel eval
echo "Starting ${NUM_GPUS} eval workers..."

for IDX in $(seq 0 $((NUM_GPUS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_vqa_loader_with_router_stats \
        --model-path "${MODEL_PATH}" \
        --question-file "${QUESTION_FILE}" \
        --image-folder "${IMAGE_FOLDER}" \
        --answers-file "${ANSWERS_DIR}/chunk_${IDX}.jsonl" \
        --router-stats-file "${ROUTER_STATS_DIR}/chunk_${IDX}_stats.json" \
        --num-chunks $NUM_GPUS \
        --chunk-idx $IDX \
        --temperature 0 \
        --conv-mode vicuna_v1 &
done

wait
echo "All GPU eval workers finished."

# Step 2: merge answer shards
echo "Merging answer shards..."
> "${FINAL_ANSWERS_FILE}"
for IDX in $(seq 0 $((NUM_GPUS-1))); do
    cat "${ANSWERS_DIR}/chunk_${IDX}.jsonl" >> "${FINAL_ANSWERS_FILE}"
done
echo "Merged answers: $(wc -l < "${FINAL_ANSWERS_FILE}") lines"

# Step 3: merge router weight statistics
echo "Merging router weight statistics..."
python -m llava.eval.merge_router_stats \
    --stats-dir "${ROUTER_STATS_DIR}" \
    --num-chunks $NUM_GPUS \
    --output-file "${FINAL_ROUTER_STATS_FILE}"

# Step 4: POPE accuracy (run from repo root; relative paths OK)
echo ""
echo "=========================================="
echo "Computing POPE accuracy..."
echo "=========================================="
python llava/eval/eval_pope.py \
    --annotation-dir "${ANNOTATION_DIR}" \
    --question-file "${QUESTION_FILE}" \
    --result-file "${FINAL_ANSWERS_FILE}"

echo ""
echo "=========================================="
echo "Done."
echo "=========================================="
echo "Answers: $FINAL_ANSWERS_FILE"
echo "Layer-wise router weight statistics: $FINAL_ROUTER_STATS_FILE"
echo "=========================================="

