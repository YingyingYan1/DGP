#!/bin/bash
set -u

# ========== ScienceQA eval (Cross Attention checkpoint) + layer-wise router weight statistics (multi-GPU) ==========
# Collect two router weight tensors:
#   1. Pre-Router: before LLM layer 0, layerwise vision fusion over 24 ViT layers -> [1, 24] weights
#   2. LLM Router: inside 32 LLM layers, one layer-wise router per layer -> [32, 24] weights
# Output JSON includes separate pre_router and llm_router router weight statistics

# Full SFT checkpoint (no --model-base needed)
MODEL_PATH="./checkpoint_glimpse-linear-cancha-loss/llava-v1.5-7b-sft-glimpse-linear-cancha-loss/checkpoint-3300/"
RUN_TAG="llava-v1.5-7b-sft-gls-linear-canchaloss-3300"

# Output files
ANSWERS_DIR="./playground/data/eval/scienceqa/answers"
ROUTER_STATS_DIR="./playground/data/eval/scienceqa/router_stats"
FINAL_ANSWERS_FILE="${ANSWERS_DIR}/${RUN_TAG}-with-stats.jsonl"
FINAL_ROUTER_STATS_FILE="${ROUTER_STATS_DIR}/${RUN_TAG}-router-weights.json"
SCIENCEQA_OUTPUT_FILE="${ANSWERS_DIR}/${RUN_TAG}_output.jsonl"
SCIENCEQA_RESULT_FILE="${ANSWERS_DIR}/${RUN_TAG}_result.json"

# GPUs (edit gpu_list as needed)
gpu_list="0,1,2,3,4,5,6,7"
IFS=',' read -ra GPULIST <<< "$gpu_list"
NUM_GPUS=${#GPULIST[@]}

echo "=========================================="
echo "ScienceQA eval - ${NUM_GPUS} GPUs + router weight statistics"
echo "=========================================="
echo "SFT model: $MODEL_PATH"
echo "=========================================="

mkdir -p $ANSWERS_DIR
mkdir -p $ROUTER_STATS_DIR

# Remove stale chunk outputs
rm -f ${ANSWERS_DIR}/chunk_*.jsonl
rm -f ${ROUTER_STATS_DIR}/chunk_*_stats.json

# Step 1: parallel eval
echo "Starting ${NUM_GPUS} eval workers (GPUs: ${gpu_list})..."

PIDS=()
for IDX in $(seq 0 $((NUM_GPUS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_vqa_science \
        --model-path $MODEL_PATH \
        --question-file ./playground/data/eval/scienceqa/llava_test_CQM-A.json \
        --image-folder ./playground/data/eval/scienceqa/test \
        --answers-file ${ANSWERS_DIR}/chunk_${IDX}.jsonl \
        --router-stats-file ${ROUTER_STATS_DIR}/chunk_${IDX}_stats.json \
        --num-chunks $NUM_GPUS \
        --chunk-idx $IDX \
        --single-pred-prompt \
        --temperature 0 \
        --conv-mode vicuna_v1 &
    PIDS+=($!)
done

FAIL=0
for PID in "${PIDS[@]}"; do
    if ! wait "$PID"; then
        FAIL=1
    fi
done

if [ "$FAIL" -ne 0 ]; then
    echo "Error: at least one eval worker failed; skipping merge and scoring."
    exit 1
fi

echo "All GPU eval workers finished."

# Step 2: merge answer shards
echo "Merging answer shards..."
> $FINAL_ANSWERS_FILE
for IDX in $(seq 0 $((NUM_GPUS-1))); do
    cat ${ANSWERS_DIR}/chunk_${IDX}.jsonl >> $FINAL_ANSWERS_FILE
done
echo "Merged answers: $(wc -l < $FINAL_ANSWERS_FILE) lines"

if [ ! -s "$FINAL_ANSWERS_FILE" ]; then
    echo "Error: merged answers file is empty; aborting."
    exit 1
fi

# Step 3: merge router weight statistics
echo "Merging router weight statistics..."
python -m llava.eval.merge_router_stats \
    --stats-dir $ROUTER_STATS_DIR \
    --num-chunks $NUM_GPUS \
    --output-file $FINAL_ROUTER_STATS_FILE

# Step 4: ScienceQA scoring
echo ""
echo "=========================================="
echo "Computing ScienceQA accuracy..."
echo "=========================================="
python llava/eval/eval_science_qa.py \
    --base-dir ./playground/data/eval/scienceqa \
    --result-file $FINAL_ANSWERS_FILE \
    --output-file $SCIENCEQA_OUTPUT_FILE \
    --output-result $SCIENCEQA_RESULT_FILE

echo ""
echo "=========================================="
echo "Done."
echo "=========================================="
echo "Answers: $FINAL_ANSWERS_FILE"
echo "Layer-wise router weight statistics: $FINAL_ROUTER_STATS_FILE"
echo "ScienceQA metrics: $SCIENCEQA_RESULT_FILE"
echo "=========================================="

