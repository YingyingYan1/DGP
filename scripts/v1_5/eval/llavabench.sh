#!/bin/bash
set -euo pipefail

# ========== LLaVA Bench in the Wild eval (8 GPUs) ==========
# Native pipeline: model_vqa -> merge -> GPT-4 review -> summarize
# Full SFT DGP ckpt: builder.py auto-loads layerwise_fusion (no --model-base)

PROJECT_ROOT="."
cd "${PROJECT_ROOT}" || exit 1

MODEL_PATH="${MODEL_PATH:-./checkpoint_glimpse-linear-cancha-loss/llava-v1.5-13b-sft-glimpse-linear-cancha-loss}"
CKPT="${CKPT:-llava-v1.5-13b-sft-glimpse-linear-cancha-loss}"

BENCH_ROOT="${PROJECT_ROOT}/playground/data/eval/llava-bench-in-the-wild"
QUESTION_FILE="${BENCH_ROOT}/questions.jsonl"
IMAGE_FOLDER="${BENCH_ROOT}/images"
CONTEXT_FILE="${BENCH_ROOT}/context.jsonl"
RULE_FILE="${PROJECT_ROOT}/llava/eval/table/rule.json"
GPT4_ANSWER_FILE="${BENCH_ROOT}/answers_gpt4.jsonl"

ANSWERS_DIR="${BENCH_ROOT}/answers"
REVIEWS_DIR="${BENCH_ROOT}/reviews"
FINAL_ANSWERS_FILE="${ANSWERS_DIR}/${CKPT}.jsonl"
FINAL_REVIEW_FILE="${REVIEWS_DIR}/${CKPT}.jsonl"

GPU_LIST="${GPU_LIST:-0,1,2,3,4,5,6,7}"
IFS=',' read -ra GPULIST <<< "${GPU_LIST}"
NUM_GPUS=${#GPULIST[@]}

if [[ ! -d "${MODEL_PATH}" ]]; then
    echo "Missing model path: ${MODEL_PATH}"
    exit 1
fi
if [[ ! -f "${QUESTION_FILE}" ]]; then
    echo "Missing questions: ${QUESTION_FILE}"
    exit 1
fi
if [[ ! -d "${IMAGE_FOLDER}" ]]; then
    echo "Missing image folder: ${IMAGE_FOLDER}"
    exit 1
fi

# Fail early if official bench images are incomplete (24 expected)
MISSING_IMGS=()
for img in 001.jpg 002.jpg 003.jpg 004.jpg 005.jpg 006.jpg 007.jpg 008.jpg 009.jpg 010.jpg \
           011.jpg 012.jpg 013.jpg 014.jpg 015.jpg 016.jpg 017.jpg 018.jpg 019.jpg 020.jpg \
           021.jpg 022.jpg 023.jpg 024.jpg; do
    if [[ ! -f "${IMAGE_FOLDER}/${img}" ]]; then
        MISSING_IMGS+=("${img}")
    fi
done
if [[ ${#MISSING_IMGS[@]} -gt 0 ]]; then
    echo "Incomplete LLaVA-W images under ${IMAGE_FOLDER}:"
    printf '  - %s\n' "${MISSING_IMGS[@]}"
    echo "Need all 24 official jpgs before a full 60-question run."
    exit 1
fi

echo "=========================================="
echo "LLaVA Bench in the Wild - ${NUM_GPUS} GPUs"
echo "=========================================="
echo "Model path: ${MODEL_PATH}"
echo "CKPT tag: ${CKPT}"
echo "GPUs: ${GPU_LIST}"
echo "=========================================="

mkdir -p "${ANSWERS_DIR}" "${REVIEWS_DIR}"
rm -f "${ANSWERS_DIR}/chunk_"*.jsonl

echo "Starting ${NUM_GPUS} parallel eval workers..."
for IDX in $(seq 0 $((NUM_GPUS - 1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_vqa \
        --model-path "${MODEL_PATH}" \
        --question-file "${QUESTION_FILE}" \
        --image-folder "${IMAGE_FOLDER}" \
        --answers-file "${ANSWERS_DIR}/chunk_${IDX}.jsonl" \
        --num-chunks "${NUM_GPUS}" \
        --chunk-idx "${IDX}" \
        --temperature 0 \
        --conv-mode vicuna_v1 &
done

wait
echo "All GPU workers finished inference."

echo "Merging chunk outputs..."
> "${FINAL_ANSWERS_FILE}"
for IDX in $(seq 0 $((NUM_GPUS - 1))); do
    cat "${ANSWERS_DIR}/chunk_${IDX}.jsonl" >> "${FINAL_ANSWERS_FILE}"
done
echo "Merged: $(wc -l < "${FINAL_ANSWERS_FILE}") lines"

echo "Running GPT review..."
python llava/eval/eval_gpt_review_bench.py \
    --question "${QUESTION_FILE}" \
    --context "${CONTEXT_FILE}" \
    --rule "${RULE_FILE}" \
    --answer-list \
        "${GPT4_ANSWER_FILE}" \
        "${FINAL_ANSWERS_FILE}" \
    --output "${FINAL_REVIEW_FILE}"

python llava/eval/summarize_gpt_review.py -f "${FINAL_REVIEW_FILE}"

echo "=========================================="
echo "Done."
echo "Answers: ${FINAL_ANSWERS_FILE}"
echo "Review file: ${FINAL_REVIEW_FILE}"
echo "=========================================="
