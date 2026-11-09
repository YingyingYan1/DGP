#!/bin/bash
set -euo pipefail
# LLaVA-W inference only (no GPT judge).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
cd "${PROJECT_ROOT}" || exit 1
# Use active conda env if present; override with CONDA_ENV_BIN if needed
if [[ -n "${CONDA_ENV_BIN:-}" ]]; then
  export PATH="${CONDA_ENV_BIN}:${PATH}"
elif [[ -n "${CONDA_PREFIX:-}" ]]; then
  export PATH="${CONDA_PREFIX}/bin:${PATH}"
fi
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export LD_LIBRARY_PATH=

MODEL_PATH="${MODEL_PATH:-${PROJECT_ROOT}/checkpoint_glimpse-linear-cancha-loss/llava-v1.5-13b-sft-glimpse-linear-cancha-loss}"
CKPT="${CKPT:-llava-v1.5-13b-sft-glimpse-linear-cancha-loss}"

BENCH_ROOT="${PROJECT_ROOT}/playground/data/eval/llava-bench-in-the-wild"
QUESTION_FILE="${BENCH_ROOT}/questions.jsonl"
IMAGE_FOLDER="${BENCH_ROOT}/images"
ANSWERS_DIR="${BENCH_ROOT}/answers"
FINAL_ANSWERS_FILE="${ANSWERS_DIR}/${CKPT}.jsonl"

if [[ -z "${GPU_LIST:-}" ]]; then
  N="$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')"
  [[ "${N}" -ge 1 ]] || { echo "No GPU found"; exit 1; }
  GPU_LIST="$(seq -s, 0 $((N - 1)))"
fi
IFS=',' read -ra GPULIST <<< "${GPU_LIST}"
NUM_GPUS=${#GPULIST[@]}

[[ -d "${MODEL_PATH}" ]] || { echo "Missing ${MODEL_PATH}"; exit 1; }
[[ -f "${QUESTION_FILE}" ]] || { echo "Missing ${QUESTION_FILE}"; exit 1; }

mkdir -p "${ANSWERS_DIR}" "${PROJECT_ROOT}/result/answers"
rm -f "${ANSWERS_DIR}/chunk_"*.jsonl

echo "LLaVA-W infer | GPUs=${GPU_LIST} | model=${MODEL_PATH}"

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

> "${FINAL_ANSWERS_FILE}"
for IDX in $(seq 0 $((NUM_GPUS - 1))); do
  cat "${ANSWERS_DIR}/chunk_${IDX}.jsonl" >> "${FINAL_ANSWERS_FILE}"
done
cp -f "${FINAL_ANSWERS_FILE}" "${PROJECT_ROOT}/result/answers/DGP-${CKPT}-llava-w.jsonl"

echo "Done. answers=${FINAL_ANSWERS_FILE}"
