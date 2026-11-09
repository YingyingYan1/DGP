#!/bin/bash
set -euo pipefail

# ========== SEED-Bench eval ==========
# 1) shard infer  2) merge  3) score (Total=SEED, image=SEED-I, video)

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

SEED_ROOT="${PROJECT_ROOT}/playground/data/eval/seed_bench"
MODEL_PATH="${MODEL_PATH:-${PROJECT_ROOT}/checkpoint_glimpse-linear-cancha-loss/llava-v1.5-7b-sft-glimpse-linear-cancha-loss}"
CKPT="${CKPT:-llava-v1.5-7b-sft-glimpse-linear-cancha-loss}"

QUESTION_FILE="${SEED_ROOT}/llava-seed-bench-clean.jsonl"
IMAGE_FOLDER="${SEED_ROOT}"
ANNOTATION_FILE="${SEED_ROOT}/SEED-Bench.json"
ANSWERS_DIR="${SEED_ROOT}/answers/${CKPT}"
OUTPUT_FILE="${ANSWERS_DIR}/merge.jsonl"
UPLOAD_FILE="${SEED_ROOT}/answers_upload/${CKPT}.jsonl"

# Auto-detect GPUs unless GPU_LIST is set
if [[ -z "${GPU_LIST:-}" ]]; then
  N="$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')"
  [[ "${N}" -ge 1 ]] || { echo "No GPU found"; exit 1; }
  GPU_LIST="$(seq -s, 0 $((N - 1)))"
fi
IFS=',' read -ra GPULIST <<< "${GPU_LIST}"
NUM_GPUS=${#GPULIST[@]}

[[ -f "${QUESTION_FILE}" ]] || { echo "Missing ${QUESTION_FILE}"; exit 1; }
[[ -f "${ANNOTATION_FILE}" ]] || { echo "Missing ${ANNOTATION_FILE}"; exit 1; }
[[ -d "${MODEL_PATH}" ]] || { echo "Missing ${MODEL_PATH}"; exit 1; }

mkdir -p "${ANSWERS_DIR}" "$(dirname "${UPLOAD_FILE}")" "${PROJECT_ROOT}/result/answers"
rm -f "${ANSWERS_DIR}"/chunk_*.jsonl "${OUTPUT_FILE}"

echo "SEED eval | GPUs=${GPU_LIST} | model=${MODEL_PATH}"

for IDX in $(seq 0 $((NUM_GPUS - 1))); do
  CUDA_VISIBLE_DEVICES="${GPULIST[$IDX]}" python -m llava.eval.model_vqa_loader \
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

> "${OUTPUT_FILE}"
for IDX in $(seq 0 $((NUM_GPUS - 1))); do
  cat "${ANSWERS_DIR}/chunk_${IDX}.jsonl" >> "${OUTPUT_FILE}"
done
cp -f "${OUTPUT_FILE}" "${PROJECT_ROOT}/result/answers/DGP-${CKPT}-seed.jsonl"

python scripts/convert_seed_for_submission.py \
  --annotation-file "${ANNOTATION_FILE}" \
  --result-file "${OUTPUT_FILE}" \
  --result-upload-file "${UPLOAD_FILE}"

echo "Done. merge=${OUTPUT_FILE}"
