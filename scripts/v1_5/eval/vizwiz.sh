#!/bin/bash
set -euo pipefail

# ========== VizWiz val eval ==========
# Full SFT DGP ckpt: builder.py auto-loads layerwise_fusion (no --model-base).

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
SPLIT="${SPLIT:-val}"

VIZWIZ_ROOT="${PROJECT_ROOT}/playground/data/eval/vizwiz"
ANNOTATION_JSON="${VIZWIZ_ROOT}/${SPLIT}.json"
IMAGE_FOLDER="${VIZWIZ_ROOT}/${SPLIT}"
QUESTION_FILE="${VIZWIZ_ROOT}/llava_${SPLIT}.jsonl"
ANSWERS_DIR="${VIZWIZ_ROOT}/answers/${CKPT}"
OUTPUT_FILE="${ANSWERS_DIR}/merge.jsonl"
UPLOAD_FILE="${VIZWIZ_ROOT}/answers_upload/${CKPT}.json"

if [[ -z "${GPU_LIST:-}" ]]; then
  N="$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')"
  [[ "${N}" -ge 1 ]] || { echo "No GPU found"; exit 1; }
  GPU_LIST="$(seq -s, 0 $((N - 1)))"
fi
IFS=',' read -ra GPULIST <<< "${GPU_LIST}"
NUM_GPUS=${#GPULIST[@]}

[[ -d "${MODEL_PATH}" ]] || { echo "Missing ${MODEL_PATH}"; exit 1; }
[[ -f "${ANNOTATION_JSON}" ]] || { echo "Missing ${ANNOTATION_JSON}"; exit 1; }
[[ -f "${QUESTION_FILE}" ]] || { echo "Missing ${QUESTION_FILE}"; exit 1; }

mkdir -p "${ANSWERS_DIR}" "$(dirname "${UPLOAD_FILE}")" "${PROJECT_ROOT}/result/answers"
rm -f "${ANSWERS_DIR}"/chunk_*.jsonl

echo "VizWiz ${SPLIT} | GPUs=${GPU_LIST} | model=${MODEL_PATH}"

for IDX in $(seq 0 $((NUM_GPUS - 1))); do
  CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_vqa_loader \
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
cp -f "${OUTPUT_FILE}" "${PROJECT_ROOT}/result/answers/DGP-${CKPT}-vizwiz.jsonl"

# soft accuracy + EvalAI upload
export ANNOTATION_JSON OUTPUT_FILE UPLOAD_FILE SPLIT
python - <<'PY'
import json, os
from pathlib import Path
from llava.eval.m4c_evaluator import EvalAIAnswerProcessor

anns = json.load(open(os.environ["ANNOTATION_JSON"]))
preds = {}
for line in open(os.environ["OUTPUT_FILE"]):
    row = json.loads(line)
    preds[int(row["question_id"])] = row["text"]

proc = EvalAIAnswerProcessor()
upload, total, n = [], 0.0, 0
for i, ann in enumerate(anns):
    if i not in preds:
        continue
    pred = proc(preds[i])
    upload.append({"image": ann["image"], "answer": pred})
    gts = [proc(a["answer"]) for a in ann.get("answers", [])]
    if gts:
        total += min(1.0, sum(1 for g in gts if g == pred) / 3.0)
        n += 1

Path(os.environ["UPLOAD_FILE"]).parent.mkdir(parents=True, exist_ok=True)
json.dump(upload, open(os.environ["UPLOAD_FILE"], "w"))
acc = (total / n * 100.0) if n else 0.0
print(f"VizWiz soft accuracy: {acc:.2f}%  ({n}/{len(anns)})")
PY

echo "Done. merge=${OUTPUT_FILE}"
