#!/bin/bash
set -euo pipefail

# Stage-1 only: run model inference and export predictions jsonl for later GPT judging.
# No API calls in this script.

REPO_ROOT="."
# External lmms-eval checkout (not vendored). Clone/install then:
#   export LMMS_ROOT=./path/to/lmms-eval
LMMS_ROOT="${LMMS_ROOT:-}"
MODEL_PATH="${MODEL_PATH:-./checkpoint_glimpse-linear/llava-v1.5-7b-sft-glimpse-linear/checkpoint-3300/}"

BENCH_ROOT="${BENCH_ROOT:-${REPO_ROOT}/playground/data/eval/llava-bench-in-the-wild}"
IMAGE_ROOT="${IMAGE_ROOT:-${BENCH_ROOT}/images}"
QUESTIONS_JSONL="${QUESTIONS_JSONL:-${BENCH_ROOT}/questions.jsonl}"
ANSWERS_GPT4_JSONL="${ANSWERS_GPT4_JSONL:-${BENCH_ROOT}/answers_gpt4.jsonl}"

GPU_LIST="${GPU_LIST:-0,1,2,3,4,5,6,7}"
CONV_TEMPLATE="${CONV_TEMPLATE:-vicuna_v1}"
NUM_GPUS="${NUM_GPUS:-8}"

RUNTIME_DIR="${BENCH_ROOT}/lmms_llava_wilder_local_runtime"
MERGED_JSONL="${RUNTIME_DIR}/llava_wilder_local.jsonl"
SHARD_DIR="${RUNTIME_DIR}/shards"
TASK_DIR="${RUNTIME_DIR}/tasks_infer_only"

# Log / prediction filename suffix (decoupled from checkpoint dir for stable naming)
RUN_LABEL="${RUN_LABEL:-glimpse-linear-3300}"

LOG_ROOT="${REPO_ROOT}/logs/llava_wilder_infer_only_8gpu_${RUN_LABEL}"
RUN_TAG="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${LOG_ROOT}/run_${RUN_LABEL}_${RUN_TAG}"
PREDICTIONS_JSONL="${RUN_DIR}/predictions_for_judge_${RUN_LABEL}.jsonl"

if [[ ! -d "${LMMS_ROOT}" ]]; then
    echo "Set LMMS_ROOT to your lmms-eval checkout (clone https://github.com/EvolvingLMMs-Lab/lmms-eval). Got: ${LMMS_ROOT:-<unset>}"
    exit 1
fi
if [[ ! -d "${MODEL_PATH}" ]]; then
    echo "Missing model path: ${MODEL_PATH}"
    exit 1
fi
if [[ ! -d "${IMAGE_ROOT}" ]]; then
    echo "Missing image root: ${IMAGE_ROOT}"
    exit 1
fi
if [[ ! -f "${QUESTIONS_JSONL}" ]]; then
    echo "Missing questions jsonl: ${QUESTIONS_JSONL}"
    exit 1
fi
if [[ ! -f "${ANSWERS_GPT4_JSONL}" ]]; then
    echo "Missing GPT4 answers jsonl: ${ANSWERS_GPT4_JSONL}"
    exit 1
fi

IFS=',' read -ra GPUS <<< "${GPU_LIST}"
if [[ "${#GPUS[@]}" -ne "${NUM_GPUS}" ]]; then
    echo "NUM_GPUS=${NUM_GPUS}, but GPU_LIST has ${#GPUS[@]} entries: ${GPU_LIST}"
    exit 1
fi

export QUESTIONS_JSONL ANSWERS_GPT4_JSONL IMAGE_ROOT MERGED_JSONL SHARD_DIR NUM_GPUS RUN_DIR PREDICTIONS_JSONL

mkdir -p "${SHARD_DIR}" "${TASK_DIR}" "${LOG_ROOT}" "${RUN_DIR}"
rm -f "${SHARD_DIR}"/shard_*.jsonl
rm -f "${TASK_DIR}"/llava_wilder_infer_shard_*.yaml "${TASK_DIR}/utils.py"

echo "Building merged jsonl: ${MERGED_JSONL}"
python - <<'PY'
import json
import os
from pathlib import Path

questions_jsonl = Path(os.environ["QUESTIONS_JSONL"])
answers_jsonl = Path(os.environ["ANSWERS_GPT4_JSONL"])
image_root = Path(os.environ["IMAGE_ROOT"])
merged_jsonl = Path(os.environ["MERGED_JSONL"])

answers = {}
with answers_jsonl.open("r", encoding="utf-8") as f:
    for line in f:
        item = json.loads(line)
        answers[item["question_id"]] = item

count = 0
missing = 0
merged_jsonl.parent.mkdir(parents=True, exist_ok=True)
with merged_jsonl.open("w", encoding="utf-8") as out_f:
    with questions_jsonl.open("r", encoding="utf-8") as qf:
        for line in qf:
            q = json.loads(line)
            qid = q["question_id"]
            ans = answers.get(qid)
            if ans is None:
                missing += 1
                continue
            record = {
                "question_id": qid,
                "category": q.get("category"),
                "Question": q.get("text", ""),
                "Answer": ans.get("text", ""),
                "image": str(image_root / q.get("image", "")),
            }
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1

print(f"Merged records: {count}, missing answers: {missing}")
PY

echo "Preparing ${NUM_GPUS} shards from merged jsonl ..."
python - <<'PY'
from pathlib import Path
import os

merged_jsonl = Path(os.environ["MERGED_JSONL"])
shard_dir = Path(os.environ["SHARD_DIR"])
num_gpus = int(os.environ["NUM_GPUS"])

writes = []
for i in range(num_gpus):
    writes.append((shard_dir / f"shard_{i}.jsonl").open("w", encoding="utf-8"))

counts = [0] * num_gpus
with merged_jsonl.open("r", encoding="utf-8") as f:
    for idx, line in enumerate(f):
        shard_idx = idx % num_gpus
        writes[shard_idx].write(line)
        counts[shard_idx] += 1

for w in writes:
    w.close()

print("Shard counts:", counts)
print("Total:", sum(counts))
PY

echo "Generating infer-only task YAMLs ..."
cat > "${TASK_DIR}/utils.py" <<'PY'
from PIL import Image


def _load_image(doc):
    image = doc.get("image")
    if isinstance(image, str):
        return Image.open(image).convert("RGB")
    return image.convert("RGB")


def llava_doc_to_visual(doc):
    return [_load_image(doc)]


def llava_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}
    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "")
    post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "")
    return f"{pre_prompt}{doc.get('Question', '')}{post_prompt}"


def llava_doc_to_target(doc):
    return doc.get("Answer", "")


def nojudge_process_results(doc, result):
    pred = result[0] if result else ""
    return {"pred_text": pred}
PY

for IDX in $(seq 0 $((NUM_GPUS - 1))); do
    cat > "${TASK_DIR}/llava_wilder_infer_shard_${IDX}.yaml" <<EOF
task: "llava_wilder_infer_shard_${IDX}"
dataset_path: json
dataset_kwargs:
  data_files:
    test: ${SHARD_DIR}/shard_${IDX}.jsonl
output_type: generate_until
doc_to_visual: !function utils.llava_doc_to_visual
doc_to_text: !function utils.llava_doc_to_text
doc_to_target: !function utils.llava_doc_to_target
test_split: test
process_results: !function utils.nojudge_process_results
metric_list:
  - metric: exact_match
    aggregation: mean
    higher_is_better: true
    ignore_case: true
    ignore_punctuation: true
lmms_eval_specific_kwargs:
  default:
    pre_prompt: ""
    post_prompt: ""
metadata:
  version: 0.0
EOF
done

echo "Launching ${NUM_GPUS} infer workers (no GPT judge) ..."
export PYTHONPATH="${LMMS_ROOT}:${PYTHONPATH:-}"

PIDS=()
for IDX in $(seq 0 $((NUM_GPUS - 1))); do
    GPU_ID="${GPUS[$IDX]}"
    OUT_DIR="${RUN_DIR}/shard_${IDX}"
    mkdir -p "${OUT_DIR}"
    CUDA_VISIBLE_DEVICES="${GPU_ID}" TOKENIZERS_PARALLELISM=false \
    python -m lmms_eval \
        --model llava \
        --model_args "pretrained=${MODEL_PATH},conv_template=${CONV_TEMPLATE},device=cuda:0,device_map=cuda:0" \
        --tasks "llava_wilder_infer_shard_${IDX}" \
        --include_path "${TASK_DIR}" \
        --batch_size 1 \
        --log_samples \
        --predict_only \
        --output_path "${OUT_DIR}" &
    PIDS+=($!)
done

FAIL=0
for PID in "${PIDS[@]}"; do
    if ! wait "${PID}"; then
        echo "A shard process failed: pid=${PID}"
        FAIL=1
    fi
done
if [[ "${FAIL}" -ne 0 ]]; then
    echo "At least one shard failed."
    exit 1
fi

echo "Merging model predictions to: ${PREDICTIONS_JSONL}"
python - <<'PY'
import glob
import json
import os
from pathlib import Path

run_dir = Path(os.environ["RUN_DIR"])
predictions_jsonl = Path(os.environ["PREDICTIONS_JSONL"])
sample_files = sorted(glob.glob(str(run_dir / "shard_*" / "**" / "*samples*.jsonl"), recursive=True))
if not sample_files:
    raise RuntimeError(f"No sample jsonl found under {run_dir}")

total = 0
predictions_jsonl.parent.mkdir(parents=True, exist_ok=True)
with predictions_jsonl.open("w", encoding="utf-8") as out_f:
    for sf in sample_files:
        shard = Path(sf).parts[-4] if len(Path(sf).parts) >= 4 else "unknown_shard"
        with open(sf, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                pred = obj.get("filtered_resps", "")
                if isinstance(pred, list):
                    pred = pred[0] if pred else ""
                media = obj.get("input_media", [])
                image_path = media[0] if media else ""
                rec = {
                    "shard": shard,
                    "doc_id": obj.get("doc_id"),
                    "question": obj.get("input", ""),
                    "answer_ref": obj.get("target", ""),
                    "model_answer": pred,
                    "image_path": image_path,
                }
                out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                total += 1

print("Merged prediction count:", total)
print("Prediction file:", predictions_jsonl)
PY

echo "Done inference-only run."
echo "Run dir: ${RUN_DIR}"
echo "Predictions for judge: ${PREDICTIONS_JSONL}"
