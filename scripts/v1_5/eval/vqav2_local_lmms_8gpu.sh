#!/bin/bash
set -euo pipefail

# 8-GPU local VQAv2 (val2014) evaluation for lmms-eval.
# - Split local jsonl into 8 shards
# - Run 8 single-GPU lmms-eval processes in parallel (batch_size=1)
# - Aggregate exact_match from all sample logs

REPO_ROOT="."
# External lmms-eval checkout (not vendored). Clone/install then:
#   export LMMS_ROOT=./path/to/lmms-eval
LMMS_ROOT="${LMMS_ROOT:-}"
MODEL_PATH="${REPO_ROOT}/checkpoint_glimpse-linear/llava-v1.5-7b-sft-glimpse-linear/"
INPUT_JSONL="${REPO_ROOT}/playground/data/eval/vqav2/vqav2_val2014_local.jsonl"
IMAGE_ROOT="${IMAGE_ROOT:-${REPO_ROOT}/playground/data/eval/vqav2/val2014}"

GPU_LIST="${GPU_LIST:-0,1,2,3,4,5,6,7}"
CONV_TEMPLATE="${CONV_TEMPLATE:-vicuna_v1}"
NUM_GPUS="${NUM_GPUS:-8}"

RUNTIME_DIR="${REPO_ROOT}/playground/data/eval/vqav2/lmms_vqav2_local_runtime"
SHARD_DIR="${RUNTIME_DIR}/shards"
TASK_DIR="${RUNTIME_DIR}/tasks"
LOG_ROOT="${REPO_ROOT}/logs/vqav2_val_local_8gpu_glimpse-linear-sft"
RUN_TAG="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${LOG_ROOT}/run_${RUN_TAG}"
MERGED_SAMPLES="${RUN_DIR}/merged_samples.jsonl"
SUMMARY_JSON="${RUN_DIR}/summary.json"

if [[ ! -d "${LMMS_ROOT}" ]]; then
    echo "Set LMMS_ROOT to your lmms-eval checkout (clone https://github.com/EvolvingLMMs-Lab/lmms-eval). Got: ${LMMS_ROOT:-<unset>}"
    exit 1
fi
if [[ ! -f "${INPUT_JSONL}" ]]; then
    echo "Missing local jsonl: ${INPUT_JSONL}"
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

IFS=',' read -ra GPUS <<< "${GPU_LIST}"
if [[ "${#GPUS[@]}" -ne "${NUM_GPUS}" ]]; then
    echo "NUM_GPUS=${NUM_GPUS}, but GPU_LIST has ${#GPUS[@]} entries: ${GPU_LIST}"
    exit 1
fi

mkdir -p "${SHARD_DIR}" "${TASK_DIR}" "${LOG_ROOT}"
rm -rf "${LOG_ROOT:?}"/*
mkdir -p "${RUN_DIR}"
export RUN_DIR MERGED_SAMPLES SUMMARY_JSON
rm -f "${SHARD_DIR}"/shard_*.jsonl
rm -f "${TASK_DIR}"/vqav2_val_local_shard_*.yaml "${TASK_DIR}/utils.py"

echo "Preparing ${NUM_GPUS} shards from ${INPUT_JSONL} ..."
python - <<'PY'
import json
from pathlib import Path

input_jsonl = Path("./playground/data/eval/vqav2/vqav2_val2014_local.jsonl")
shard_dir = Path("./playground/data/eval/vqav2/lmms_vqav2_local_runtime/shards")
num_gpus = int("8")

writers = []
for i in range(num_gpus):
    writers.append((shard_dir / f"shard_{i}.jsonl").open("w", encoding="utf-8"))

counts = [0] * num_gpus
with input_jsonl.open("r", encoding="utf-8") as f:
    for idx, line in enumerate(f):
        shard_idx = idx % num_gpus
        writers[shard_idx].write(line)
        counts[shard_idx] += 1

for w in writers:
    w.close()

print("Shard counts:", counts)
print("Total:", sum(counts))
PY

echo "Generating shard task YAMLs ..."
# Custom local-task helper kept in this repo (not part of upstream lmms-eval).
cp "${REPO_ROOT}/scripts/v1_5/eval/lmms_local_tasks/vqav2_local/utils.py" "${TASK_DIR}/utils.py"
for IDX in $(seq 0 $((NUM_GPUS - 1))); do
    cat > "${TASK_DIR}/vqav2_val_local_shard_${IDX}.yaml" <<EOF
task: "vqav2_val_local_shard_${IDX}"
dataset_path: json
dataset_kwargs:
  data_files:
    validation: ${SHARD_DIR}/shard_${IDX}.jsonl
output_type: generate_until
doc_to_visual: !function utils.vqav2_local_doc_to_visual
doc_to_text: !function utils.vqav2_local_doc_to_text
doc_to_target: !function utils.vqav2_local_doc_to_target
test_split: validation
metric_list:
  - metric: exact_match
    aggregation: mean
    higher_is_better: true
    ignore_case: true
    ignore_punctuation: true
process_results: !function utils.vqav2_process_results_val
lmms_eval_specific_kwargs:
  default:
    image_root: ${IMAGE_ROOT}
    pre_prompt: ""
    post_prompt: "\\nAnswer the question using a single word or phrase."
EOF
done

echo "Launching ${NUM_GPUS} single-GPU processes (batch_size=1) ..."
PIDS=()
for IDX in $(seq 0 $((NUM_GPUS - 1))); do
    GPU_ID="${GPUS[$IDX]}"
    OUT_DIR="${RUN_DIR}/shard_${IDX}"
    mkdir -p "${OUT_DIR}"
    CUDA_VISIBLE_DEVICES="${GPU_ID}" HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
    python -m lmms_eval \
        --model llava \
        --model_args "pretrained=${MODEL_PATH},conv_template=${CONV_TEMPLATE},device=cuda:0,device_map=cuda:0" \
        --tasks "vqav2_val_local_shard_${IDX}" \
        --include_path "${TASK_DIR}" \
        --batch_size 1 \
        --log_samples \
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

echo "Aggregating exact_match from sample logs ..."
python - <<'PY'
import glob
import json
import os
from pathlib import Path

run_dir = Path(os.environ["RUN_DIR"])
merged_samples = Path(os.environ["MERGED_SAMPLES"])
summary_json = Path(os.environ["SUMMARY_JSON"])

sample_files = sorted(
    glob.glob(str(run_dir / "shard_*" / "**" / "*samples*.jsonl"), recursive=True)
)
if not sample_files:
    raise RuntimeError(f"No sample jsonl found under {run_dir}")

total = 0
correct_sum = 0.0
per_file = {}

with merged_samples.open("w", encoding="utf-8") as merged_f:
    for sf in sample_files:
        cnt = 0
        acc = 0.0
        with open(sf, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                if "exact_match" not in obj:
                    continue
                val = float(obj["exact_match"])
                cnt += 1
                acc += val
                merged_f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        per_file[sf] = {"count": cnt, "sum_exact_match": acc}
        total += cnt
        correct_sum += acc

if total == 0:
    raise RuntimeError("No exact_match records found in sample files.")

summary = {
    "total_samples": total,
    "sum_exact_match": correct_sum,
    "exact_match_mean": correct_sum / total,
    "sample_files": per_file,
}
with summary_json.open("w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

print("Total samples:", total)
print("Exact match mean:", round(summary["exact_match_mean"], 6))
print("Exact match (%):", f'{summary["exact_match_mean"] * 100:.2f}%')
print("Merged samples:", merged_samples)
print("Summary:", summary_json)
PY

echo "Done. Run dir: ${RUN_DIR}"
echo "Done. Summary file: ${SUMMARY_JSON}"
