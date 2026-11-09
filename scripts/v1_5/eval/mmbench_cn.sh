#!/bin/bash
set -euo pipefail

# 8-GPU local MMBench-CN dev evaluation via lmms-eval (local TSV -> jsonl).
# Supports split workflow:
#   MODE=infer : only run model inference and save predictions_for_judge.jsonl
#   MODE=judge : only run GPT judging from an existing predictions_for_judge.jsonl
#   MODE=all   : infer + judge in one run (default)

REPO_ROOT="."
# External lmms-eval checkout (not vendored). Clone/install then:
#   export LMMS_ROOT=./path/to/lmms-eval
LMMS_ROOT="${LMMS_ROOT:-}"
MODEL_PATH="${MODEL_PATH:-${REPO_ROOT}/llava-v1.5-7b/}"
MODEL_DIR="${MODEL_PATH%/}"
MODEL_NAME="$(basename "${MODEL_DIR}")"
MODEL_PARENT="$(basename "$(dirname "${MODEL_DIR}")")"
if [[ "${MODEL_NAME}" == checkpoint-* ]]; then
    MODEL_TAG_DEFAULT="${MODEL_PARENT}_${MODEL_NAME}"
else
    MODEL_TAG_DEFAULT="${MODEL_NAME}"
fi
MODEL_TAG="${MODEL_TAG:-${MODEL_TAG_DEFAULT}}"

INPUT_TSV="${REPO_ROOT}/playground/data/eval/mmbench/mmbench_dev_cn_20231003.tsv"

GPU_LIST="${GPU_LIST:-0,1,2,3,4,5,6,7}"
CONV_TEMPLATE="${CONV_TEMPLATE:-vicuna_v1}"
NUM_GPUS="${NUM_GPUS:-8}"
MODE="${MODE:-all}"  # infer | judge | all

RUNTIME_DIR="${REPO_ROOT}/playground/data/eval/mmbench/lmms_mmbench_cn_dev_runtime"
MERGED_JSONL="${RUNTIME_DIR}/mmbench_cn_dev.jsonl"
IMAGE_DIR="${RUNTIME_DIR}/images"
SHARD_DIR="${RUNTIME_DIR}/shards"
TASK_DIR="${RUNTIME_DIR}/tasks"
LOG_ROOT="${REPO_ROOT}/logs/mmbench_cn_dev_8gpu_${MODEL_TAG}"
RUN_TAG="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${LOG_ROOT}/run_${MODEL_TAG}_${RUN_TAG}"
MERGED_SAMPLES="${RUN_DIR}/merged_samples.jsonl"
SUMMARY_JSON="${RUN_DIR}/summary.json"
PREDICTIONS_FOR_JUDGE="${RUN_DIR}/predictions_for_judge_${MODEL_TAG}.jsonl"
JUDGE_INPUT_FILE="${JUDGE_INPUT_FILE:-${PREDICTIONS_FOR_JUDGE}}"
JUDGE_TAG="${JUDGE_TAG:-$(echo "${MODEL_VERSION:-gpt}" | tr '/:' '__')}"
JUDGE_DIR="${JUDGE_DIR:-${RUN_DIR}/judge_${JUDGE_TAG}}"
JUDGE_SUMMARY_JSON="${JUDGE_DIR}/judge_summary.json"

if [[ ! -d "${LMMS_ROOT}" ]]; then
    echo "Set LMMS_ROOT to your lmms-eval checkout (clone https://github.com/EvolvingLMMs-Lab/lmms-eval). Got: ${LMMS_ROOT:-<unset>}"
    exit 1
fi
if [[ "${MODE}" != "infer" && "${MODE}" != "judge" && "${MODE}" != "all" ]]; then
    echo "Unsupported MODE=${MODE}. Use infer|judge|all."
    exit 1
fi

if [[ "${MODE}" == "infer" || "${MODE}" == "all" ]]; then
    if [[ ! -f "${INPUT_TSV}" ]]; then
        echo "Missing MMBench CN TSV: ${INPUT_TSV}"
        exit 1
    fi
    if [[ ! -d "${MODEL_PATH}" ]]; then
        echo "Missing model path: ${MODEL_PATH}"
        exit 1
    fi

    IFS=',' read -ra GPUS <<< "${GPU_LIST}"
    if [[ "${#GPUS[@]}" -ne "${NUM_GPUS}" ]]; then
        echo "NUM_GPUS=${NUM_GPUS}, but GPU_LIST has ${#GPUS[@]} entries: ${GPU_LIST}"
        exit 1
    fi

    mkdir -p "${IMAGE_DIR}" "${SHARD_DIR}" "${TASK_DIR}" "${LOG_ROOT}" "${RUN_DIR}"
    rm -f "${SHARD_DIR}"/shard_*.jsonl
    rm -f "${TASK_DIR}"/mmbench_cn_dev_shard_*.yaml "${TASK_DIR}/utils.py"
    export INPUT_TSV MERGED_JSONL IMAGE_DIR SHARD_DIR NUM_GPUS RUN_DIR MERGED_SAMPLES SUMMARY_JSON PREDICTIONS_FOR_JUDGE

    echo "Converting TSV to jsonl + images: ${MERGED_JSONL}"
    python - <<'PY'
import base64
import json
import os
from pathlib import Path
import pandas as pd

tsv_path = Path(os.environ["INPUT_TSV"])
merged_jsonl = Path(os.environ["MERGED_JSONL"])
image_dir = Path(os.environ["IMAGE_DIR"])

df = pd.read_csv(tsv_path, sep="\t")
image_dir.mkdir(parents=True, exist_ok=True)
merged_jsonl.parent.mkdir(parents=True, exist_ok=True)

def decode_image(b64_str, output_path):
    if not isinstance(b64_str, str) or not b64_str.strip():
        raise ValueError("Empty image field in TSV.")
    if "base64," in b64_str:
        b64_str = b64_str.split("base64,")[-1]
    output_path.write_bytes(base64.b64decode(b64_str))

def to_str(v):
    # Avoid NaN/float mixing with string columns for HF datasets/pyarrow.
    if v is None:
        return ""
    try:
        if pd.isna(v):
            return ""
    except Exception:
        pass
    return str(v)

with merged_jsonl.open("w", encoding="utf-8") as f:
    for _, row in df.iterrows():
        index = int(row["index"])
        image_path = image_dir / f"{index}.jpg"
        decode_image(row["image"], image_path)
        record = {
            "index": index,
            "question": to_str(row.get("question", "")),
            "hint": to_str(row.get("hint", "")),
            "A": to_str(row.get("A", "")),
            "B": to_str(row.get("B", "")),
            "C": to_str(row.get("C", "")),
            "D": to_str(row.get("D", "")),
            "answer": to_str(row.get("answer", "")),
            "category": to_str(row.get("category", "")),
            "L2-category": to_str(row.get("l2-category", row.get("L2-category", ""))),
            "image": str(image_path),
            "source": to_str(row.get("source", "")),
            "split": to_str(row.get("split", "dev")) or "dev",
        }
        f.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
print(f"Wrote {len(df)} records to {merged_jsonl}")
PY

    echo "Preparing ${NUM_GPUS} shards from ${MERGED_JSONL} ..."
    python - <<'PY'
import os
from pathlib import Path

merged_jsonl = Path(os.environ["MERGED_JSONL"])
shard_dir = Path(os.environ["SHARD_DIR"])
num_gpus = int(os.environ["NUM_GPUS"])
writers = [(shard_dir / f"shard_{i}.jsonl").open("w", encoding="utf-8") for i in range(num_gpus)]
counts = [0] * num_gpus
with merged_jsonl.open("r", encoding="utf-8") as f:
    for idx, line in enumerate(f):
        si = idx % num_gpus
        writers[si].write(line)
        counts[si] += 1
for w in writers:
    w.close()
print("Shard counts:", counts)
print("Total:", sum(counts))
PY

    echo "Generating shard task YAMLs ..."
    cat > "${TASK_DIR}/utils.py" <<'PY'
from PIL import Image

from lmms_eval.tasks.mmbench import cn_utils


def mmbench_doc_to_visual(doc):
    image = doc["image"]
    if isinstance(image, str):
        return [Image.open(image).convert("RGB")]
    return [image.convert("RGB")]


mmbench_doc_to_text = cn_utils.mmbench_doc_to_text
mmbench_process_results = cn_utils.mmbench_process_results
mmbench_aggregate_dev_results_eval = cn_utils.mmbench_aggregate_dev_results_eval
mmbench_aggregate_dev_results_submission = cn_utils.mmbench_aggregate_dev_results
PY

    for IDX in $(seq 0 $((NUM_GPUS - 1))); do
        cat > "${TASK_DIR}/mmbench_cn_dev_shard_${IDX}.yaml" <<EOF
task: "mmbench_cn_dev_shard_${IDX}"
dataset_path: json
dataset_kwargs:
  data_files:
    test: ${SHARD_DIR}/shard_${IDX}.jsonl
output_type: generate_until
doc_to_visual: !function utils.mmbench_doc_to_visual
doc_to_text: !function utils.mmbench_doc_to_text
doc_to_target: "answer"
test_split: test
process_results: !function utils.mmbench_process_results
metric_list:
  - metric: submission
    aggregation: !function utils.mmbench_aggregate_dev_results_submission
    higher_is_better: true
EOF
    done

    echo "Launching ${NUM_GPUS} single-GPU processes (batch_size=1) ..."
    PIDS=()
    for IDX in $(seq 0 $((NUM_GPUS - 1))); do
        GPU_ID="${GPUS[$IDX]}"
        OUT_DIR="${RUN_DIR}/shard_${IDX}"
        mkdir -p "${OUT_DIR}"
        CUDA_VISIBLE_DEVICES="${GPU_ID}" TOKENIZERS_PARALLELISM=false \
        python -m lmms_eval \
            --model llava \
            --model_args "pretrained=${MODEL_PATH},conv_template=${CONV_TEMPLATE},device=cuda:0,device_map=cuda:0" \
            --tasks "mmbench_cn_dev_shard_${IDX}" \
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

    echo "Merging prediction rows for later judging ..."
    python - <<'PY'
import glob
import json
import os
from pathlib import Path

run_dir = Path(os.environ["RUN_DIR"])
merged_samples = Path(os.environ["MERGED_SAMPLES"])
predictions = Path(os.environ["PREDICTIONS_FOR_JUDGE"])
summary_json = Path(os.environ["SUMMARY_JSON"])

sample_files = sorted(glob.glob(str(run_dir / "shard_*" / "**" / "*samples*.jsonl"), recursive=True))
if not sample_files:
    raise RuntimeError(f"No sample jsonl found under {run_dir}")

rows = []
with merged_samples.open("w", encoding="utf-8") as merged_f:
    for sf in sample_files:
        with open(sf, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                if "submission" not in obj:
                    continue
                rows.append(obj["submission"])
                merged_f.write(json.dumps(obj, ensure_ascii=False) + "\n")

with predictions.open("w", encoding="utf-8") as f:
    for row in rows:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

summary = {
    "mode": "infer",
    "total_samples": len(rows),
    "predictions_for_judge": str(predictions),
    "sample_files": sample_files,
    "merged_samples": str(merged_samples),
}
with summary_json.open("w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

print("Total samples:", summary["total_samples"])
print("Predictions for judge:", predictions)
print("Summary:", summary_json)
PY

    if [[ "${MODE}" == "infer" ]]; then
        echo "Infer finished. Use MODE=judge with JUDGE_INPUT_FILE=${PREDICTIONS_FOR_JUDGE}"
        exit 0
    fi
fi

if [[ "${MODE}" == "judge" || "${MODE}" == "all" ]]; then
    if [[ ! -f "${JUDGE_INPUT_FILE}" ]]; then
        echo "Missing JUDGE_INPUT_FILE: ${JUDGE_INPUT_FILE}"
        exit 1
    fi
    mkdir -p "${JUDGE_DIR}"
    export JUDGE_INPUT_FILE JUDGE_DIR JUDGE_SUMMARY_JSON

    echo "Running GPT judge from ${JUDGE_INPUT_FILE} ..."
python - <<'PY'
import json
import os
from pathlib import Path

from lmms_eval.tasks.mmbench import cn_utils

judge_input = Path(os.environ["JUDGE_INPUT_FILE"])
judge_dir = Path(os.environ["JUDGE_DIR"])
judge_summary = Path(os.environ["JUDGE_SUMMARY_JSON"])

submission_rows = []
with judge_input.open("r", encoding="utf-8") as f:
    for line in f:
        submission_rows.append(json.loads(line))

class Args:
    output_path = str(judge_dir)

overall_acc = cn_utils.mmbench_aggregate_dev_results_eval(submission_rows, Args())
cn_utils.mmbench_aggregate_dev_results(submission_rows, Args())

summary = {
    "mode": "judge",
    "judge_input_file": str(judge_input),
    "judge_output_dir": str(judge_dir),
    "total_samples": len(submission_rows),
    "mmbench_cn_dev_acc": overall_acc,
}
with judge_summary.open("w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

print("Total samples:", summary["total_samples"])
print("MMBench-CN Dev Acc (%):", overall_acc)
print("Judge summary:", judge_summary)
PY
fi

echo "Done. Run dir: ${RUN_DIR}"
echo "Infer summary file: ${SUMMARY_JSON}"
echo "Judge summary file: ${JUDGE_SUMMARY_JSON}"
