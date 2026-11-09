#!/bin/bash
set -euo pipefail

# 8-GPU local SEED-Bench evaluation via lmms-eval (local jsonl + local images).

REPO_ROOT="."
# External lmms-eval checkout (not vendored). Clone/install then:
#   export LMMS_ROOT=./path/to/lmms-eval
LMMS_ROOT="${LMMS_ROOT:-}"
MODEL_PATH="${MODEL_PATH:-./checkpoint_glimpse-fu/llava-v1.5-7b-sft-glimpse-fu/checkpoint-3100/}"

SEED_ROOT="${REPO_ROOT}/playground/data/eval/seed_bench"
IMAGE_ROOT="${SEED_ROOT}/SEED-Bench-image"
VIDEO_IMAGE_ROOT="${SEED_ROOT}/SEED-Bench-video-image"
ANNOTATION_JSON="${SEED_ROOT}/SEED-Bench.json"

GPU_LIST="${GPU_LIST:-0,1,2,3,4,5,6,7}"
CONV_TEMPLATE="${CONV_TEMPLATE:-vicuna_v1}"
NUM_GPUS="${NUM_GPUS:-8}"
MODEL_TAG="${MODEL_TAG:-$(basename "${MODEL_PATH%/}")}"

RUNTIME_DIR="${SEED_ROOT}/lmms_seedbench_local_runtime"
MERGED_JSONL="${RUNTIME_DIR}/seedbench_local.jsonl"
SHARD_DIR="${RUNTIME_DIR}/shards"
TASK_DIR="${RUNTIME_DIR}/tasks"
LOG_ROOT="${REPO_ROOT}/logs/seedbench_local_8gpu_${MODEL_TAG}"
RUN_TAG="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${LOG_ROOT}/run_${MODEL_TAG}_${RUN_TAG}"
MERGED_SAMPLES="${RUN_DIR}/merged_samples.jsonl"
SUMMARY_JSON="${RUN_DIR}/summary.json"

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
if [[ ! -d "${VIDEO_IMAGE_ROOT}" ]]; then
    echo "Missing video-image root: ${VIDEO_IMAGE_ROOT}"
    exit 1
fi
if [[ ! -f "${ANNOTATION_JSON}" ]]; then
    echo "Missing annotation json: ${ANNOTATION_JSON}"
    exit 1
fi

IFS=',' read -ra GPUS <<< "${GPU_LIST}"
if [[ "${#GPUS[@]}" -ne "${NUM_GPUS}" ]]; then
    echo "NUM_GPUS=${NUM_GPUS}, but GPU_LIST has ${#GPUS[@]} entries: ${GPU_LIST}"
    exit 1
fi

# Export variables used in Python heredocs.
export ANNOTATION_JSON IMAGE_ROOT VIDEO_IMAGE_ROOT MERGED_JSONL SHARD_DIR NUM_GPUS
export RUN_DIR MERGED_SAMPLES SUMMARY_JSON

mkdir -p "${SHARD_DIR}" "${TASK_DIR}" "${LOG_ROOT}" "${RUN_DIR}"
rm -f "${SHARD_DIR}"/shard_*.jsonl
rm -f "${TASK_DIR}"/seedbench_local_shard_*.yaml "${TASK_DIR}/utils.py"

echo "Building local seedbench jsonl: ${MERGED_JSONL}"
python - <<'PY'
import json
import os
from pathlib import Path

annotation_json = Path(os.environ["ANNOTATION_JSON"])
image_root = Path(os.environ["IMAGE_ROOT"])
video_image_root = Path(os.environ["VIDEO_IMAGE_ROOT"])
merged_jsonl = Path(os.environ["MERGED_JSONL"])

data = json.load(annotation_json.open("r", encoding="utf-8"))

merged_jsonl.parent.mkdir(parents=True, exist_ok=True)
count = 0
with merged_jsonl.open("w", encoding="utf-8") as f:
    for q in data["questions"]:
        data_type = q["data_type"]
        if data_type == "image":
            image_path = image_root / q["data_id"]
        else:
            image_path = video_image_root / f"{q['question_type_id']}_{q['question_id']}.png"

        record = {
            "question_id": q["question_id"],
            "question": q["question"],
            "choice_a": q["choice_a"],
            "choice_b": q["choice_b"],
            "choice_c": q["choice_c"],
            "choice_d": q["choice_d"],
            "answer": q["answer"],
            "data_type": data_type,
            "image": str(image_path),
        }
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        count += 1

print(f"Merged records: {count}")
PY

echo "Preparing ${NUM_GPUS} shards from ${MERGED_JSONL} ..."
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

echo "Generating shard task YAMLs ..."
cat > "${TASK_DIR}/utils.py" <<'PY'
from PIL import Image


def _load_image(value):
    if isinstance(value, str):
        return Image.open(value).convert("RGB")
    return value.convert("RGB")


def seed_doc_to_visual(doc):
    image = doc["image"]
    if isinstance(image, list):
        return [_load_image(item) for item in image]
    return [_load_image(image)]


def seed_doc_to_text(doc):
    question = doc["question"]
    question += "\n" + f"A. {doc['choice_a']}\n"
    question += f"B. {doc['choice_b']}\n"
    question += f"C. {doc['choice_c']}\n"
    question += f"D. {doc['choice_d']}"
    return f"{question}\nAnswer with the option's letter from the given choices directly."


def seed_process_result(doc, result):
    pred = result[0].strip()
    if len(pred) > 1:
        pred = pred[0]
    answer = doc["answer"]
    data_type = doc["data_type"]
    return {
        f"seed_{data_type}": {"pred": pred, "answer": answer, "question_id": doc["question_id"]},
        "seed_all": {"pred": pred, "answer": answer, "question_id": doc["question_id"]},
    }


def seed_aggregation_result(results):
    total_count = 0
    total_correct = 0
    for result in results:
        if result["pred"].lower().strip() == result["answer"].lower().strip():
            total_correct += 1
        total_count += 1
    return total_correct / total_count if total_count else 0.0
PY

for IDX in $(seq 0 $((NUM_GPUS - 1))); do
    cat > "${TASK_DIR}/seedbench_local_shard_${IDX}.yaml" <<EOF
task: "seedbench_local_shard_${IDX}"
dataset_path: json
dataset_kwargs:
  data_files:
    test: ${SHARD_DIR}/shard_${IDX}.jsonl
output_type: generate_until
doc_to_visual: !function utils.seed_doc_to_visual
doc_to_text: !function utils.seed_doc_to_text
doc_to_target: "answer"
test_split: test
process_results: !function utils.seed_process_result
metric_list:
  - metric: seed_image
    aggregation: !function utils.seed_aggregation_result
    higher_is_better: true
  - metric: seed_video
    aggregation: !function utils.seed_aggregation_result
    higher_is_better: true
  - metric: seed_all
    aggregation: !function utils.seed_aggregation_result
    higher_is_better: true
metadata:
  - version: 0.0
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
        --tasks "seedbench_local_shard_${IDX}" \
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

echo "Aggregating seedbench accuracy from sample logs ..."
python - <<'PY'
import glob
import json
import os
from pathlib import Path

run_dir = Path(os.environ["RUN_DIR"])
merged_samples = Path(os.environ["MERGED_SAMPLES"])
summary_json = Path(os.environ["SUMMARY_JSON"])

sample_files = sorted(glob.glob(str(run_dir / "shard_*" / "**" / "*samples*.jsonl"), recursive=True))
if not sample_files:
    raise RuntimeError(f"No sample jsonl found under {run_dir}")

stats = {
    "seed_image": {"total": 0, "correct": 0},
    "seed_video": {"total": 0, "correct": 0},
    "seed_all": {"total": 0, "correct": 0},
}

def update_metric(entry, key):
    stats[key]["total"] += 1
    if entry["pred"].lower().strip() == entry["answer"].lower().strip():
        stats[key]["correct"] += 1

with merged_samples.open("w", encoding="utf-8") as merged_f:
    for sf in sample_files:
        with open(sf, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                if "seed_all" in obj:
                    update_metric(obj["seed_all"], "seed_all")
                    merged_f.write(json.dumps(obj, ensure_ascii=False) + "\n")
                if "seed_image" in obj:
                    update_metric(obj["seed_image"], "seed_image")
                if "seed_video" in obj:
                    update_metric(obj["seed_video"], "seed_video")

def acc(key):
    total = stats[key]["total"]
    correct = stats[key]["correct"]
    return correct / total if total else 0.0

summary = {
    "total_samples": stats["seed_all"]["total"],
    "seed_image_acc": acc("seed_image"),
    "seed_video_acc": acc("seed_video"),
    "seed_all_acc": acc("seed_all"),
    "sample_files": sample_files,
}

with summary_json.open("w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

print("Total samples:", summary["total_samples"])
print("Seed Image Acc:", round(summary["seed_image_acc"] * 100, 2), "%")
print("Seed Video Acc:", round(summary["seed_video_acc"] * 100, 2), "%")
print("Seed All Acc:", round(summary["seed_all_acc"] * 100, 2), "%")
print("Merged samples:", merged_samples)
print("Summary:", summary_json)
PY

echo "Done. Run dir: ${RUN_DIR}"
echo "Done. Summary file: ${SUMMARY_JSON}"
