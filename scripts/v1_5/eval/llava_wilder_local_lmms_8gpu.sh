#!/bin/bash
set -euo pipefail

# 8-GPU local LLaVA-Bench-in-the-Wild evaluation via lmms-eval (llava_wilder-style).
# - Build a merged jsonl with Question/Answer/image
# - Split into N shards
# - Run N single-GPU lmms-eval processes in parallel
# - Aggregate gpt_eval_llava_all from sample logs

REPO_ROOT="."
# External lmms-eval checkout (not vendored). Clone/install then:
#   export LMMS_ROOT=./path/to/lmms-eval
LMMS_ROOT="${LMMS_ROOT:-}"
MODEL_PATH="${MODEL_PATH:-${REPO_ROOT}/llava-v1.5-7b/}"

BENCH_ROOT="${BENCH_ROOT:-${REPO_ROOT}/playground/data/eval/llava-bench-in-the-wild}"
IMAGE_ROOT="${IMAGE_ROOT:-${BENCH_ROOT}/images}"
QUESTIONS_JSONL="${QUESTIONS_JSONL:-${BENCH_ROOT}/questions.jsonl}"
ANSWERS_GPT4_JSONL="${ANSWERS_GPT4_JSONL:-${BENCH_ROOT}/answers_gpt4.jsonl}"

GPU_LIST="${GPU_LIST:-0,1,2,3,4,5,6,7}"
CONV_TEMPLATE="${CONV_TEMPLATE:-vicuna_v1}"
NUM_GPUS="${NUM_GPUS:-8}"

OPENAI_API_URL="${OPENAI_API_URL:-https://api.openai.com/v1}"
MODEL_VERSION="${MODEL_VERSION:-gpt-4o-2024-11-20}"

RUNTIME_DIR="${BENCH_ROOT}/lmms_llava_wilder_local_runtime"
MERGED_JSONL="${RUNTIME_DIR}/llava_wilder_local.jsonl"
SHARD_DIR="${RUNTIME_DIR}/shards"
TASK_DIR="${RUNTIME_DIR}/tasks"
LOG_ROOT="${REPO_ROOT}/logs/llava_wilder_local_8gpu"
RUN_TAG="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${LOG_ROOT}/run_${RUN_TAG}"
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
if [[ ! -f "${QUESTIONS_JSONL}" ]]; then
    echo "Missing questions jsonl: ${QUESTIONS_JSONL}"
    exit 1
fi
if [[ ! -f "${ANSWERS_GPT4_JSONL}" ]]; then
    echo "Missing GPT4 answers jsonl: ${ANSWERS_GPT4_JSONL}"
    exit 1
fi
if [[ -z "${OPENAI_API_KEY:-}" ]]; then
    echo "Missing OPENAI_API_KEY for GPT judge."
    exit 1
fi

IFS=',' read -ra GPUS <<< "${GPU_LIST}"
if [[ "${#GPUS[@]}" -ne "${NUM_GPUS}" ]]; then
    echo "NUM_GPUS=${NUM_GPUS}, but GPU_LIST has ${#GPUS[@]} entries: ${GPU_LIST}"
    exit 1
fi

# Export variables used inside Python heredocs.
export QUESTIONS_JSONL ANSWERS_GPT4_JSONL IMAGE_ROOT MERGED_JSONL SHARD_DIR NUM_GPUS
export RUN_DIR MERGED_SAMPLES SUMMARY_JSON

mkdir -p "${SHARD_DIR}" "${TASK_DIR}" "${LOG_ROOT}" "${RUN_DIR}"
rm -f "${SHARD_DIR}"/shard_*.jsonl
rm -f "${TASK_DIR}"/llava_wilder_local_shard_*.yaml "${TASK_DIR}/utils.py"

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

echo "Generating task YAMLs + utils ..."
cat > "${TASK_DIR}/utils.py" <<'PY'
import base64
import os
import time
from io import BytesIO
from pathlib import Path

import numpy as np
from loguru import logger as eval_logger
from PIL import Image

from lmms_eval.llm_judge import Request, ServerConfig, get_server

NUM_SECONDS_TO_SLEEP = 5

judge_rules = (
    "We would like to request your feedback on the performance of two AI assistants in response to the user "
    "question displayed above. The user asks the question on observing an image shown to you. \n"
    "Please rate the helpfulness, relevance, accuracy, level of details of their responses. Each assistant "
    "receives an overall score on a scale of 1 to 10, where a higher score indicates better overall performance. "
    "Assume assistant 1 always receive a score of 10 and is the correct answer.\n"
    "Please first output a single line containing only two values indicating the scores for Assistant 1 and 2, "
    "respectively. The two scores are separated by a space.\n"
    "In the subsequent line, please provide a comprehensive explanation of your evaluation, avoiding any potential "
    "bias and ensuring that the order in which the responses were presented does not affect your judgment."
)

GPT_EVAL_MODEL_NAME = os.getenv("MODEL_VERSION", "gpt-4o-2024-11-20")
API_TYPE = os.getenv("LMMS_JUDGE_API_TYPE", "openai")

server_config = ServerConfig(model_name=GPT_EVAL_MODEL_NAME, temperature=0.0, max_tokens=1024)
server = get_server(server_name=API_TYPE, config=server_config)


def _load_image(doc):
    image = doc.get("image")
    if isinstance(image, str):
        return Image.open(Path(image)).convert("RGB")
    return image.convert("RGB")


def get_chat_response(base64_image, prompt, max_retries=5, wait_time=10):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
            ],
        }
    ]
    custom_config = ServerConfig(model_name=GPT_EVAL_MODEL_NAME, temperature=0.0, max_tokens=1024)
    for attempt in range(max_retries):
        try:
            # Do not pass base64 via Request.images. OpenAI provider treats string images as file paths.
            request = Request(messages=messages, images=None, config=custom_config)
            response = server.evaluate(request)
            content = response.content if response.content else ""
            return content, response.model_used
        except Exception as e:
            eval_logger.warning(f"Request failed on attempt {attempt + 1}: {e}")
            time.sleep(wait_time)
            if attempt == max_retries - 1:
                eval_logger.error(f"Failed to get response after {max_retries} attempts")
                return "", GPT_EVAL_MODEL_NAME


def image_to_base64(pil_image):
    buffered = BytesIO()
    pil_image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def parse_score(review):
    try:
        score_pair = review.split("\n")[0].strip()
        score_pair = score_pair.replace(",", " ")
        sp = score_pair.split(" ")
        if len(sp) == 2:
            return [float(sp[0]), float(sp[1])]
        eval_logger.debug(f"Can not split: {review}. Returning [-1, -1]")
        return [-1, -1]
    except Exception as e:
        eval_logger.debug(f"Error: {e}. Returning [-1, -1]")
        return [-1, -1]


def llava_process_results(doc, result):
    try:
        question = doc.get("Question", "")
        ans1 = doc.get("Answer", "")
        ans2 = result[0] if result else ""
        content = (
            f"[Question]\n{question}\n\n"
            f"[Assistant 1]\n{ans1}\n\n[End of Assistant 1]\n\n"
            f"[Assistant 2]\n{ans2}\n\n[End of Assistant 2]\n\n"
            f"[System]\n{judge_rules}\n\n"
        )
        pil_image = _load_image(doc)
        base64_image = image_to_base64(pil_image)
        review, model_name = get_chat_response(base64_image, content)
        scores = parse_score(review)
    except Exception as e:
        eval_logger.error(f"Error for Question ID: {doc.get('question_id', 'Unknown')}: {e}")
        review = "Failed to Get a Proper Review."
        model_name = "Failed Request"
        scores = [-1, -1]

    data_dict = {
        "question": question,
        "ans1": ans1,
        "ans2": ans2,
        "review": review,
        "scores": scores,
        "eval_model": model_name,
        "content": content,
    }
    return {"gpt_eval_llava_all": data_dict}


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


def llava_all_aggregation(results):
    return llava_aggregation(results, "all")


def llava_aggregation(results, category):
    try:
        scores = []
        for result in results:
            if -999 in result["scores"] or -1 in result["scores"]:
                continue
            scores.append(result["scores"])
        stats = np.asarray(scores).mean(0).tolist()
        stats = [round(x, 3) for x in stats]
        return round(stats[1] / stats[0] * 100, 1)
    except Exception as e:
        eval_logger.info(f"Error in llava_aggregation: {e}, and in category: {category}")
        return None
PY

for IDX in $(seq 0 $((NUM_GPUS - 1))); do
    cat > "${TASK_DIR}/llava_wilder_local_shard_${IDX}.yaml" <<EOF
task: "llava_wilder_local_shard_${IDX}"
dataset_path: json
dataset_kwargs:
  data_files:
    test: ${SHARD_DIR}/shard_${IDX}.jsonl
output_type: generate_until
doc_to_visual: !function utils.llava_doc_to_visual
doc_to_text: !function utils.llava_doc_to_text
doc_to_target: !function utils.llava_doc_to_target
test_split: test
process_results: !function utils.llava_process_results
process_results_use_image: true
metric_list:
  - metric: gpt_eval_llava_all
    aggregation: !function utils.llava_all_aggregation
    higher_is_better: true
metadata:
  version: 0.0
  api_type: openai
  gpt_eval_model_name: "${MODEL_VERSION}"
lmms_eval_specific_kwargs:
  default:
    pre_prompt: ""
    post_prompt: ""
EOF
done

echo "Launching ${NUM_GPUS} single-GPU processes (batch_size=1) ..."
export OPENAI_API_KEY OPENAI_API_URL MODEL_VERSION
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
        --tasks "llava_wilder_local_shard_${IDX}" \
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

echo "Aggregating gpt_eval_llava_all from sample logs ..."
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

scores = []
total = 0
with merged_samples.open("w", encoding="utf-8") as merged_f:
    for sf in sample_files:
        with open(sf, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                data = obj.get("gpt_eval_llava_all")
                if not data or "scores" not in data:
                    continue
                pair = data["scores"]
                if -999 in pair or -1 in pair:
                    continue
                scores.append(pair)
                total += 1
                merged_f.write(json.dumps(obj, ensure_ascii=False) + "\n")

if total == 0:
    raise RuntimeError("No valid gpt_eval_llava_all scores found in sample files.")

avg_a = sum(x[0] for x in scores) / total
avg_b = sum(x[1] for x in scores) / total
ratio = round(avg_b / avg_a * 100, 1)

summary = {
    "total_samples": total,
    "avg_gpt_score": avg_a,
    "avg_model_score": avg_b,
    "model_vs_gpt_ratio": ratio,
    "sample_files": sample_files,
}
with summary_json.open("w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

print("Total samples:", total)
print("Avg GPT score:", round(avg_a, 3))
print("Avg model score:", round(avg_b, 3))
print("Model/GPT ratio (%):", ratio)
print("Merged samples:", merged_samples)
print("Summary:", summary_json)
PY

echo "Done. Run dir: ${RUN_DIR}"
echo "Done. Summary file: ${SUMMARY_JSON}"
