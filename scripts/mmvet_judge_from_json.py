import argparse
import json
from typing import Dict, Any, List

from tqdm import tqdm

try:
    from lmms_eval.tasks.mmvet.utils import mmvet_process_results, mmvet_aggregate_results
except ImportError as err:
    raise ImportError(
        "mmvet judge requires an external lmms-eval install. "
        "Clone https://github.com/EvolvingLMMs-Lab/lmms-eval, pip install -e ., "
        "and ensure it is on PYTHONPATH (or set LMMS_ROOT and export PYTHONPATH=$LMMS_ROOT)."
    ) from err


def load_predictions(path: str) -> Dict[str, str]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("Prediction file must be a JSON object: {question_id: answer}")
    return {str(k): str(v) for k, v in data.items()}


def build_doc_map_from_local(mmvet_gt_json: str) -> Dict[str, Dict[str, Any]]:
    with open(mmvet_gt_json, "r", encoding="utf-8") as f:
        raw = json.load(f)

    if not isinstance(raw, dict):
        raise ValueError("MMVet GT json should be an object keyed by question_id.")

    doc_map: Dict[str, Dict[str, Any]] = {}
    for qid, item in raw.items():
        capability = item.get("capability", "")
        if isinstance(capability, list):
            capability = ",".join(capability)
        doc_map[str(qid)] = {
            "question_id": str(qid),
            "question": item.get("question", ""),
            "answer": item.get("answer", ""),
            "capability": capability,
        }
    return doc_map


def main() -> None:
    parser = argparse.ArgumentParser(description="MMVet judge scoring from prediction JSON.")
    parser.add_argument("--pred-file", required=True, help="Path to prediction JSON.")
    parser.add_argument("--output-file", required=True, help="Path to save judged results JSON.")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of samples (0 = all).")
    parser.add_argument(
        "--mmvet-gt-json",
        default="./playground/data/eval/mm-vet/mm-vet/mm-vet.json",
        help="Local MMVet ground-truth JSON path.",
    )
    args = parser.parse_args()

    preds = load_predictions(args.pred_file)
    doc_map = build_doc_map_from_local(args.mmvet_gt_json)

    results: List[Dict[str, Any]] = []
    missing = 0
    keys = list(preds.keys())
    if args.limit and args.limit > 0:
        keys = keys[: args.limit]

    for qid in tqdm(keys, desc="Judging"):
        doc = doc_map.get(qid)
        if doc is None:
            missing += 1
            continue
        pred = preds[qid]
        judged = mmvet_process_results(doc, [pred])
        results.append(judged["gpt_eval_score"])

    if missing:
        print(f"Warning: {missing} predictions did not match dataset question_id.")

    overall = mmvet_aggregate_results(results) if results else 0.0
    payload = {
        "overall_score": overall,
        "results": results,
    }
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"Overall Score: {overall:.2f}")
    print(f"Saved: {args.output_file}")


if __name__ == "__main__":
    main()
