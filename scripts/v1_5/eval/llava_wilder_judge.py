#!/usr/bin/env python3
import argparse
import base64
import json
import time
from io import BytesIO
from pathlib import Path

import requests
from PIL import Image

JUDGE_RULES = (
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


def parse_args():
    parser = argparse.ArgumentParser(description="Judge llava_wilder predictions with OpenAI-compatible API.")
    parser.add_argument("--predictions", type=str, required=True, help="Inference-only predictions jsonl")
    parser.add_argument("--output-jsonl", type=str, required=True, help="Per-sample judged output jsonl")
    parser.add_argument("--summary-json", type=str, required=True, help="Summary metrics json")
    parser.add_argument("--api-url", type=str, default=None, help="OpenAI-compatible chat completions endpoint")
    parser.add_argument("--api-key", type=str, default=None, help="API key, or read OPENAI_API_KEY")
    parser.add_argument("--model", type=str, default=None, help="Judge model id, or read MODEL_VERSION")
    parser.add_argument("--timeout", type=int, default=180, help="HTTP timeout seconds")
    parser.add_argument("--retries", type=int, default=8, help="Outer retries for each sample")
    parser.add_argument("--retry-delay", type=float, default=8.0, help="Sleep seconds between retries")
    parser.add_argument("--max-tokens", type=int, default=512, help="Judge max output tokens")
    parser.add_argument("--image-max-edge", type=int, default=768, help="Resize long edge before base64")
    parser.add_argument("--image-quality", type=int, default=60, help="JPEG quality for base64 image")
    return parser.parse_args()


def image_to_base64(path: str, max_edge: int, quality: int) -> str:
    img = Image.open(path).convert("RGB")
    long_edge = max(img.size)
    if long_edge > max_edge:
        scale = max_edge / float(long_edge)
        new_size = (max(1, int(img.size[0] * scale)), max(1, int(img.size[1] * scale)))
        img = img.resize(new_size)
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=quality, optimize=True)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def build_prompt(question: str, ans1: str, ans2: str) -> str:
    return (
        f"[Question]\n{question}\n\n"
        f"[Assistant 1]\n{ans1}\n\n[End of Assistant 1]\n\n"
        f"[Assistant 2]\n{ans2}\n\n[End of Assistant 2]\n\n"
        f"[System]\n{JUDGE_RULES}\n\n"
    )


def parse_score(review: str):
    try:
        first = review.split("\n")[0].strip().replace(",", " ")
        parts = [p for p in first.split(" ") if p]
        if len(parts) >= 2:
            return [float(parts[0]), float(parts[1])]
    except Exception:
        pass
    return [-1.0, -1.0]


def call_judge(api_url: str, api_key: str, model: str, prompt: str, image_b64: str, timeout: int, max_tokens: int):
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
                ],
            }
        ],
        "temperature": 0.0,
        "max_tokens": max_tokens,
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    resp = requests.post(api_url, headers=headers, json=payload, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"], data.get("model", model)


def main():
    args = parse_args()

    import os
    api_url = args.api_url or os.getenv("OPENAI_API_URL")
    if not api_url:
        raise SystemExit("Missing API url. Set --api-url or OPENAI_API_URL.")

    api_key = args.api_key
    if not api_key:
        api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        raise SystemExit("Missing API key. Set --api-key or OPENAI_API_KEY.")

    model = args.model
    if not model:
        model = os.getenv("MODEL_VERSION", "gpt-4o-2024-11-20")

    pred_path = Path(args.predictions)
    out_jsonl = Path(args.output_jsonl)
    summary_json = Path(args.summary_json)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    summary_json.parent.mkdir(parents=True, exist_ok=True)

    records = []
    with pred_path.open("r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))

    total = 0
    valid = 0
    sum_ref = 0.0
    sum_model = 0.0

    with out_jsonl.open("w", encoding="utf-8") as out_f:
        for idx, rec in enumerate(records, 1):
            total += 1
            question = rec.get("question", "")
            ans1 = rec.get("answer_ref", "")
            ans2 = rec.get("model_answer", "")
            image_path = rec.get("image_path", "")
            prompt = build_prompt(question, ans1, ans2)

            review = ""
            model_used = model
            scores = [-1.0, -1.0]
            err = None

            try:
                image_b64 = image_to_base64(image_path, args.image_max_edge, args.image_quality)
                for attempt in range(args.retries):
                    try:
                        review, model_used = call_judge(
                            api_url=api_url,
                            api_key=api_key,
                            model=model,
                            prompt=prompt,
                            image_b64=image_b64,
                            timeout=args.timeout,
                            max_tokens=args.max_tokens,
                        )
                        scores = parse_score(review)
                        break
                    except Exception as e:
                        err = str(e)
                        if attempt == args.retries - 1:
                            break
                        time.sleep(args.retry_delay)
            except Exception as e:
                err = str(e)

            if scores[0] > 0 and scores[1] >= 0:
                valid += 1
                sum_ref += scores[0]
                sum_model += scores[1]

            out = {
                **rec,
                "review": review,
                "scores": scores,
                "eval_model": model_used,
                "error": err,
            }
            out_f.write(json.dumps(out, ensure_ascii=False) + "\n")
            if idx % 10 == 0:
                print(f"[judge] processed {idx}/{len(records)}")

    avg_ref = (sum_ref / valid) if valid else 0.0
    avg_model = (sum_model / valid) if valid else 0.0
    ratio = (avg_model / avg_ref * 100.0) if avg_ref > 0 else 0.0

    summary = {
        "total_samples": total,
        "valid_scored_samples": valid,
        "avg_ref_score": round(avg_ref, 4),
        "avg_model_score": round(avg_model, 4),
        "model_vs_ref_ratio": round(ratio, 2),
        "predictions": str(pred_path),
        "judged_output": str(out_jsonl),
    }
    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
