from PIL import Image

from lmms_eval.tasks.vqav2.utils import (
    vqav2_process_results_val,
    vqav2_process_results_test,
)


def vqav2_local_doc_to_visual(doc, lmms_eval_specific_kwargs=None):
    image_path = doc.get("image_path")
    if not image_path and lmms_eval_specific_kwargs is not None:
        image_root = lmms_eval_specific_kwargs.get("image_root", "")
        image_name = doc.get("image", "")
        if image_root and image_name:
            image_path = f"{image_root.rstrip('/')}/{image_name}"
    if not image_path:
        raise ValueError("Missing image_path for VQAv2 local doc.")
    return [Image.open(image_path).convert("RGB")]


def vqav2_local_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    pre_prompt = ""
    post_prompt = "\nAnswer the question using a single word or phrase."
    if lmms_eval_specific_kwargs is not None:
        pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", pre_prompt)
        post_prompt = lmms_eval_specific_kwargs.get("post_prompt", post_prompt)
    return f"{pre_prompt}{doc['question']}{post_prompt}"


def vqav2_local_doc_to_target(doc):
    answers = doc.get("answers", [])
    if answers:
        return answers[0].get("answer", "")
    return ""

