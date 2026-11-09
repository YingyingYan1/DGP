import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path

from PIL import Image
import math


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)

    # Optional: collect layer-wise router weights (same format as model_vqa_loader_with_router_stats)
    collect_router_stats = getattr(args, 'router_stats_file', None) and len(os.path.expanduser(args.router_stats_file or '').strip()) > 0
    if collect_router_stats:
        if hasattr(model.model, 'layerwise_fusion') and model.model.layerwise_fusion is not None:
            model.model.layerwise_fusion.enable_router_weight_collection()
            print("[RouterStats] Layer-wise router weight collection enabled (ScienceQA)")
        else:
            print("[RouterStats] Warning: no layerwise_fusion module; cannot collect router weights")
            collect_router_stats = False

    questions = json.load(open(os.path.expanduser(args.question_file), "r"))
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")
    if collect_router_stats:
        os.makedirs(os.path.dirname(os.path.expanduser(args.router_stats_file)), exist_ok=True)
    per_sample_weights = []
    for i, line in enumerate(tqdm(questions)):
        idx = line["id"]
        question = line['conversations'][0]
        qs = question['value'].replace('<image>', '').strip()
        cur_prompt = qs

        if 'image' in line:
            image_file = line["image"]
            image = Image.open(os.path.join(args.image_folder, image_file))
            image_tensor = process_images([image], image_processor, model.config)[0]
            images = image_tensor.unsqueeze(0).half().cuda()
            image_sizes = [image.size]
            if getattr(model.config, 'mm_use_im_start_end', False):
                qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN + '\n' + qs
            cur_prompt = '<image>' + '\n' + cur_prompt
        else:
            images = None
            image_sizes = None

        if args.single_pred_prompt:
            qs = qs + '\n' + "Answer with the option's letter from the given choices directly."
            cur_prompt = cur_prompt + '\n' + "Answer with the option's letter from the given choices directly."

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=images,
                image_sizes=image_sizes,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                max_new_tokens=1024,
                use_cache=True,
            )

        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

        if collect_router_stats and images is not None:
            model.model.layerwise_fusion.finalize_current_sample()
            all_weights = model.model.layerwise_fusion.get_all_router_weights()
            if all_weights:
                current_sample_weight = all_weights[-1]
                sample_weight_dict = {
                    'question_id': idx,
                    'llm_router_weights': {},
                    'pre_router_weights': None
                }
                for layer_key, w in current_sample_weight.items():
                    if isinstance(w, torch.Tensor):
                        weight_list = w[0].detach().cpu().tolist()
                    else:
                        weight_list = w
                    if layer_key == 'pre_router':
                        sample_weight_dict['pre_router_weights'] = weight_list
                    else:
                        sample_weight_dict['llm_router_weights'][layer_key] = weight_list
                per_sample_weights.append(sample_weight_dict)

        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({"question_id": idx,
                                   "prompt": cur_prompt,
                                   "text": outputs,
                                   "answer_id": ans_id,
                                   "model_id": model_name,
                                   "metadata": {}}) + "\n")
        ans_file.flush()
    ans_file.close()

    if collect_router_stats and per_sample_weights:
        stats = model.model.layerwise_fusion.get_router_weights_statistics()
        if stats is not None:
            has_pre_router = 'pre_router_mean' in stats
            output_data = {
                'metadata': {
                    'num_samples': int(stats['num_samples']),
                    'num_llm_layers': int(stats['num_llm_layers']),
                    'num_vit_layers': int(stats['num_vit_layers']),
                    'has_pre_router': has_pre_router,
                    'model_path': model_path,
                    'dataset': args.question_file
                },
                'llm_router': {
                    'mean_weights': stats['mean'].tolist(),
                    'std_weights': stats['std'].tolist(),
                },
                'per_sample_weights': per_sample_weights
            }
            if has_pre_router:
                output_data['pre_router'] = {
                    'mean_weights': stats['pre_router_mean'].tolist(),
                    'std_weights': stats['pre_router_std'].tolist(),
                }
            router_stats_file = os.path.expanduser(args.router_stats_file)
            with open(router_stats_file, 'w') as f:
                json.dump(output_data, f, indent=2)
            print(f"[RouterStats] Saved: {router_stats_file} (num_samples={len(per_sample_weights)})")
        model.model.layerwise_fusion.disable_router_weight_collection()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.json")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--router-stats-file", type=str, default=None,
                        help="If set, collect layer-wise router weights during inference and write mean + per-sample stats to this JSON")
    parser.add_argument("--conv-mode", type=str, default="llava_v0")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--answer-prompter", action="store_true")
    parser.add_argument("--single-pred-prompt", action="store_true")
    args = parser.parse_args()

    eval_model(args)
