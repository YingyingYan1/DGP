"""
TextVQA evaluation with layer-wise router weight statistics.

Collects per-sample router weights during inference and writes aggregated router weight statistics.
"""
import argparse
import torch
import os
import json
import numpy as np
from tqdm import tqdm
import shortuuid

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from torch.utils.data import Dataset, DataLoader

from PIL import Image
import math


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


class CustomDataset(Dataset):
    def __init__(self, questions, image_folder, tokenizer, image_processor, model_config, conv_mode):
        self.questions = questions
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model_config = model_config
        self.conv_mode = conv_mode

    def __getitem__(self, index):
        line = self.questions[index]
        image_file = line["image"]
        qs = line["text"]
        if self.model_config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        image = Image.open(os.path.join(self.image_folder, image_file)).convert('RGB')
        image_tensor = process_images([image], self.image_processor, self.model_config)[0]

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')

        return input_ids, image_tensor, image.size

    def __len__(self):
        return len(self.questions)


def collate_fn(batch):
    input_ids, image_tensors, image_sizes = zip(*batch)
    input_ids = torch.stack(input_ids, dim=0)
    image_tensors = torch.stack(image_tensors, dim=0)
    return input_ids, image_tensors, image_sizes


def create_data_loader(questions, image_folder, tokenizer, image_processor, model_config, conv_mode, batch_size=1, num_workers=4):
    assert batch_size == 1, "batch_size must be 1"
    dataset = CustomDataset(questions, image_folder, tokenizer, image_processor, model_config, conv_mode)
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, collate_fn=collate_fn)
    return data_loader


def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)

    # Enable router weight collection (layerwise vision fusion)
    if hasattr(model.model, 'layerwise_fusion') and model.model.layerwise_fusion is not None:
        model.model.layerwise_fusion.enable_router_weight_collection()
        print("[RouterStats] Layer-wise router weight collection enabled")
    else:
        print("[RouterStats] Warning: no layerwise_fusion module; cannot collect router weights")
        return

    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    
    # Optional cap on number of samples (quick smoke test)
    if args.num_samples is not None and args.num_samples > 0:
        questions = questions[:args.num_samples]
        print(f"[Quick test] Evaluating only the first {len(questions)} samples")
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    # Output path for aggregated router weight statistics
    router_stats_file = args.router_stats_file
    os.makedirs(os.path.dirname(router_stats_file), exist_ok=True)

    conv_mode = args.conv_mode
    if 'plain' in model_name and 'finetune' not in model_name.lower() and 'mmtag' not in conv_mode:
        conv_mode = conv_mode + '_mmtag'
        print(f'It seems that this is a plain model, but it is not using a mmtag prompt, auto switching to {conv_mode}.')

    data_loader = create_data_loader(questions, args.image_folder, tokenizer, image_processor, model.config, conv_mode)

    # Per-sample router weight records (for merge_router_stats.py)
    per_sample_weights = []

    # When LLAVA_DEBUG_SEQ=1, print image, question, per-layer router, and hidden-state traces (see llava_arch)
    debug_seq = os.environ.get("LLAVA_DEBUG_SEQ", "0") == "1"
    debug_seq_max_samples = 2  # Match _LLAVA_DEBUG_SEQ_MAX_BATCHES in llava_arch

    for (input_ids, image_tensor, image_sizes), line in tqdm(zip(data_loader, questions), total=len(questions), desc="Evaluating"):
        idx = line["question_id"]
        cur_prompt = line["text"]

        # Align eval-side DEBUG_SEQ prints with llava_arch [DEBUG_SEQ] traces
        if debug_seq and len(per_sample_weights) < debug_seq_max_samples:
            image_file = line.get("image", "")
            print("\n[DEBUG_SEQ] Eval sample: question_id=%s image=%s" % (idx, image_file))
            print("[DEBUG_SEQ] Question (text): %s" % (cur_prompt[:200] + ("..." if len(cur_prompt) > 200 else "")))

        input_ids = input_ids.to(device='cuda', non_blocking=True)

        # Rebuild full prompt (system + user) for optional decode/debug
        qs = cur_prompt
        if model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs
        
        conv = conv_templates[conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        full_prompt = conv.get_prompt()
        
        # Decode input_ids to text for verification (best-effort)
        decoded_input = None
        try:
            # Only decode in-vocab token IDs (image / glimpse indices may be out of range)
            valid_ids = input_ids[0].cpu().clone()
            vocab_size = len(tokenizer)
            valid_mask = (valid_ids >= 0) & (valid_ids < vocab_size)
            if valid_mask.any():
                valid_ids_filtered = valid_ids[valid_mask]
                decoded_input = tokenizer.decode(valid_ids_filtered, skip_special_tokens=False)
            else:
                decoded_input = "[decode skipped: all token IDs out of vocab range]"
        except Exception as e:
            decoded_input = f"[decode failed: {str(e)}]"

        # Optional: full input embedding stats (mostly commented out below)
        with torch.inference_mode():
            # Token embeddings for the text stream (image / glimpse tokens may be OOR for embed lookup)
            token_embeds = None
            invalid_positions_mask = None  # Positions with OOR token IDs
            vocab_size = len(tokenizer)
            valid_mask = (input_ids >= 0) & (input_ids < vocab_size)
            if valid_mask.all():
                # All IDs in embedding range
                if hasattr(model.model, 'embed_tokens'):
                    token_embeds = model.model.embed_tokens(input_ids)
                elif hasattr(model, 'get_input_embeddings'):
                    token_embeds = model.get_input_embeddings()(input_ids)
            else:
                # Mixed valid / invalid IDs: substitute pad for OOR positions for embed lookup only
                valid_ids = input_ids.clone()
                # Replace invalid IDs with pad_token_id (or 0) for embedding lookup
                pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
                valid_ids[~valid_mask] = pad_token_id
                if hasattr(model.model, 'embed_tokens'):
                    token_embeds = model.model.embed_tokens(valid_ids)
                elif hasattr(model, 'get_input_embeddings'):
                    token_embeds = model.get_input_embeddings()(valid_ids)
                invalid_positions_mask = ~valid_mask[0].cpu()  # [seq_len] — mask invalid embed positions
            
            # Full embedding dump (commented; enable for deep debugging)
            # print(f"\n{'='*80}")
            # print(f"Sample ID: {idx}")
            # print(f"{'='*80}")
            # print(f"\n[Raw question]\n{cur_prompt}")
            # print(f"\n[Full prompt (incl. system)]\n{full_prompt}")
            # print(f"\n[Decoded input_ids]\n{decoded_input}")
            
            # if token_embeds is not None:
            #     print(f"\n[Input embedding tensor]")
            #     print(f"  shape: {tuple(token_embeds.shape)}")
            #     print(f"  dtype: {token_embeds.dtype}")
            #     print(f"  mean: {token_embeds.mean().item():.6f}")
            #     print(f"  std: {token_embeds.std().item():.6f}")
            #     print(f"  min: {token_embeds.min().item():.6f}")
            #     print(f"  max: {token_embeds.max().item():.6f}")
            #     print(f"  nonzero count: {(token_embeds != 0).sum().item()}/{token_embeds.numel()}")
            # else:
            #     print(f"\nWarning: token embeddings unavailable")

            output_ids = model.generate(
                input_ids,
                images=image_tensor.to(dtype=torch.float16, device='cuda', non_blocking=True),
                image_sizes=image_sizes,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
                use_cache=True)

        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        
        # print(f"\nAnswer: {outputs}")
        # print(f"{'='*80}\n")

        model.model.layerwise_fusion.finalize_current_sample()

        all_weights = model.model.layerwise_fusion.get_all_router_weights()
        if all_weights:
            current_sample_weight = all_weights[-1]  # Latest record == current sample
            # Serializable dict: pre-router vs per-LLM-layer router weights
            sample_weight_dict = {
                'question_id': idx,
                'llm_router_weights': {},  # Per LLM layer (layer-wise router)
                'pre_router_weights': None  # Optional router before first LLM layer
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
    
    ans_file.close()

    print("\n" + "=" * 70)
    print("Computing layer-wise router weight statistics...")
    print("=" * 70)
    
    stats = model.model.layerwise_fusion.get_router_weights_statistics()
    
    if stats is None:
        print("Warning: no router weight data collected")
        return
    
    print(f"num_samples: {stats['num_samples']}")
    print(f"num_llm_layers: {stats['num_llm_layers']}")
    print(f"num_vit_layers (ViT depth): {stats['num_vit_layers']}")
    has_pre_router = 'pre_router_mean' in stats
    print(f"pre-router (before LLM layer 0): {'present' if has_pre_router else 'absent'}")
    
    # Persist aggregates and per-sample router weights
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
            'mean_weights': stats['mean'].tolist(),  # [num_llm_layers, num_vit_layers]
            'std_weights': stats['std'].tolist(),
        },
        'per_sample_weights': per_sample_weights
    }
    
    # Optional pre-router block
    if has_pre_router:
        output_data['pre_router'] = {
            'mean_weights': stats['pre_router_mean'].tolist(),  # [num_vit_layers]
            'std_weights': stats['pre_router_std'].tolist(),
        }
    
    with open(router_stats_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\nRouter weight statistics saved to: {router_stats_file}")
    
    if has_pre_router:
        print("\n" + "=" * 70)
        print("Pre-router mean weights (before LLM layer 0)")
        print("=" * 70)
        pre_router_mean = stats['pre_router_mean']
        top_3_vit = np.argsort(pre_router_mean)[-3:][::-1]
        print(f"Pre-router: top-3 ViT layer indices = {list(top_3_vit)}, weights = {[f'{pre_router_mean[i]:.4f}' for i in top_3_vit]}")
        print(f"All ViT layer weights: {[f'{w:.4f}' for w in pre_router_mean]}")
    
    print("\n" + "=" * 70)
    print("Per-LLM-layer router mean weights (layer-wise router)")
    print("=" * 70)
    mean_weights = stats['mean']
    for llm_layer in [0, 8, 16, 24, 31]:
        if llm_layer < stats['num_llm_layers']:
            weights = mean_weights[llm_layer]
            top_3_vit = np.argsort(weights)[-3:][::-1]
            print(f"LLM Layer {llm_layer:2d}: Top-3 ViT layers = {list(top_3_vit)}, weights = {[f'{weights[i]:.4f}' for i in top_3_vit]}")
    
    model.model.layerwise_fusion.disable_router_weight_collection()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--router-stats-file", type=str, default="router_stats.json", 
                        help="Path to write aggregated layer-wise router weight statistics (JSON)")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--num-samples", type=int, default=None,
                        help="Evaluate only the first N samples (quick test)")
    args = parser.parse_args()

    eval_model(args)

