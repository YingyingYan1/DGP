#!/usr/bin/env python3
"""
Visualize question-conditioned gated / layer-wise fused visual features (paper Fig.7).

For one image + one question, run the DGP-LLaVA model, collect per-LLM-layer router
weights over 24 ViT layers, fuse patch features in ViT space, and save one feature
map per selected LLM layer (plus an optional mean map).

Notes:
- Router weights depend on the question (glimpse query from LLM hidden states).
- Fusion here is in ViT space (weighted sum of 24-layer patches), then visualized;
  the mm_projector is not applied.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, List

# Script lives in scripts/; import this repo's llava, not another editable install.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from PIL import Image

from llava.constants import (
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    IMAGE_TOKEN_INDEX,
)
from llava.conversation import conv_templates
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init


def parse_args():
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Visualize LLM-layer-wise gated fused features."
    )
    parser.add_argument(
        "--model_path",
        default="",
        help="Path to your trained checkpoint.",
    )
    parser.add_argument("--model_base", default=None, help="Optional base model path.")
    parser.add_argument(
        "--image_path",
        default="",
        help="Single image path.",
    )
    parser.add_argument(
        "--question",
        default="Describe this image in detail.",
        help="Question text used to trigger adaptive routing.",
    )
    parser.add_argument(
        "--qa_jsonl",
        default="",
        help='Batch mode jsonl. Each line: {"image":"xxx.jpg", "question":"..."}.',
    )
    parser.add_argument(
        "--image_root",
        default=str(repo_root / "datafeature"),
        help="Root dir for images in qa_jsonl mode.",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=0,
        help="If >0, only process first N samples in qa_jsonl mode.",
    )
    parser.add_argument(
        "--conv_mode",
        default="llava_v1",
        help="Conversation template name.",
    )
    parser.add_argument(
        "--image_process_mode",
        choices=["clip", "llava_pad"],
        default="clip",
        help=(
            "clip: same preprocessing as the CLIP feature-map scripts "
            "(no square padding band); "
            "llava_pad: follow model_cfg image_aspect_ratio='pad'."
        ),
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=16)
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="Single device for model and tensors, e.g. cuda:0 or cpu.",
    )
    parser.add_argument(
        "--device_map",
        default="cuda:0",
        help="device_map passed to load_pretrained_model.",
    )

    parser.add_argument(
        "--agg",
        choices=["l2", "mean", "rms", "pca", "channel"],
        default="rms",
    )
    parser.add_argument("--channel", type=int, default=256)
    parser.add_argument(
        "--norm_mode",
        choices=["percentile", "minmax", "none"],
        default="percentile",
    )
    parser.add_argument("--percentile_low", type=float, default=5.0)
    parser.add_argument("--percentile_high", type=float, default=95.0)
    parser.add_argument("--upsample_size", type=int, default=384)
    parser.add_argument(
        "--upsample_interp",
        choices=["nearest", "bilinear", "bicubic"],
        default="nearest",
    )
    parser.add_argument(
        "--imshow_interp",
        choices=["nearest", "bilinear", "bicubic"],
        default="bilinear",
    )
    parser.add_argument("--two_step", action="store_true")
    parser.add_argument("--cmap", default="viridis")
    parser.add_argument(
        "--layers",
        default="all",
        help="Which LLM layers to visualize, e.g. 'all' or '0,8,16,24,31'.",
    )
    parser.add_argument(
        "--save_mean_map",
        action="store_true",
        help="Also save mean map over selected LLM layers.",
    )
    parser.add_argument(
        "--output_dir",
        default=str(repo_root / "datafeature-visualization" / "gated_fusion"),
        help="Directory to save fused feature maps.",
    )
    return parser.parse_args()


def normalize_percentile(
    tensor: torch.Tensor, low: float = 5.0, high: float = 95.0
) -> torch.Tensor:
    flat = tensor.flatten().float()
    low_q = torch.quantile(flat, low / 100.0)
    high_q = torch.quantile(flat, high / 100.0)
    clipped = torch.clamp(tensor.float(), min=low_q, max=high_q)
    denom = (high_q - low_q).clamp_min(1e-6)
    return (clipped - low_q) / denom


def normalize_minmax(tensor: torch.Tensor) -> torch.Tensor:
    t_min = tensor.min()
    t_max = tensor.max()
    denom = (t_max - t_min).clamp_min(1e-6)
    return (tensor - t_min) / denom


def aggregate_patch_map(
    patch_tokens: torch.Tensor, agg: str, channel_idx: int
) -> torch.Tensor:
    # patch_tokens: [N, D]
    x = patch_tokens
    if agg == "l2":
        return torch.norm(x, dim=-1)
    if agg == "mean":
        return x.mean(dim=-1)
    if agg == "rms":
        return torch.sqrt((x.float() ** 2).mean(dim=-1) + 1e-12)
    if agg == "channel":
        if channel_idx < 0 or channel_idx >= x.shape[-1]:
            raise ValueError(
                f"--channel {channel_idx} out of range [0, {x.shape[-1]-1}]"
            )
        return x[:, channel_idx].float()
    if agg == "pca":
        x_center = x - x.mean(dim=0, keepdim=True)
        U, S, _ = torch.linalg.svd(x_center.float(), full_matrices=False)
        return U[:, 0] * S[0]
    raise ValueError(f"Unknown agg: {agg}")


def upsample_map_2d(
    fmap_hw: torch.Tensor,
    target_side: int,
    mode: str = "nearest",
    two_step: bool = False,
) -> torch.Tensor:
    if target_side <= 0:
        return fmap_hw
    h, w = fmap_hw.shape[-2], fmap_hw.shape[-1]
    if h == target_side and w == target_side:
        return fmap_hw
    x = fmap_hw.unsqueeze(0).unsqueeze(0).float()
    kwargs = {}
    if mode in {"bilinear", "bicubic"}:
        kwargs["align_corners"] = False
    if two_step and target_side > h * 8:
        mid = min(max(h * 14, h * 8), target_side)
        if mid < target_side:
            x = F.interpolate(x, size=(mid, mid), mode=mode, **kwargs)
    x = F.interpolate(x, size=(target_side, target_side), mode=mode, **kwargs)
    return x.squeeze(0).squeeze(0)


def parse_layers_arg(layers_arg: str, max_layers: int) -> List[int]:
    if layers_arg.strip().lower() == "all":
        return list(range(max_layers))
    out = []
    for s in layers_arg.split(","):
        s = s.strip()
        if not s:
            continue
        idx = int(s)
        if idx < 0 or idx >= max_layers:
            raise ValueError(f"Layer idx {idx} out of range [0, {max_layers-1}]")
        out.append(idx)
    return sorted(set(out))


def build_prompt(question: str, model_config, conv_mode: str) -> str:
    qs = question
    if model_config.mm_use_im_start_end:
        qs = (
            DEFAULT_IM_START_TOKEN
            + DEFAULT_IMAGE_TOKEN
            + DEFAULT_IM_END_TOKEN
            + "\n"
            + qs
        )
    else:
        qs = DEFAULT_IMAGE_TOKEN + "\n" + qs
    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    return conv.get_prompt()


def save_map(arr, path, cmap="viridis", interp="bilinear", upsample_size=384):
    fig_inches = max(4.0, upsample_size / 150.0)
    plt.figure(figsize=(fig_inches, fig_inches))
    plt.imshow(arr, cmap=cmap, interpolation=interp)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0)
    plt.close()


def parse_qa_jsonl(path: str, image_root: str, max_samples: int = 0):
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            image_name = obj.get("image", "").strip()
            question = obj.get("question", "").strip()
            if not image_name or not question:
                continue
            image_path = (
                image_name
                if os.path.isabs(image_name)
                else os.path.join(image_root, image_name)
            )
            items.append(
                {"image": image_name, "image_path": image_path, "question": question}
            )
            if max_samples > 0 and len(items) >= max_samples:
                break
    return items


def collect_sample_weights_and_features(
    model,
    tokenizer,
    image_processor,
    image_path: str,
    question: str,
    conv_mode: str,
    temperature: float,
    top_p: float,
    num_beams: int,
    max_new_tokens: int,
    image_process_mode: str,
):
    image = Image.open(image_path).convert("RGB")
    if image_process_mode == "clip":
        # Match visualize_vit23_feature_map.py / visualize_vit_avg_feature_map.py
        # (no pad-induced borders on non-square images).
        image_tensor = image_processor(images=image, return_tensors="pt")["pixel_values"][0]
    else:
        image_tensor = process_images([image], image_processor, model.config)[0]
    prompt = build_prompt(question, model.config, conv_mode)
    model_device = next(model.parameters()).device
    input_ids = (
        tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        .unsqueeze(0)
        .to(model_device)
    )

    lf = model.model.layerwise_fusion
    lf.enable_router_weight_collection()
    with torch.inference_mode():
        _ = model.generate(
            input_ids,
            images=image_tensor.unsqueeze(0).to(
                dtype=torch.float16, device=model_device, non_blocking=True
            ),
            image_sizes=[image.size],
            do_sample=(temperature > 0),
            temperature=temperature,
            top_p=top_p,
            num_beams=num_beams,
            max_new_tokens=max_new_tokens,
            use_cache=True,
        )
    lf.finalize_current_sample()
    all_weights = lf.get_all_router_weights()
    lf.disable_router_weight_collection()
    if not all_weights:
        raise RuntimeError("No router weights collected from this sample.")
    sample_weights: Dict = all_weights[-1]

    vision_tower = model.get_vision_tower()
    with torch.no_grad():
        patch_features, _ = vision_tower.get_all_layer_features(
            image_tensor.unsqueeze(0).to(
                device=vision_tower.device, dtype=vision_tower.dtype
            )
        )
    if len(patch_features) < 24:
        raise RuntimeError(f"Expected 24 ViT layers, got {len(patch_features)}")
    stacked = torch.stack([p[0].float().cpu() for p in patch_features[:24]], dim=0)
    return sample_weights, stacked


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if not args.qa_jsonl and not args.image_path:
        raise ValueError("Provide --image_path or --qa_jsonl.")

    disable_torch_init()
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, _ = load_pretrained_model(
        args.model_path,
        args.model_base,
        model_name,
        device_map=args.device_map,
        device=args.device,
    )
    model.eval()

    if not (
        hasattr(model.model, "layerwise_fusion")
        and model.model.layerwise_fusion is not None
    ):
        raise RuntimeError("Loaded model has no layerwise_fusion module.")

    if args.qa_jsonl:
        samples = parse_qa_jsonl(args.qa_jsonl, args.image_root, args.max_samples)
        if not samples:
            raise RuntimeError(f"No valid records found in qa_jsonl: {args.qa_jsonl}")
    else:
        samples = [{"image_path": args.image_path, "question": args.question}]

    num_llm_layers = model.config.num_hidden_layers
    selected_layers = parse_layers_arg(args.layers, num_llm_layers)
    print(f"Selected LLM layers: {selected_layers}")
    print(f"Total samples: {len(samples)}")

    processed, skipped = 0, 0
    for idx, sample in enumerate(samples, start=1):
        image_path = sample["image_path"]
        question = sample["question"]
        image_base = os.path.splitext(os.path.basename(image_path))[0]
        print(f"\n[{idx}/{len(samples)}] Processing {image_base}")
        if not os.path.exists(image_path):
            print(f"[warn] missing image, skip: {image_path}")
            skipped += 1
            continue

        try:
            sample_weights, stacked = collect_sample_weights_and_features(
                model=model,
                tokenizer=tokenizer,
                image_processor=image_processor,
                image_path=image_path,
                question=question,
                conv_mode=args.conv_mode,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
                image_process_mode=args.image_process_mode,
            )
        except Exception as e:
            print(f"[warn] failed sample {image_base}: {e}")
            skipped += 1
            continue

        fused_maps = []
        sample_dir = os.path.join(args.output_dir, image_base)
        os.makedirs(sample_dir, exist_ok=True)

        for l in selected_layers:
            if l not in sample_weights:
                print(f"[warn] LLM layer {l} has no collected weights; skip.")
                continue
            w = sample_weights[l]
            if isinstance(w, torch.Tensor):
                w = w[0].detach().float().cpu()
            else:
                w = torch.tensor(w, dtype=torch.float32)
            if w.numel() != 24:
                print(f"[warn] LLM layer {l} weights numel={w.numel()} != 24; skip.")
                continue
            w = w / (w.sum().clamp_min(1e-8))
            fused_tokens = torch.sum(w[:, None, None] * stacked, dim=0)
            score = aggregate_patch_map(fused_tokens, args.agg, args.channel)

            n = score.numel()
            side = int(math.sqrt(n))
            if side * side != n:
                raise ValueError(f"Patch count {n} is not a square number.")
            fmap = score.view(side, side).float()
            if args.norm_mode == "percentile":
                fmap = normalize_percentile(fmap, args.percentile_low, args.percentile_high)
            elif args.norm_mode == "minmax":
                fmap = normalize_minmax(fmap)
            fmap = upsample_map_2d(
                fmap, args.upsample_size, args.upsample_interp, args.two_step
            )
            arr = fmap.numpy()
            fused_maps.append(arr)

            out_name = f"{image_base}_llm{l:02d}_fused_{args.agg}"
            if args.agg == "channel":
                out_name += f"_c{args.channel}"
            out_path = os.path.join(sample_dir, out_name + ".png")
            save_map(
                arr,
                out_path,
                cmap=args.cmap,
                interp=args.imshow_interp,
                upsample_size=args.upsample_size,
            )

        if args.save_mean_map and fused_maps:
            mean_arr = torch.tensor(fused_maps).mean(dim=0).numpy()
            out_name = f"{image_base}_llm_mean_fused_{args.agg}"
            if args.agg == "channel":
                out_name += f"_c{args.channel}"
            out_path = os.path.join(sample_dir, out_name + ".png")
            save_map(
                mean_arr,
                out_path,
                cmap=args.cmap,
                interp=args.imshow_interp,
                upsample_size=args.upsample_size,
            )

        processed += 1
        print(f"[ok] {image_base} done, saved in: {sample_dir}")

    print(f"\nDone. processed={processed}, skipped={skipped}, total={len(samples)}")
    print(
        "Note: fusion is question-conditioned and sample-adaptive; "
        "changing the question or image changes all layer weights and maps."
    )


if __name__ == "__main__":
    main()
