#!/usr/bin/env python3
"""
Visualize patch features averaged over CLIP ViT layers 1–24.

- Normalization: percentile clip (p5–p95 by default), then linear stretch to [0, 1].
- Display: bilinear upsample + bilinear imshow by default.
- Aggregation: l2 / mean / pca (1st principal component) / channel.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPImageProcessor, CLIPVisionModel


def parse_args():
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Visualize averaged patch features from CLIP ViT layers 1-24."
    )
    parser.add_argument(
        "--input_path",
        default=str(repo_root / "datafeature"),
        help="Input image path or a directory containing images.",
    )
    parser.add_argument(
        "--output_dir",
        default=str(repo_root / "datafeature-visualization"),
        help="Directory to save the visualization.",
    )
    parser.add_argument(
        "--model_path",
        default=str(repo_root / "clip-vit-large-patch14-336"),
        help="Local CLIP vision model path.",
    )
    parser.add_argument(
        "--agg",
        choices=["l2", "mean", "pca", "channel"],
        default="pca",
        help="How to reduce patch features [N,D] to length-N scalars.",
    )
    parser.add_argument(
        "--channel",
        type=int,
        default=0,
        help="When --agg channel: which feature dimension index to plot (0..D-1).",
    )
    parser.add_argument(
        "--percentile_low",
        type=float,
        default=5.0,
        help="Lower percentile for clipping (0-100), default 5.",
    )
    parser.add_argument(
        "--percentile_high",
        type=float,
        default=95.0,
        help="Upper percentile for clipping (0-100), default 95.",
    )
    parser.add_argument(
        "--norm_mode",
        choices=["percentile", "minmax", "none"],
        default="percentile",
        help="Normalization mode before visualization.",
    )
    parser.add_argument(
        "--shared_norm_from",
        default="",
        help="Path to .json/.npy providing shared vmin/vmax for fair comparison.",
    )
    parser.add_argument(
        "--upsample_size",
        type=int,
        default=336,
        help="Upsample the HxW map to this square side (default 336).",
    )
    parser.add_argument(
        "--upsample_interp",
        choices=["nearest", "bicubic", "bilinear"],
        default="bilinear",
        help="Interpolation mode for upsampling.",
    )
    parser.add_argument(
        "--imshow_interp",
        choices=["nearest", "bicubic", "bilinear"],
        default="bilinear",
        help="Interpolation used by imshow.",
    )
    parser.add_argument(
        "--two_step",
        action="store_true",
        help="Enable two-step upsampling (off by default).",
    )
    parser.add_argument(
        "--cmap",
        default="viridis",
        help="Matplotlib colormap.",
    )
    parser.add_argument(
        "--suffix",
        default="vit23",
        help="Suffix for output filename.",
    )
    return parser.parse_args()


def normalize_percentile(
    tensor: torch.Tensor,
    low: float = 5.0,
    high: float = 95.0,
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


def load_shared_norm(path: str):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".json":
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        return float(obj["vmin"]), float(obj["vmax"])
    if ext == ".npy":
        arr = np.load(path, allow_pickle=True)
        if isinstance(arr, np.ndarray) and arr.shape == ():
            obj = arr.item()
            return float(obj["vmin"]), float(obj["vmax"])
        if isinstance(arr, np.ndarray) and arr.size >= 2:
            flat = arr.reshape(-1)
            return float(flat[0]), float(flat[1])
    raise ValueError(f"Unsupported shared_norm_from format: {path}")


def apply_shared_norm(tensor: torch.Tensor, vmin: float, vmax: float) -> torch.Tensor:
    t = tensor.float()
    vmin_t = torch.tensor(vmin, dtype=t.dtype, device=t.device)
    vmax_t = torch.tensor(vmax, dtype=t.dtype, device=t.device)
    clipped = torch.clamp(t, min=vmin_t.item(), max=vmax_t.item())
    denom = (vmax_t - vmin_t).clamp_min(1e-6)
    return (clipped - vmin_t) / denom


def aggregate_patch_map(
    patch_tokens: torch.Tensor,
    agg: str,
    channel_idx: int,
) -> torch.Tensor:
    """patch_tokens: [B, N, D] -> [N] (single batch)."""
    x = patch_tokens.squeeze(0)  # [N, D]
    if agg == "l2":
        return torch.norm(x, dim=-1)
    if agg == "mean":
        return x.mean(dim=-1)
    if agg == "channel":
        if channel_idx < 0 or channel_idx >= x.shape[-1]:
            raise ValueError(f"--channel {channel_idx} out of range [0, {x.shape[-1] - 1}]")
        return x[:, channel_idx].float()
    if agg == "pca":
        return pca_first_component_scores(x)
    raise ValueError(f"Unknown agg: {agg}")


def pca_first_component_scores(X: torch.Tensor) -> torch.Tensor:
    """X: [N, D]. Score of each patch on the first principal component (centered SVD)."""
    n = X.shape[0]
    if n < 2:
        return X.abs().mean(dim=-1)
    Xc = X - X.mean(dim=0, keepdim=True)
    U, S, _Vh = torch.linalg.svd(Xc.float(), full_matrices=False)
    return U[:, 0] * S[0]


def upsample_map_2d(
    fmap_hw: torch.Tensor,
    target_side: int,
    mode: str = "nearest",
    two_step: bool = False,
) -> torch.Tensor:
    """fmap_hw: [H, W] -> [target_side, target_side]."""
    if target_side <= 0:
        return fmap_hw
    h, w = fmap_hw.shape[-2], fmap_hw.shape[-1]
    if h == target_side and w == target_side:
        return fmap_hw
    x = fmap_hw.unsqueeze(0).unsqueeze(0).float()  # [1,1,H,W]
    interp_kwargs = {}
    if mode in {"bicubic", "bilinear"}:
        interp_kwargs["align_corners"] = False

    if two_step and target_side > h * 8:
        mid = min(max(h * 14, h * 8), target_side)
        if mid < target_side:
            x = F.interpolate(x, size=(mid, mid), mode=mode, **interp_kwargs)
    y = F.interpolate(x, size=(target_side, target_side), mode=mode, **interp_kwargs)
    return y.squeeze(0).squeeze(0)


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CLIPVisionModel.from_pretrained(args.model_path).to(device)
    processor = CLIPImageProcessor.from_pretrained(args.model_path)
    model.eval()

    input_path = args.input_path
    if os.path.isdir(input_path):
        candidates = [
            os.path.join(input_path, name)
            for name in sorted(os.listdir(input_path))
            if name.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))
        ]
    else:
        candidates = [input_path]

    if not candidates:
        raise FileNotFoundError(f"No images found in {input_path}")

    os.makedirs(args.output_dir, exist_ok=True)

    shared_vmin = None
    shared_vmax = None
    if args.shared_norm_from:
        shared_vmin, shared_vmax = load_shared_norm(args.shared_norm_from)
        print(
            f"[shared_norm] loaded vmin={shared_vmin:.6f}, vmax={shared_vmax:.6f} "
            f"from {args.shared_norm_from}"
        )

    for image_path in candidates:
        image = Image.open(image_path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device)

        with torch.no_grad():
            outputs = model(pixel_values, output_hidden_states=True)

        hidden_states = outputs.hidden_states
        if hidden_states is None or len(hidden_states) < 3:
            raise RuntimeError("CLIP hidden_states not found or too short.")

        # hidden_states[0] = embedding; hidden_states[1:25] = ViT layers 1–24
        vit_layers = hidden_states[1:]
        if len(vit_layers) < 24:
            raise RuntimeError(f"Expected at least 24 ViT layers, got {len(vit_layers)}")
        vit_layers = vit_layers[:24]
        layer_patch_tokens = [feat[:, 1:, :] for feat in vit_layers]  # each: [B, N, D]
        patch_tokens = torch.stack(layer_patch_tokens, dim=1).mean(dim=1)  # [B, N, D]

        feature_map = aggregate_patch_map(patch_tokens, args.agg, args.channel)

        num_patches = feature_map.numel()
        grid_size = int(math.sqrt(num_patches))
        if grid_size * grid_size != num_patches:
            raise ValueError(f"Patch count {num_patches} is not a square number.")

        feature_map = feature_map.view(grid_size, grid_size).float()
        if shared_vmin is not None and shared_vmax is not None:
            feature_map = apply_shared_norm(feature_map, shared_vmin, shared_vmax)
        else:
            if args.norm_mode == "percentile":
                feature_map = normalize_percentile(
                    feature_map,
                    low=args.percentile_low,
                    high=args.percentile_high,
                )
            elif args.norm_mode == "minmax":
                feature_map = normalize_minmax(feature_map)
            elif args.norm_mode == "none":
                pass
        feature_map = upsample_map_2d(
            feature_map,
            args.upsample_size,
            mode=args.upsample_interp,
            two_step=args.two_step,
        )
        arr = feature_map.cpu().numpy()

        base = os.path.splitext(os.path.basename(image_path))[0]
        out_suffix = f"{args.suffix}_{args.agg}"
        if args.agg == "channel":
            out_suffix += f"_c{args.channel}"
        output_path = os.path.join(args.output_dir, f"{base}_{out_suffix}.png")

        fig_inches = max(4.0, args.upsample_size / 150.0)
        plt.figure(figsize=(fig_inches, fig_inches))
        plt.imshow(arr, cmap=args.cmap, interpolation=args.imshow_interp)
        plt.axis("off")
        plt.tight_layout(pad=0)
        plt.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0)
        plt.close()
        print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
