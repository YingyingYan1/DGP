import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPImageProcessor, CLIPVisionModel


def build_parser():
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Visualize layer-wise CLS-to-patch attention heatmaps for CLIP ViT."
    )
    parser.add_argument(
        "--vision-tower",
        type=str,
        default=str(repo_root / "clip-vit-large-patch14-336"),
        help="Path to CLIP vision tower checkpoint.",
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default=str(repo_root / "data"),
        help="Directory containing input images.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(repo_root / "visualization_outputs" / "cls_attention"),
        help="Directory to save per-layer heatmap images.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.45,
        help="Overlay alpha for heatmap on image.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run inference on.",
    )
    return parser


def normalize_map(attn_map: np.ndarray) -> np.ndarray:
    min_v = float(attn_map.min())
    max_v = float(attn_map.max())
    if max_v - min_v < 1e-12:
        return np.zeros_like(attn_map, dtype=np.float32)
    return ((attn_map - min_v) / (max_v - min_v)).astype(np.float32)


def gather_images(input_dir: str):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    paths = []
    for name in sorted(os.listdir(input_dir)):
        p = Path(input_dir) / name
        if p.is_file() and p.suffix.lower() in exts:
            paths.append(p)
    return paths


@torch.no_grad()
def visualize_one_image(
    image_path: Path,
    model: CLIPVisionModel,
    processor: CLIPImageProcessor,
    output_dir: Path,
    alpha: float = 0.45,
    device: str = "cuda",
):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)

    outputs = model(pixel_values=pixel_values, output_attentions=True)
    attentions = outputs.attentions  # tuple(num_layers), each [B, heads, seq, seq]

    image_size = model.config.image_size
    patch_size = model.config.patch_size
    grid_size = image_size // patch_size
    num_patches = grid_size * grid_size

    base_vis = image.resize((image_size, image_size), Image.BICUBIC)
    base_vis = np.asarray(base_vis, dtype=np.float32) / 255.0

    num_layers = len(attentions)
    image_output_dir = output_dir / image_path.stem
    image_output_dir.mkdir(parents=True, exist_ok=True)

    for i, attn in enumerate(attentions):
        # CLS token is index 0; keep only patch tokens [1:].
        cls_to_patch = attn.mean(dim=1)[0, 0, 1 : 1 + num_patches]  # [576]
        attn_map = cls_to_patch.reshape(1, 1, grid_size, grid_size)
        attn_map = F.interpolate(
            attn_map.to(device),
            size=(image_size, image_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0).squeeze(0)
        attn_map = normalize_map(attn_map.detach().cpu().numpy())

        fig, ax = plt.subplots(1, 1, figsize=(4, 4), dpi=180)
        ax.imshow(base_vis)
        ax.imshow(attn_map, cmap="jet", alpha=alpha)
        ax.set_title(f"Layer {i + 1}", fontsize=11)
        ax.axis("off")
        plt.tight_layout()

        out_path = image_output_dir / f"{image_path.stem}_layer_{i + 1:02d}.png"
        fig.savefig(out_path, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)
        print(f"[Saved] {out_path}")


def main():
    args = build_parser().parse_args()
    device = torch.device(args.device)

    model = CLIPVisionModel.from_pretrained(args.vision_tower).to(device)
    model.eval()
    processor = CLIPImageProcessor.from_pretrained(args.vision_tower)

    image_paths = gather_images(args.input_dir)
    if not image_paths:
        raise FileNotFoundError(f"No images found in: {args.input_dir}")

    output_dir = Path(args.output_dir)
    for image_path in image_paths:
        visualize_one_image(
            image_path=image_path,
            model=model,
            processor=processor,
            output_dir=output_dir,
            alpha=args.alpha,
            device=str(device),
        )

    print(f"Done. Total images processed: {len(image_paths)}")


if __name__ == "__main__":
    main()
