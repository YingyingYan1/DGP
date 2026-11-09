#!/usr/bin/env python3
"""
Grouped bar chart across benchmarks for a fixed LLM layer:
- Pick representative ViT layers per benchmark (default: auto top-K by cross-bench variance)
- Y-axis: mean router weight at (LLM layer, ViT layer)

Default inputs: four router-weights.json files (SEED / LLaVA-W / VizWiz / GQA) under playground/data/eval/...
"""

import argparse
import json
import os
from statistics import pvariance

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = ["Trebuchet MS", "DejaVu Sans", "Arial", "sans-serif"]


DEFAULT_DATASETS = {
    "SEED": "./playground/data/eval/seed_bench/router_stats/llava-v1.5-7b-sft-glimpse-linear-cancha-loss_20260426_174158_router-weights.json",
    "LlavaWild": "./playground/data/eval/llava-bench-in-the-wild/router_stats/glimpse-linear-3300_20260426_173333_router-weights.json",
    "VizWiz": "./playground/data/eval/vizwiz/router_stats/llava-v1.5-7b-sft-glimpse-linear-cancha-loss-val-router-weights.json",
    "GQA": "./playground/data/eval/gqa/router_stats/llava-v1.5-7b-sft-glpse-linear-canchaloss-5197-llava_gqa_testdev_balanced-router-weights.json",
}


def load_llm_row(json_path: str, llm_layer: int):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    mean_weights = data["llm_router"]["mean_weights"]
    if llm_layer < 0 or llm_layer >= len(mean_weights):
        raise ValueError(f"llm_layer={llm_layer} out of range [0, {len(mean_weights) - 1}]")

    return mean_weights[llm_layer]


def pick_discriminative_vit_layers(dataset_rows, topk=4):
    """
    Choose ViT layers with highest cross-bench variance (most discriminative).

    dataset_rows: dict[str, list[float]] mapping benchmark name -> 24 weights
    """
    vit_count = len(next(iter(dataset_rows.values())))
    scored = []
    for vit_layer in range(vit_count):
        values = [dataset_rows[name][vit_layer] for name in dataset_rows]
        score = pvariance(values)
        val_range = max(values) - min(values)
        scored.append((vit_layer, score, val_range))

    scored.sort(key=lambda x: (x[1], x[2]), reverse=True)
    return [x[0] for x in scored[:topk]], scored


def parse_vit_layers(vit_layers_text):
    parts = [p.strip() for p in vit_layers_text.split(",") if p.strip()]
    return [int(p) for p in parts]


def plot_grouped_bars(dataset_rows, vit_layers, llm_layer, output_path, title=None):
    dataset_names = list(dataset_rows.keys())
    font_scale = 1.8
    display_name_map = {
        "SQA": "SQA-I",
        "LlavaWild": r"LLaVA$^{W}$",
    }
    display_names = [display_name_map.get(name, name) for name in dataset_names]

    matrix = np.array([[dataset_rows[d][v] for v in vit_layers] for d in dataset_names])

    n_groups = len(dataset_names)
    n_bars = len(vit_layers)

    x = np.arange(n_groups)
    total_width = 0.89
    bar_width = total_width / n_bars
    offsets = (np.arange(n_bars) - (n_bars - 1) / 2.0) * bar_width

    plt.figure(figsize=(14, 6), dpi=180)
    custom_colors = ["#eaf3e2", "#b4deb6", "#7bc6be", "#439cc4"]

    for j, vit_layer in enumerate(vit_layers):
        plt.bar(
            x + offsets[j],
            matrix[:, j],
            width=bar_width * 0.92,
            color=custom_colors[j % len(custom_colors)],
            edgecolor="black",
            linewidth=0.6,
            label=f"ViT {vit_layer}",
        )

    plt.xticks(x, display_names, fontsize=11 * font_scale)
    plt.yticks(fontsize=11 * font_scale)
    plt.tick_params(axis="both", which="major", labelsize=11 * font_scale)
    plt.ylabel("Normalized Weights", fontsize=12 * font_scale, fontweight="normal")
    plt.margins(x=0.02)

    if title:
        plt.title(title, fontsize=13 * font_scale, fontweight="bold")

    y_max = float(np.max(matrix))
    y_top = y_max * 1.01
    plt.ylim(0.0, y_top)
    plt.legend(
        ncol=1,
        frameon=True,
        fontsize=10 * font_scale,
        loc="upper right",
        bbox_to_anchor=(1.007, 1.02),
        handlelength=1.2,
        handletextpad=0.6,
        borderpad=0.35,
        labelspacing=0.35,
    )
    plt.tight_layout()

    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def build_dataset_rows_for_layer(llm_layer):
    dataset_rows = {}
    for name, path in DEFAULT_DATASETS.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"Dataset {name}: file not found: {path}")
        dataset_rows[name] = load_llm_row(path, llm_layer)
    return dataset_rows


def main():
    parser = argparse.ArgumentParser(description="Grouped bars of router weights across 4 benchmarks")
    parser.add_argument("--llm-layer", type=int, default=31, help="LLM layer index (default 31)")
    parser.add_argument(
        "--topk",
        type=int,
        default=4,
        help="Auto-pick this many ViT layers by cross-bench variance (when not using explicit --vit-layers)",
    )
    parser.add_argument(
        "--vit-layers",
        type=str,
        default="13,14,1,10",
        help="Comma-separated ViT layers, e.g. '13,14,1,10' (overrides --topk)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./playground/data/eval/llm31_top4_grouped_bar_4bench_seed_wild_vizwiz_gqa.png",
        help="Output image path",
    )
    parser.add_argument("--title", type=str, default="", help="Custom figure title")
    parser.add_argument(
        "--batch-all-layers",
        action="store_true",
        help="Render one figure per LLM layer 0..31 (32 files)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="",
        help="Batch mode output directory (defaults to dirname of --output)",
    )

    args = parser.parse_args()

    if args.batch_all_layers:
        out_dir = args.output_dir or os.path.dirname(args.output)
        if not out_dir:
            out_dir = "."
        os.makedirs(out_dir, exist_ok=True)

        print(f"[INFO] Batch output directory: {out_dir}")
        for llm_layer in range(32):
            dataset_rows = build_dataset_rows_for_layer(llm_layer)
            if args.vit_layers:
                vit_layers = parse_vit_layers(args.vit_layers)
            else:
                vit_layers, _ = pick_discriminative_vit_layers(dataset_rows, topk=args.topk)

            out_path = os.path.join(
                out_dir,
                f"llm_layer_{llm_layer:02d}_top{len(vit_layers)}_grouped_bar.png",
            )
            title = args.title if args.title else None
            plot_grouped_bars(
                dataset_rows=dataset_rows,
                vit_layers=vit_layers,
                llm_layer=llm_layer,
                output_path=out_path,
                title=title,
            )
            print(f"[DONE] LLM {llm_layer:02d} | ViT {vit_layers} | {out_path}")
        return

    dataset_rows = build_dataset_rows_for_layer(args.llm_layer)
    if args.vit_layers:
        vit_layers = parse_vit_layers(args.vit_layers)
        scored = None
    else:
        vit_layers, scored = pick_discriminative_vit_layers(dataset_rows, topk=args.topk)

    print(f"[INFO] LLM layer = {args.llm_layer}")
    print(f"[INFO] ViT layers = {vit_layers}")

    if scored is not None:
        print("[INFO] Top-10 ViT layers by cross-bench variance:")
        for vit_layer, var, rg in scored[:10]:
            print(f"  ViT {vit_layer:2d} | var={var:.12f} | range={rg:.12f}")

    for ds_name in dataset_rows:
        vals = [dataset_rows[ds_name][v] for v in vit_layers]
        vals_text = ", ".join([f"ViT{v}={w:.6f}" for v, w in zip(vit_layers, vals)])
        print(f"[DATA] {ds_name}: {vals_text}")

    plot_grouped_bars(
        dataset_rows=dataset_rows,
        vit_layers=vit_layers,
        llm_layer=args.llm_layer,
        output_path=args.output,
        title=args.title if args.title else None,
    )
    print(f"[DONE] Saved: {args.output}")


if __name__ == "__main__":
    main()
