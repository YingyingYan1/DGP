#!/usr/bin/env python3
"""
Router Weights Curve Visualization

For each LLM layer, plot the layer-wise router fusion weights over ViT layers:
- X: ViT layer (0-23)
- Y: router fusion weight (attention to that ViT layer)
- One subplot per LLM layer (32 total by default)
- Shows both raw (dashed) and Gaussian-smoothed (solid) curves
"""

import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from pathlib import Path

# ============ Default paths ============
DEFAULT_JSON_PATH = "./playground/data/eval/textvqa/router_stats/llava-v1.5-7b-sft-glimpse-linear-router-weights.json"
DEFAULT_OUTPUT_DIR = "./playground/data/eval/textvqa/router_stats/"


def load_router_weights(json_path: str) -> dict:
    """Load router-stats JSON."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data


def smooth_curve(y: np.ndarray, sigma: float = 1.5) -> np.ndarray:
    """1D Gaussian smoothing."""
    return gaussian_filter1d(y, sigma=sigma)


def plot_single_layer(
    ax,
    vit_layers: np.ndarray,
    weights: np.ndarray,
    llm_layer: int,
    show_smoothed: bool = True,
    sigma: float = 1.5,
    color: str = 'red'
):
    """Plot router fusion weights for one LLM layer."""
    ax.plot(
        vit_layers,
        weights,
        linestyle='--',
        color=color,
        alpha=0.7,
        linewidth=1.2,
        label='Raw Attention',
    )

    if show_smoothed:
        smoothed = smooth_curve(weights, sigma=sigma)
        ax.plot(
            vit_layers,
            smoothed,
            linestyle='-',
            color=color,
            linewidth=2,
            label='Smoothed Attention',
        )

    max_idx = np.argmax(weights)
    ax.scatter(
        [max_idx],
        [weights[max_idx]],
        color=color,
        s=50,
        zorder=5,
        edgecolor='white',
        linewidth=1,
    )

    ax.set_title(
        f'Decoder Layer {llm_layer:02d} - Router Fusion Weights',
        fontsize=10,
        fontweight='bold',
    )
    ax.set_xlabel('CLIP Visual Layer Index', fontsize=9)
    ax.set_ylabel('Layerwise Attention Score', fontsize=9)
    ax.set_xlim(-0.5, 23.5)
    ax.set_ylim(0, max(weights.max() * 1.1, 0.05))
    ax.set_xticks(range(0, 24, 5))
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.legend(loc='upper right', fontsize=7, framealpha=0.9)


def plot_all_layers_grid(
    weights: np.ndarray,
    output_path: str,
    title: str = "Router Weights per LLM Layer",
    color: str = 'red',
    sigma: float = 1.5,
    ncols: int = 4
):
    """
    Grid of subplots for all LLM layers.

    Args:
        weights: [32, 24] with rows = LLM layers, cols = ViT layers
        output_path: output image path
        title: figure suptitle
        color: line color
        sigma: Gaussian smoothing width
        ncols: subplots per row
    """
    n_llm_layers = weights.shape[0]
    nrows = (n_llm_layers + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))
    fig.suptitle(title, fontsize=16, fontweight='bold', y=1.02)

    vit_layers = np.arange(24)

    for i in range(n_llm_layers):
        row, col = i // ncols, i % ncols
        ax = axes[row, col] if nrows > 1 else axes[col]

        plot_single_layer(
            ax=ax,
            vit_layers=vit_layers,
            weights=weights[i],
            llm_layer=i,
            show_smoothed=True,
            sigma=sigma,
            color=color,
        )

    for i in range(n_llm_layers, nrows * ncols):
        row, col = i // ncols, i % ncols
        ax = axes[row, col] if nrows > 1 else axes[col]
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"[Curves] Saved: {output_path}")


def plot_selected_layers(
    weights: np.ndarray,
    output_path: str,
    selected_layers: list = None,
    title: str = "Selected Decoder Layers - Router Fusion Weights",
    color: str = 'red',
    sigma: float = 1.5
):
    """
    Subplots for a small list of LLM layers (2 columns).

    Args:
        weights: [32, 24]
        output_path: output image path
        selected_layers: LLM layer indices to plot
        title: figure suptitle
        color: line color
        sigma: Gaussian smoothing width
    """
    if selected_layers is None:
        selected_layers = [0, 1, 10, 31]
    n = len(selected_layers)
    ncols = 2
    nrows = (n + 1) // 2

    fig, axes = plt.subplots(nrows, ncols, figsize=(10, 4 * nrows))
    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)

    vit_layers = np.arange(24)

    for idx, llm_layer in enumerate(selected_layers):
        row, col = idx // ncols, idx % ncols
        if nrows > 1:
            ax = axes[row, col]
        else:
            ax = axes[col] if n > 1 else axes

        plot_single_layer(
            ax=ax,
            vit_layers=vit_layers,
            weights=weights[llm_layer],
            llm_layer=llm_layer,
            show_smoothed=True,
            sigma=sigma,
            color=color,
        )

    if n % 2 == 1 and nrows > 1:
        axes[nrows - 1, 1].axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"[Selected] Saved: {output_path}")


def plot_comparison_curves(
    weights_list: list,
    labels: list,
    output_path: str,
    selected_layers: list = None,
    colors: list = None
):
    """
    Overlay multiple checkpoints at the same LLM layers.

    Args:
        weights_list: list of [32, 24] arrays
        labels: legend label per checkpoint
        output_path: output image path
        selected_layers: which LLM layers to plot
        colors: one color per checkpoint (optional)
    """
    if selected_layers is None:
        selected_layers = [0, 15, 31]
    if colors is None:
        colors = plt.cm.tab10(np.linspace(0, 1, len(weights_list)))

    n_layers = len(selected_layers)
    fig, axes = plt.subplots(1, n_layers, figsize=(5 * n_layers, 4))

    if n_layers == 1:
        axes = [axes]

    vit_layers = np.arange(24)

    for ax, llm_layer in zip(axes, selected_layers):
        for weights, label, color in zip(weights_list, labels, colors):
            w = weights[llm_layer]
            ax.plot(vit_layers, w, linewidth=2, label=label, color=color)

        ax.set_title(f'LLM Layer {llm_layer}', fontsize=12, fontweight='bold')
        ax.set_xlabel('ViT Layer Index', fontsize=10)
        ax.set_ylabel('Attention Weight', fontsize=10)
        ax.set_xlim(-0.5, 23.5)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"[Comparison Curves] Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Plot layer-wise router fusion weight curves')
    parser.add_argument(
        '--json-path', '-j',
        type=str,
        default=DEFAULT_JSON_PATH,
        help=f'Path to router-stats JSON (default: {DEFAULT_JSON_PATH})'
    )
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help=f'Output directory (default: {DEFAULT_OUTPUT_DIR})'
    )
    parser.add_argument(
        '--color', '-c',
        type=str,
        default='red',
        help='Line color (e.g. red, blue, green, orange, purple)'
    )
    parser.add_argument(
        '--sigma', '-s',
        type=float,
        default=1.5,
        help='Gaussian smoothing sigma'
    )
    parser.add_argument(
        '--ncols',
        type=int,
        default=4,
        help='Subplots per row in the full-grid mode'
    )
    parser.add_argument(
        '--selected',
        type=int,
        nargs='+',
        default=None,
        help='Plot only these LLM layer indices, e.g. --selected 0 1 10 31'
    )
    parser.add_argument(
        '--compare',
        type=str,
        nargs='+',
        default=None,
        help='Additional JSON files to compare against --json-path'
    )
    parser.add_argument(
        '--compare-layers',
        type=int,
        nargs='+',
        default=[0, 15, 31],
        help='LLM layers to show in --compare mode (default: 0 15 31)'
    )

    args = parser.parse_args()

    print("=" * 80)
    print("Router Weights Curve Visualization")
    print("=" * 80)

    print(f"\n[Load] {args.json_path}")
    data = load_router_weights(args.json_path)

    if 'llm_router' in data:
        mean_weights = np.array(data['llm_router']['mean_weights'])
    else:
        mean_weights = np.array(data['mean_weights'])

    print(f"[Info] LLM router weight shape {mean_weights.shape} (LLM layers x ViT layers)")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_name = Path(args.json_path).stem.replace('-router-weights', '')

    if args.compare:
        all_paths = [args.json_path] + args.compare
        weights_list = []
        labels = []
        for p in all_paths:
            d = load_router_weights(p)
            if 'llm_router' in d:
                w = np.array(d['llm_router']['mean_weights'])
            else:
                w = np.array(d['mean_weights'])
            weights_list.append(w)
            labels.append(Path(p).stem.replace('-router-weights', '').split('-')[-1])

        output_path = str(output_dir / 'comparison_curves.png')
        plot_comparison_curves(
            weights_list,
            labels,
            output_path,
            selected_layers=args.compare_layers,
        )
        return

    if args.selected:
        output_path = str(output_dir / f'{model_name}-selected-curves.png')
        title = f"Selected Decoder Layers - {model_name}"

        plot_selected_layers(
            weights=mean_weights,
            output_path=output_path,
            selected_layers=args.selected,
            title=title,
            color=args.color,
            sigma=args.sigma,
        )
        return

    output_path = str(output_dir / f'{model_name}-llm-router-curves.png')
    title = f"LLM Router Weights per Layer\n{model_name}"

    plot_all_layers_grid(
        weights=mean_weights,
        output_path=output_path,
        title=title,
        color=args.color,
        sigma=args.sigma,
        ncols=args.ncols,
    )

    print("\n" + "=" * 80)
    print("Output figure:")
    print("=" * 80)
    print(f"  LLM router curves: {output_path}")
    print("=" * 80)


if __name__ == '__main__':
    main()
