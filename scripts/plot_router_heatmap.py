#!/usr/bin/env python3
"""
Router Weights Heatmap Visualization

Heatmaps for layerwise vision fusion router weights:
1. LLM (layer-wise) router: X = LLM layer (0-31), Y = ViT layer (0-23)
2. Pre-router: single-row heatmap over 24 ViT layers
"""

import json
import argparse
import logging
import os
from typing import Optional
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import cm as mpl_cm
import matplotlib.font_manager as fm
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm, BoundaryNorm, Normalize
from matplotlib.patches import FancyBboxPatch
from matplotlib.ticker import FormatStrFormatter
from pathlib import Path


# Global font / title spacing (tweak as needed)
TITLE_PAD = 8

# Font sizes aligned with bar-plot scripts (1.5x prior default)
TITLE_FONTSIZE = 36
AXIS_LABEL_FONTSIZE = 34
TICK_FONTSIZE = 30
CBAR_LABEL_FONTSIZE = 34
CBAR_TICK_FONTSIZE = 30

_FONT_FALLBACK_NOTICE_PRINTED = False


def _available_font_families() -> set[str]:
    """Return lowercase family names visible to matplotlib's font manager."""
    names = set()
    try:
        for font in fm.fontManager.ttflist:
            if getattr(font, "name", None):
                names.add(str(font.name).lower())
    except Exception:
        # Best-effort only; never block plotting
        pass
    return names


def _pick_first_available_font(candidates: list[str], available: set[str]) -> Optional[str]:
    for c in candidates:
        if not c:
            continue
        if c.lower() in available:
            return c
    return None


def _resolve_plot_font_family() -> str:
    """
    Resolve the plot font family:
    - Prefer Trebuchet MS by default (override with ROUTER_PLOT_FONT_FAMILY)
    - On Linux it is often missing: fall back to an installed sans font to avoid findfont spam
    """
    global _FONT_FALLBACK_NOTICE_PRINTED

    requested = os.environ.get("ROUTER_PLOT_FONT_FAMILY", "Trebuchet MS").strip() or "Trebuchet MS"
    available = _available_font_families()

    # Some systems register Trebuchet under alternate family names
    preferred_order = [
        requested,
        "Trebuchet MS",
        "Trebuchet",
        "Liberation Sans",
        "Arial",
        "Helvetica",
        "DejaVu Sans",
    ]

    chosen = _pick_first_available_font(preferred_order, available) or "DejaVu Sans"

    if chosen.lower() != requested.lower() and not _FONT_FALLBACK_NOTICE_PRINTED:
        print(
            f"[Font] Requested {requested!r} not found; using {chosen!r} instead. "
            f"Install a font package or set ROUTER_PLOT_FONT_FAMILY."
        )
        _FONT_FALLBACK_NOTICE_PRINTED = True

    return chosen


# Effective plot font (auto-fallback when Trebuchet is unavailable)
FONT_FAMILY = _resolve_plot_font_family()

# Quieter matplotlib font-manager logs (plots still use FONT_FAMILY above)
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)


def _apply_plot_style():
    """Apply matplotlib sans-serif defaults (Trebuchet preferred; fallback if missing)."""
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = [
        FONT_FAMILY,
        "DejaVu Sans",
        "Arial",
        "Liberation Sans",
        "sans-serif",
    ]


def _colorbar_tick_values(vmin: float, vmax: float, max_ticks: int = 6) -> np.ndarray:
    """Build sparser colorbar ticks so high-end values (e.g. 0.8/0.9) do not overlap."""
    span = vmax - vmin
    if span <= 0 or not np.isfinite(span):
        return np.array([vmin, vmax], dtype=float)

    # Target step size on a clean order-of-magnitude grid
    raw = span / max(1, max_ticks - 1)
    exp = np.floor(np.log10(raw))
    frac = raw / (10 ** exp)
    nice_frac = 1.0
    for cand in (1.0, 2.0, 5.0, 10.0):
        if frac <= cand:
            nice_frac = cand
            break
    step = nice_frac * (10 ** exp)

    start = np.ceil(vmin / step) * step
    ticks = np.arange(start, vmax + 1e-12 * span, step, dtype=float)

    # Include endpoints
    ticks = np.unique(np.concatenate([[vmin], ticks, [vmax]]))

    # Drop ticks that are too close together
    merged = [ticks[0]]
    min_gap = 0.08 * span if span > 0 else 1e-6
    for t in ticks[1:]:
        if t - merged[-1] >= min_gap or t == vmax:
            merged.append(t)
    out = np.array(merged, dtype=float)

    # Cap tick count
    if len(out) > max_ticks:
        idx = np.linspace(0, len(out) - 1, max_ticks).round().astype(int)
        out = np.unique(out[idx])

    return out


def _style_llm_vertical_colorbar(colorbar, boundaries, vmin: float, vmax: float) -> None:
    """Style the right-hand vertical colorbar for the LLM router heatmap."""
    if colorbar is None:
        return

    if boundaries is not None:
        max_ticks = min(8, len(boundaries))
        idx = np.linspace(0, len(boundaries) - 1, max_ticks).round().astype(int)
        tick_vals = np.unique(boundaries[idx])
    else:
        tick_vals = _colorbar_tick_values(vmin, vmax, max_ticks=6)

    # Hide 0.20 and 0.60 tick marks (plot-specific)
    tick_vals = np.array(
        [
            t
            for t in np.asarray(tick_vals, dtype=float)
            if (not np.isclose(t, 0.20, atol=5e-3)) and (not np.isclose(t, 0.60, atol=5e-3))
        ]
    )
    colorbar.set_ticks(tick_vals)
    colorbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    colorbar.ax.tick_params(labelsize=CBAR_TICK_FONTSIZE)
    for t in colorbar.ax.get_yticklabels():
        t.set_fontfamily(FONT_FAMILY)

    ylab = colorbar.ax.yaxis.get_label()
    label_text = ylab.get_text() or 'Normalized Values'
    colorbar.set_label(
        label_text,
        fontsize=CBAR_LABEL_FONTSIZE,
        fontweight='normal',
        fontfamily=FONT_FAMILY,
        labelpad=-40,
        rotation=90,
    )

    for spine in colorbar.ax.spines.values():
        spine.set_visible(True)
        spine.set_color('black')
        spine.set_linewidth(1.5)


def _draw_rounded_cell_matrix(
    ax,
    weights: np.ndarray,
    norm: Normalize,
    cmap,
    *,
    cell_gap: float = 0.22,
    rounding_size: Optional[float] = None,
    show_values: bool = False,
    value_fmt: str = '.3f',
    annot_fontsize: int = 8,
):
    """
    Draw the heatmap with rounded rectangles (tile / "Q-block" style).
    weights: [nrows, ncols]; origin matches sns.heatmap(origin='upper'): row 0 at the top.
    """
    weights = np.asarray(weights, dtype=float)
    nrows, ncols = weights.shape
    if nrows <= 0 or ncols <= 0:
        return

    gap = float(np.clip(cell_gap, 0.0, 0.45))
    w = 1.0 * (1.0 - gap)
    h = 1.0 * (1.0 - gap)
    if w <= 0 or h <= 0:
        raise ValueError(f"cell_gap={cell_gap} is too large: cell width/height must be positive")

    if rounding_size is None:
        rs = 0.35 * min(w, h)
    else:
        rs = float(rounding_size)
    rs_max = 0.5 * min(w, h) - 1e-6
    rs = float(np.clip(rs, 0.0, rs_max))

    sm = mpl_cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])

    for i in range(nrows):
        for j in range(ncols):
            v = weights[i, j]
            face = sm.to_rgba(v, bytes=False)
            x0 = j - w / 2.0
            y0 = i - h / 2.0
            patch = FancyBboxPatch(
                (x0, y0),
                w,
                h,
                boxstyle=f"round,pad=0,rounding_size={rs}",
                linewidth=0.0,
                edgecolor="none",
                facecolor=face,
                clip_on=True,
                zorder=1,
            )
            ax.add_patch(patch)

            if show_values:
                ax.text(
                    j,
                    i,
                    format(v, value_fmt),
                    ha='center',
                    va='center',
                    fontsize=annot_fontsize,
                    fontfamily=FONT_FAMILY,
                    color='black',
                    zorder=2,
                )

    ax.set_xlim(-0.5, ncols - 0.5)
    ax.set_ylim(nrows - 0.5, -0.5)  # Row 0 at top (same as seaborn heatmap)
    ax.set_xticks(np.arange(ncols))
    ax.set_yticks(np.arange(nrows))


# ============ Default paths ============
DEFAULT_JSON_PATH ="./playground/data/eval/pope/router_stats/llava-v1.5-7b-glimpse-linear-cancahloss-5197-router-weights.json"
DEFAULT_OUTPUT_DIR = "./playground/data/eval/pope/router_stats/"

# Diverging colormap: deep blue - white - deep red (similar feel to RdBu_r)
PEARSON_BLUE_RED_CMAP = LinearSegmentedColormap.from_list(
    "pearson_blue_red",
    [
        "#053061",
        "#2166AC",
        "#4393C3",
        "#92C5DE",
        "#D1E5F0",
        "#F7F7F7",
        "#FDDBC7",
        "#F4A582",
        "#D6604D",
        "#B2182B",
        "#67001F",
    ],
    N=256,
)


def load_router_weights(json_path: str) -> dict:
    """Load router weights from JSON."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data


def build_weights_from_per_sample(
    data: dict,
    sample_idx: int = None,
    question_id: str = None
) -> np.ndarray:
    """
    Build a 32x24 weight matrix from per_sample_weights.
    - If question_id is set: select that sample and plot its 32x24
    - Elif sample_idx is set: take the sample_idx-th entry
    - Else: average across all samples
    Returns shape [24, 32] for plot_llm_router_heatmap (rows = ViT, cols = LLM).
    """
    if 'per_sample_weights' not in data or not data['per_sample_weights']:
        raise ValueError("JSON has no per_sample_weights or it is empty")
    samples = data['per_sample_weights']
    if question_id is not None:
        idx = None
        for i, s in enumerate(samples):
            if str(s.get('question_id', '')) == str(question_id):
                idx = i
                break
        if idx is None:
            some_ids = [s.get('question_id', '?') for s in samples[:5]]
            raise ValueError(
                f"question_id={question_id!r} not found. "
                f"First few ids: {some_ids}"
            )
        samples = [samples[idx]]
    elif sample_idx is not None:
        if sample_idx < 0 or sample_idx >= len(samples):
            raise ValueError(f"sample_idx={sample_idx} out of range [0, {len(samples)-1}]")
        samples = [samples[sample_idx]]
    # Per sample: llm_router_weights is {"0": [24 floats], ... "31": [24 floats]}
    matrices = []
    for s in samples:
        rw = s.get('llm_router_weights', {})
        rows = [rw[str(llm)] for llm in range(32) if str(llm) in rw]
        if len(rows) != 32:
            raise ValueError(f"Sample llm_router_weights has {len(rows)} LLM layers, expected 32")
        mat = np.array(rows, dtype=np.float64)  # [32, 24]
        matrices.append(mat)
    out = np.stack(matrices, axis=0).mean(axis=0)  # [32, 24]
    return out.T  # [24, 32] for heatmap


def plot_llm_router_heatmap(
    weights: np.ndarray,
    output_path: str,
    title: str = "Gated Connection Weights on POPE",
    cmap="pearson_blue_red",
    figsize: tuple = (16, 10),
    show_values: bool = False,
    levels: int = 0,
    vmin: float = None,
    vmax: float = None,
    rounded_cells: bool = False,
    cell_gap: float = 0.22,
    rounding_size: Optional[float] = None,
):
    """
    Plot the LLM (layer-wise) router heatmap.

    Args:
        weights: [24, 32] with rows = ViT layers, cols = LLM layers
        output_path: output image path
        title: figure title
        cmap: colormap; try 'RdBu_r', 'coolwarm', 'viridis', 'YlOrRd'
        figsize: figure size
        show_values: annotate cells with numeric values
        levels: number of discrete color bins (0 = continuous)
        vmin, vmax: color scale limits
        rounded_cells: draw rounded rectangle cells (tile style)
        cell_gap: gap ratio in rounded mode (larger = smaller tiles; 0~0.45 typical)
        rounding_size: corner radius (None = auto from cell size)
    """
    fig, ax = plt.subplots(figsize=figsize)

    _apply_plot_style()

    plt.rcParams['font.size'] = TICK_FONTSIZE
    plt.rcParams['axes.titlesize'] = TITLE_FONTSIZE
    plt.rcParams['axes.labelsize'] = AXIS_LABEL_FONTSIZE
    
    # Color limits: use data min/max by default.
    # If all values share sign, center the colormap at the midpoint so both ends are visible.
    if vmin is None:
        vmin = float(np.min(weights))
    if vmax is None:
        vmax = float(np.max(weights))
    if np.isclose(vmin, vmax):
        vmax = vmin + 1e-8

    boundaries = None  # when levels>1: bin edges for colorbar ticks

    # Optional discretization for clearer band separation
    if levels is not None and levels > 1:
        boundaries = np.linspace(vmin, vmax, levels + 1)
        discrete_cmap = cmap.resampled(levels) if hasattr(cmap, "resampled") else plt.get_cmap(cmap, levels)
        norm = BoundaryNorm(boundaries, ncolors=discrete_cmap.N, clip=True)
        heatmap_cmap = discrete_cmap
    else:
        # Continuous mode: emphasize [0, 0.05] with extra colormap span; >0.05 ramps to deep red
        pivot = 0.05
        low_portion = 0.75  # fraction of cmap for [vmin, pivot] to highlight small weights

        # [vmin, pivot]: deep blue -> light blue -> light pink (ends at #FFBBC2)
        low_colors = [
    
            "#439cc4",
            "#7bc6be",
            "#b4deb6",
            "#eaf3e2",
            "#E6F7FF",
            "#FFE2E5",
            "#FFCCD1",
            "#FFBBC2",
        ]
        # >pivot: #FFBBC2 -> deep red (low segment already ends at #FFBBC2 at the pivot)
        high_colors = [
            "#FF929D",
            "#F57582",
            "#E65E67",
            "#D9333B",
            "#B8292F",
            "#800D00",
        ]

        if pivot > vmin and pivot < vmax:
            p0, p1, p2 = 0.0, low_portion, 1.0
            anchors = []
            anchors += list(zip(np.linspace(p0, p1, len(low_colors)), low_colors))
            anchors += list(zip(np.linspace(p1, p2, len(high_colors)), high_colors))
            heatmap_cmap = LinearSegmentedColormap.from_list("router_bwr_focus", anchors, N=256)

            class PiecewiseNormalize(Normalize):
                def __init__(self, vmin, vmax, pivot, low_pos, clip=False):
                    super().__init__(vmin=vmin, vmax=vmax, clip=clip)
                    self.pivot = pivot
                    self.low_pos = low_pos

                def __call__(self, value, clip=None):
                    val = np.asarray(value, dtype=float)
                    out = np.interp(val, [self.vmin, self.pivot, self.vmax], [0.0, self.low_pos, 1.0])
                    return np.ma.masked_array(out)

            norm = PiecewiseNormalize(vmin=vmin, vmax=vmax, pivot=pivot, low_pos=low_portion)
        else:
            if vmin < 0 < vmax:
                center_value = 0.0
            else:
                center_value = (vmin + vmax) / 2.0
            norm = TwoSlopeNorm(vmin=vmin, vcenter=center_value, vmax=vmax)
            heatmap_cmap = cmap

    colorbar = None
    if rounded_cells:
        _draw_rounded_cell_matrix(
            ax,
            weights,
            norm,
            heatmap_cmap,
            cell_gap=cell_gap,
            rounding_size=rounding_size,
            show_values=show_values,
            value_fmt='.3f',
            annot_fontsize=8,
        )
        sm = mpl_cm.ScalarMappable(norm=norm, cmap=heatmap_cmap)
        sm.set_array([])
        colorbar = fig.colorbar(
            sm,
            ax=ax,
            shrink=0.82,
            pad=0.02,
        )
    else:
        sns.heatmap(
            weights,
            ax=ax,
            cmap=heatmap_cmap,
            annot=show_values,
            fmt='.3f' if show_values else None,
            annot_kws={'size': 8, 'family': FONT_FAMILY} if show_values else None,
            xticklabels=range(32),
            yticklabels=range(24),
            cbar_kws={
                'label': 'Normalized Values',
                'shrink': 0.82,
                # Slightly larger pad keeps colorbar from hugging the main axes
                'pad': 0.02,
            },
            norm=norm,
            linewidths=0.5,
            linecolor='white'
        )
        colorbar = ax.collections[0].colorbar
    
    ax.set_xlabel('LLM Layer Index', fontsize=AXIS_LABEL_FONTSIZE, fontweight='normal', fontfamily=FONT_FAMILY)
    ax.set_ylabel('ViT Layer Index', fontsize=AXIS_LABEL_FONTSIZE, fontweight='normal', fontfamily=FONT_FAMILY)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('black')
        spine.set_linewidth(1.5)

    _style_llm_vertical_colorbar(colorbar, boundaries, vmin, vmax)

    ax.set_xticklabels(ax.get_xticklabels(), rotation=0, fontsize=TICK_FONTSIZE, fontfamily=FONT_FAMILY)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=TICK_FONTSIZE, fontfamily=FONT_FAMILY)
    
    # Show every 4th tick to reduce clutter
    for i, label in enumerate(ax.get_xticklabels()):
        if i % 4 != 0:
            label.set_visible(False)
        else:
            label.set_fontfamily(FONT_FAMILY)
    for i, label in enumerate(ax.get_yticklabels()):
        if i % 4 != 0:
            label.set_visible(False)
        else:
            label.set_fontfamily(FONT_FAMILY)
    
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"[LLM Router Heatmap] Saved: {output_path}")


def plot_pre_router_heatmap(
    weights: np.ndarray,
    output_path: str,
    title: str = "Pre-Router Weights: Attention to 24 ViT Layers",
    cmap="pearson_blue_red",
    figsize: tuple = (16, 3),
    show_values: bool = True
):
    """
    Plot the pre-router heatmap (single row).

    Args:
        weights: length-24 vector over ViT layers
        output_path: output image path
        title: figure title
        cmap: colormap name or object
        figsize: figure size
        show_values: annotate cells
    """
    plt.figure(figsize=figsize)

    _apply_plot_style()

    plt.rcParams['font.size'] = TICK_FONTSIZE
    plt.rcParams['axes.titlesize'] = TITLE_FONTSIZE
    plt.rcParams['axes.labelsize'] = AXIS_LABEL_FONTSIZE
    
    weights_2d = weights.reshape(1, -1)

    ax = sns.heatmap(
        weights_2d,
        cmap=cmap,
        annot=show_values,
        fmt='.3f' if show_values else None,
        annot_kws={'size': 11, 'fontweight': 'normal', 'family': FONT_FAMILY} if show_values else None,
        xticklabels=range(24),
        yticklabels=['Pre-Router'],
        cbar_kws={'label': 'Weight', 'shrink': 0.5, 'orientation': 'horizontal', 'pad': 0.25},
        linewidths=1,
        linecolor='white'
    )
    
    ax.set_xlabel('ViT Layer (0-23)', fontsize=AXIS_LABEL_FONTSIZE, fontweight='normal', fontfamily=FONT_FAMILY)
    ax.set_title(title, fontsize=TITLE_FONTSIZE, fontweight='normal', pad=TITLE_PAD, fontfamily=FONT_FAMILY)

    ax.set_xticklabels(ax.get_xticklabels(), rotation=0, fontsize=TICK_FONTSIZE, fontfamily=FONT_FAMILY)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=TICK_FONTSIZE, fontweight='normal', fontfamily=FONT_FAMILY)
    ax.tick_params(axis='both', labelsize=TICK_FONTSIZE)
    cb = ax.collections[0].colorbar
    if cb is not None:
        cb.ax.tick_params(labelsize=CBAR_TICK_FONTSIZE)
        cb.set_label('Weight', fontsize=CBAR_LABEL_FONTSIZE, fontweight='normal', fontfamily=FONT_FAMILY)
        for t in cb.ax.get_xticklabels():
            t.set_fontfamily(FONT_FAMILY)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"[Pre-Router Heatmap] Saved: {output_path}")


def plot_pre_router_bar(
    weights: np.ndarray,
    output_path: str,
    title: str = "Pre-Router Weights Distribution",
    figsize: tuple = (14, 5)
):
    """
    Bar chart for pre-router weights.

    Args:
        weights: length-24 vector
        output_path: output image path
        title: figure title
        figsize: figure size
    """
    plt.figure(figsize=figsize)

    _apply_plot_style()

    plt.rcParams['font.size'] = TICK_FONTSIZE
    plt.rcParams['axes.titlesize'] = TITLE_FONTSIZE
    plt.rcParams['axes.labelsize'] = AXIS_LABEL_FONTSIZE
    
    colors = plt.cm.YlOrRd(weights / weights.max())
    
    bars = plt.bar(range(24), weights, color=colors, edgecolor='black', linewidth=0.5)
    
    for i, (bar, w) in enumerate(zip(bars, weights)):
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f'{w:.3f}',
            ha='center',
            va='bottom',
            fontsize=10,
            rotation=45,
            fontfamily=FONT_FAMILY,
        )
    
    plt.xlabel('ViT Layer (0-23)', fontsize=AXIS_LABEL_FONTSIZE, fontweight='normal', fontfamily=FONT_FAMILY)
    plt.ylabel('Weight', fontsize=AXIS_LABEL_FONTSIZE, fontweight='normal', fontfamily=FONT_FAMILY)
    plt.title(title, fontsize=TITLE_FONTSIZE, fontweight='normal', pad=TITLE_PAD, fontfamily=FONT_FAMILY)
    plt.xticks(range(24), fontsize=TICK_FONTSIZE)
    plt.tick_params(axis='y', labelsize=TICK_FONTSIZE)
    ax_bar = plt.gca()
    for t in ax_bar.get_xticklabels():
        t.set_fontfamily(FONT_FAMILY)
    for t in ax_bar.get_yticklabels():
        t.set_fontfamily(FONT_FAMILY)
    plt.grid(axis='y', alpha=0.3)
    
    max_idx = weights.argmax()
    plt.axhline(y=weights[max_idx], color='red', linestyle='--', alpha=0.5, label=f'Max: Layer {max_idx} = {weights[max_idx]:.4f}')
    plt.legend(loc='upper left', prop={'family': FONT_FAMILY, 'size': TICK_FONTSIZE})
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"[Pre-Router Bar Chart] Saved: {output_path}")


def plot_comparison_heatmaps(
    weights_list: list,
    labels: list,
    output_path: str,
    cmap="pearson_blue_red"
):
    """
    Side-by-side comparison heatmaps for multiple checkpoints.
    """
    n = len(weights_list)

    _apply_plot_style()

    plt.rcParams['font.size'] = TICK_FONTSIZE
    plt.rcParams['axes.titlesize'] = TITLE_FONTSIZE
    plt.rcParams['axes.labelsize'] = AXIS_LABEL_FONTSIZE

    fig, axes = plt.subplots(1, n, figsize=(8*n, 10))
    
    if n == 1:
        axes = [axes]
    
    vmin = min(w.min() for w in weights_list)
    vmax = max(w.max() for w in weights_list)
    
    for ax, weights, label in zip(axes, weights_list, labels):
        sns.heatmap(
            weights,
            ax=ax,
            cmap=cmap,
            xticklabels=range(32),
            yticklabels=range(24),
            vmin=vmin,
            vmax=vmax,
            cbar_kws={'shrink': 0.6}
        )
        ax.set_xlabel('LLM Layer', fontsize=AXIS_LABEL_FONTSIZE, fontfamily=FONT_FAMILY)
        ax.set_ylabel('ViT Layer', fontsize=AXIS_LABEL_FONTSIZE, fontfamily=FONT_FAMILY)
        ax.set_title(label, fontsize=TITLE_FONTSIZE, fontweight='normal', pad=TITLE_PAD, fontfamily=FONT_FAMILY)
        
        for i, l in enumerate(ax.get_xticklabels()):
            if i % 8 != 0:
                l.set_visible(False)
            else:
                l.set_fontsize(TICK_FONTSIZE)
                l.set_fontfamily(FONT_FAMILY)
        for i, l in enumerate(ax.get_yticklabels()):
            if i % 6 != 0:
                l.set_visible(False)
            else:
                l.set_fontsize(TICK_FONTSIZE)
                l.set_fontfamily(FONT_FAMILY)

        cb = ax.collections[0].colorbar
        if cb is not None:
            cb.ax.tick_params(labelsize=CBAR_TICK_FONTSIZE)
            for t in cb.ax.get_yticklabels():
                t.set_fontfamily(FONT_FAMILY)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"[Comparison] Saved: {output_path}")


def analyze_weights(weights: np.ndarray, name: str = ""):
    """Print router weight statistics."""
    print(f"\n{'='*60}")
    print(f"[Analysis] {name}")
    print(f"{'='*60}")
    print(f"Shape: {weights.shape}")
    print(f"Min: {weights.min():.6f}")
    print(f"Max: {weights.max():.6f}")
    print(f"Mean: {weights.mean():.6f}")
    print(f"Std: {weights.std():.6f}")

    print(f"\nTop-1 ViT layer per LLM layer (argmax along ViT rows):")
    top_vit = weights.argmax(axis=0)
    for llm_layer in range(0, 32, 4):
        print(f"  LLM Layer {llm_layer:2d}: ViT Layer {top_vit[llm_layer]:2d} (weight={weights[top_vit[llm_layer], llm_layer]:.4f})")
    
    print(f"\nMean attention to each ViT layer (mean over LLM columns):")
    vit_avg = weights.mean(axis=1)
    top5_vit = np.argsort(vit_avg)[-5:][::-1]
    for vit_layer in top5_vit:
        print(f"  ViT Layer {vit_layer:2d}: mean weight = {vit_avg[vit_layer]:.4f}")


def analyze_pre_router(weights: np.ndarray):
    """Print pre-router weight statistics."""
    print(f"\n{'='*60}")
    print(f"[Analysis] Pre-router weights")
    print(f"{'='*60}")
    print(f"Shape: {weights.shape}")
    print(f"Min: {weights.min():.6f} (ViT layer {weights.argmin()})")
    print(f"Max: {weights.max():.6f} (ViT layer {weights.argmax()})")
    print(f"Mean: {weights.mean():.6f}")
    print(f"Std: {weights.std():.6f}")

    print(f"\nTop-5 ViT layers by weight:")
    top5 = np.argsort(weights)[-5:][::-1]
    for rank, vit_layer in enumerate(top5, 1):
        print(f"  #{rank}: ViT layer {vit_layer:2d} = {weights[vit_layer]:.4f}")

    print(f"\nAll 24 ViT layer weights:")
    for i, w in enumerate(weights):
        bar = '█' * int(w * 50)  # simple text bar
        print(f"  Layer {i:2d}: {w:.4f} {bar}")


def main():
    parser = argparse.ArgumentParser(description='Plot router-weight heatmaps for layerwise vision fusion')
    parser.add_argument(
        '--json-path', '-j',
        type=str,
        default=DEFAULT_JSON_PATH,
        help=f'Path to router-stats JSON (default: {DEFAULT_JSON_PATH})'
    )
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default=None,
        help='Output directory (default: same directory as the input JSON)'
    )
    parser.add_argument(
        '--title', '-t',
        type=str,
        default=None,
        help='Figure title'
    )
    parser.add_argument(
        '--cmap', '-c',
        type=str,
        default='pearson_blue_red',
        choices=['pearson_blue_red', 'RdBu_r', 'coolwarm', 'viridis', 'YlOrRd', 'Blues', 'Reds', 'hot'],
        help='Matplotlib colormap name'
    )
    parser.add_argument(
        '--show-values',
        action='store_true',
        help='Annotate each cell with its numeric value'
    )
    parser.add_argument(
        '--levels',
        type=int,
        default=0,
        help='Number of discrete color levels (0 = continuous; set >1 for stepped colormap)'
    )
    parser.add_argument(
        '--rounded-cells',
        action='store_true',
        help='Draw LLM-router heatmap cells as rounded rectangles (tile style)'
    )
    parser.add_argument(
        '--cell-gap',
        type=float,
        default=0.22,
        help='Gap ratio between rounded cells (larger = smaller tiles; typical 0.15-0.30; default 0.22)'
    )
    parser.add_argument(
        '--rounding-size',
        type=float,
        default=None,
        help='Corner rounding radius (default: auto; oversized values are clipped)'
    )
    parser.add_argument(
        '--analyze',
        action='store_true',
        default=True,
        help='Print router weight statistics (on by default)'
    )
    parser.add_argument(
        '--compare',
        type=str,
        nargs='+',
        default=None,
        help='Additional JSON files to compare against --json-path'
    )
    parser.add_argument(
        '--use-per-sample',
        action='store_true',
        help='Build weights from per_sample_weights (averages all samples unless --question-id/--sample-idx is set)'
    )
    parser.add_argument(
        '--question-id',
        type=str,
        default=None,
        help='Select one sample by question_id for the heatmap; --use-per-sample is implicit'
    )
    parser.add_argument(
        '--sample-idx',
        type=int,
        default=None,
        help='0-based sample index when using --use-per-sample without --question-id; --question-id wins if both are set'
    )
    
    args = parser.parse_args()
    cmap_obj = PEARSON_BLUE_RED_CMAP if args.cmap == 'pearson_blue_red' else args.cmap
    
    print("=" * 80)
    print("Router Weights Heatmap Visualization")
    print("=" * 80)
    
    json_path = args.json_path
    print(f"\n[Load] {json_path}")
    data = load_router_weights(json_path)

    model_name = Path(json_path).stem.replace('-router-weights', '')

    if args.output_dir is None:
        output_dir = Path(json_path).parent
    else:
        output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # question_id filter or build 32x24 weights from per_sample_weights
    use_per_sample_mode = args.use_per_sample or (args.question_id is not None)
    question_id_for_output = args.question_id
    if use_per_sample_mode:
        qid = args.question_id
        sidx = args.sample_idx if qid is None else None
        llm_weights = build_weights_from_per_sample(data, sample_idx=sidx, question_id=qid)
        has_pre_router = False
        print(f"[Info] Built from per_sample_weights: shape {llm_weights.shape} (ViT x LLM)")
        if qid is not None:
            print(f"[Info] Selected sample question_id={qid}")
        elif sidx is not None:
            print(f"[Info] Using sample index {sidx}")
        else:
            n_samples = len(data.get('per_sample_weights', []))
            print(f"[Info] Averaging over {n_samples} samples")
    else:
        # New schema: llm_router.mean_weights; legacy: top-level mean_weights
        if 'llm_router' in data:
            llm_mean_weights = np.array(data['llm_router']['mean_weights'])  # [32, 24]
            has_pre_router = 'pre_router' in data and 'mean_weights' in data['pre_router']
            if has_pre_router:
                pre_router_weights = np.array(data['pre_router']['mean_weights'])  # [24]
        else:
            llm_mean_weights = np.array(data['mean_weights'])  # [32, 24]
            has_pre_router = False

        llm_weights = llm_mean_weights.T  # [24, 32]: rows = ViT, cols = LLM

        print(f"[Info] LLM router weight shape {llm_mean_weights.shape} -> transposed {llm_weights.shape}")
    print(f"[Info] Has pre-router stats: {bool(has_pre_router)}")
    if has_pre_router:
        print(f"[Info] Pre-router weight shape: {pre_router_weights.shape}")

    if args.analyze:
        analyze_weights(llm_weights, f"LLM Router - {model_name}")
        if has_pre_router:
            analyze_pre_router(pre_router_weights)
    
    rounded_tag = "-rounded" if args.rounded_cells else ""
    if question_id_for_output:
        llm_output_path = output_dir / f"{model_name}-llm-router-heatmap{rounded_tag}-q-{question_id_for_output}.png"
        llm_title = args.title if args.title else "Gated Connection Weights on POPE"
    else:
        llm_output_path = output_dir / f"{model_name}-llm-router-heatmap{rounded_tag}.png"
        llm_title = args.title if args.title else "Gated Connection Weights on POPE"
    
    plot_llm_router_heatmap(
        weights=llm_weights,
        output_path=str(llm_output_path),
        title=llm_title,
        cmap=cmap_obj,
        show_values=args.show_values,
        levels=args.levels,
        rounded_cells=args.rounded_cells,
        cell_gap=args.cell_gap,
        rounding_size=args.rounding_size,
    )
    
    if has_pre_router:
        pre_heatmap_path = output_dir / f"{model_name}-pre-router-heatmap.png"
        pre_title = f"Pre-Router Weights: Fusion of 24 ViT Layers\n{model_name}"
        
        plot_pre_router_heatmap(
            weights=pre_router_weights,
            output_path=str(pre_heatmap_path),
            title=pre_title,
            cmap=cmap_obj,
            show_values=True
        )
        
        pre_bar_path = output_dir / f"{model_name}-pre-router-bar.png"
        plot_pre_router_bar(
            weights=pre_router_weights,
            output_path=str(pre_bar_path),
            title=f"Pre-Router Weights Distribution\n{model_name}"
        )
    
    if args.compare:
        all_paths = [json_path] + args.compare
        weights_list = []
        labels = []
        for p in all_paths:
            d = load_router_weights(p)
            if 'llm_router' in d:
                w = np.array(d['llm_router']['mean_weights']).T
            else:
                w = np.array(d['mean_weights']).T
            weights_list.append(w)
            labels.append(Path(p).stem.replace('-router-weights', ''))
        
        compare_output_path = output_dir / 'comparison_heatmap.png'
        plot_comparison_heatmaps(weights_list, labels, str(compare_output_path), cmap_obj)
    
    print("\n" + "=" * 80)
    print("Output figures:")
    print("=" * 80)
    print(f"  1. LLM router heatmap: {llm_output_path}")
    if has_pre_router:
        print(f"  2. Pre-router heatmap: {pre_heatmap_path}")
        print(f"  3. Pre-router bar chart: {pre_bar_path}")
    print("=" * 80)


if __name__ == '__main__':
    main()
