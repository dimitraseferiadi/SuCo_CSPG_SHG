"""Shared paper-style configuration for thesis benchmark plots.

The canonical 5-index palette is anchored on the cross-algorithm Pareto
comparison (cf. ``plot_router_paper.py`` and the "Cross-Algorithm Results"
section of the experimental chapter). All thesis figures should resolve
colours and markers via this module so the algorithm identity stays
consistent across chapters.

Variants and baselines outside the canonical set (e.g. ``SHG-NO-LB``,
``CSPG(m=2,lambda=0.5)``, ``IVFFlat``, ``LSH``) fall back to a
complementary colourblind-friendly palette. Variants of a canonical
algorithm reuse its base hue with a different shade / linestyle so the
algorithm family stays visually grouped.
"""

from __future__ import annotations

import os
import re


CANONICAL_INDICES = ("SuCo", "SHG", "CSPG", "HNSW32", "HNSW48")

INDEX_COLOR = {
    "SuCo":   "#1f77b4",
    "SHG":    "#f4a261",
    "CSPG":   "#2ca02c",
    "HNSW32": "#d62728",
    "HNSW48": "#9467bd",
}

INDEX_MARKER = {
    "SuCo":   "o",
    "SHG":    "s",
    "CSPG":   "D",
    "HNSW32": "^",
    "HNSW48": "v",
}


_EXTRA_COLOR = {
    "HNSW":     "#d62728",
    "IVFFlat":  "#17becf",
    "IVFPQ":    "#bcbd22",
    "OPQ":      "#8c564b",
    "LSH":      "#e377c2",
    "PANORAMA": "#7f7f7f",

    "DynAct":   "#1f77b4",
    "MultiSeq": "#6baed6",

    "SHG-NO-SHORTCUT": "#e07b39",
    "SHG-NO-LB":       "#c4651e",
    "SHG-NO-BOTH":     "#9c5215",

    "CSPG-HNSW": "#2ca02c",
}

_EXTRA_MARKER = {
    "HNSW":     "^",
    "IVFFlat":  "P",
    "IVFPQ":    "X",
    "OPQ":      "*",
    "LSH":      "x",
    "PANORAMA": "h",

    "DynAct":   "o",
    "MultiSeq": "s",

    "SHG-NO-SHORTCUT": "x",
    "SHG-NO-LB":       "+",
    "SHG-NO-BOTH":     "*",

    "CSPG-HNSW": "D",
}


def _canon(name: str) -> str:
    """Normalise variant names like 'shg_no_lb' or 'IVFFLAT' to canonical form."""
    if not name:
        return ""
    s = str(name).strip()
    # Direct hits first
    if s in INDEX_COLOR or s in _EXTRA_COLOR:
        return s
    upper = s.upper().replace("_", "-")
    lookup = {k.upper().replace("_", "-"): k for k in {**INDEX_COLOR, **_EXTRA_COLOR}}
    return lookup.get(upper, s)


def color_for(name: str, default: str = "#555555") -> str:
    """Resolve an index/series name to its paper-standard colour."""
    key = _canon(name)
    if key in INDEX_COLOR:
        return INDEX_COLOR[key]
    return _EXTRA_COLOR.get(key, default)


def marker_for(name: str, default: str = "o") -> str:
    """Resolve an index/series name to its paper-standard marker."""
    key = _canon(name)
    if key in INDEX_MARKER:
        return INDEX_MARKER[key]
    return _EXTRA_MARKER.get(key, default)


# Variant shades, used when a single plot draws several sub-variants of the
# same algorithm family (e.g. an ablation grid).
def shade(hex_color: str, factor: float) -> str:
    """Lighten (factor > 0) or darken (factor < 0) a hex colour."""
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    if factor >= 0:
        r = int(r + (255 - r) * factor)
        g = int(g + (255 - g) * factor)
        b = int(b + (255 - b) * factor)
    else:
        f = 1.0 + factor
        r = int(r * f); g = int(g * f); b = int(b * f)
    return f"#{r:02x}{g:02x}{b:02x}"


def variant_palette(base: str, n: int) -> list[str]:
    """Return ``n`` shades of the canonical colour for ``base``."""
    c = color_for(base)
    if n <= 1:
        return [c]
    factors = [-0.35 + 0.7 * i / max(1, n - 1) for i in range(n)]
    return [shade(c, f) for f in factors]


# ---------------------------------------------------------------------------
# Global paper-polish rcParams
# ---------------------------------------------------------------------------

PAPER_RCPARAMS = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Linux Libertine O",
                   "DejaVu Serif"],
    "mathtext.fontset": "dejavuserif",
    "font.size": 9.5,
    "axes.titlesize": 10.5,
    "axes.labelsize": 9.5,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8.5,
    "legend.title_fontsize": 9,
    "legend.frameon": True,
    "legend.framealpha": 0.95,
    "legend.fancybox": False,
    "legend.edgecolor": "0.65",
    "legend.borderpad": 0.35,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.7,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "xtick.minor.width": 0.5,
    "ytick.minor.width": 0.5,
    "xtick.major.size": 3.0,
    "ytick.major.size": 3.0,
    "xtick.minor.size": 2.0,
    "ytick.minor.size": 2.0,
    "lines.linewidth": 1.5,
    "lines.markersize": 5.0,
    "patch.linewidth": 0.5,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "grid.linewidth": 0.4,
    "grid.linestyle": ":",
    "grid.alpha": 0.4,
    "axes.grid": False,
    "figure.dpi": 150,
    "savefig.dpi": 220,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.04,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}


def apply_paper_style() -> None:
    """Install the shared paper rcParams. Call once near script start."""
    import matplotlib.pyplot as plt
    plt.rcParams.update(PAPER_RCPARAMS)


def grid_for(n: int, max_cols: int = 3) -> tuple[int, int]:
    """Return ``(rows, cols)`` for a panel grid of ``n`` subplots.

    Small panel counts (``n <= max_cols + 1``) stay on a single row -- so
    a 4-panel figure renders as 1 x 4 rather than 2 x 2. Larger counts
    wrap at ``max_cols`` columns, so a 6-panel figure renders as 3 x 2
    (three on top, three below) rather than a flat 1 x 6 strip.
    """
    if n <= 0:
        return 1, 1
    if n <= max_cols + 1:
        return 1, n
    cols = min(max_cols, n)
    rows = (n + cols - 1) // cols
    return rows, cols


def clean_ax(ax) -> None:
    """Standard axis cleanup: hide top/right spines, place grid behind data."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("0.2")
    ax.spines["bottom"].set_color("0.2")
    ax.set_axisbelow(True)


def save_paper_fig(fig, out_dir: str, name: str,
                   formats=("png", "pdf")) -> None:
    """Save ``fig`` under ``out_dir`` with the standard paper formats."""
    import matplotlib.pyplot as plt
    os.makedirs(out_dir, exist_ok=True)
    for fmt in formats:
        fig.savefig(os.path.join(out_dir, f"{name}.{fmt}"),
                    bbox_inches="tight")
    plt.close(fig)
