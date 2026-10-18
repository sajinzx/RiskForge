"""Paper-style matplotlib helpers: fixed colour order, thin lines, recessive grid, one axis."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Okabe-Ito colour-blind-safe palette, assigned in a fixed order per entity (never cycled).
PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#7a7a7a"]


def new_axes(title: str, xlabel: str, ylabel: str, size: tuple[float, float] = (6.4, 3.8)):  # type: ignore[no-untyped-def]
    fig, ax = plt.subplots(figsize=size)
    ax.set_title(title, fontsize=11, loc="left")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, color="#e5e5e5", linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    return fig, ax


def save(fig, out_base: Path) -> None:  # type: ignore[no-untyped-def]
    """Write both PNG and PDF next to each other."""
    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_base.with_suffix(".png"), dpi=160)
    fig.savefig(out_base.with_suffix(".pdf"))
    plt.close(fig)
