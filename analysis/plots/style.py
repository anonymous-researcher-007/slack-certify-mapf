"""Matplotlib paper-style configuration.

Sets a non-interactive Agg backend before importing pyplot so the
analysis pipeline can run headless in CI. Exposes the Okabe-Ito
colourblind-friendly palette and two figure-size context managers.

The Okabe-Ito palette (Wong, *Points of View: Color blindness*, Nature
Methods 2011) is the most widely cited eight-colour palette designed
for protanopia/deuteranopia/tritanopia accessibility.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator

import matplotlib

matplotlib.use("Agg")  # noqa: E402  must precede pyplot import

import matplotlib.pyplot as plt  # noqa: E402

__all__ = [
    "ONE_COLUMN_FIGSIZE",
    "PAPER_PALETTE",
    "TWO_COLUMN_FIGSIZE",
    "apply_paper_style",
    "one_column_fig",
    "two_column_fig",
]


# Okabe-Ito 8-colour palette: black, orange, sky-blue, bluish-green,
# yellow, blue, vermilion, reddish-purple.
PAPER_PALETTE: list[str] = [
    "#000000",
    "#E69F00",
    "#56B4E9",
    "#009E73",
    "#F0E442",
    "#0072B2",
    "#D55E00",
    "#CC79A7",
]


# IEEE single- and double-column figure sizes (inches).
ONE_COLUMN_FIGSIZE: tuple[float, float] = (3.5, 2.5)
TWO_COLUMN_FIGSIZE: tuple[float, float] = (7.0, 2.5)


def apply_paper_style() -> None:
    """Apply paper-style matplotlib rcParams.

    Idempotent: calling this multiple times is safe. The colour cycle
    is set to the Okabe-Ito palette.

    Examples
    --------
    >>> apply_paper_style()
    >>> import matplotlib as _mpl
    >>> _mpl.rcParams["axes.spines.top"]
    False
    """
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "STIX Two Text", "DejaVu Serif"],
            "font.size": 9.0,
            "axes.titlesize": 9.0,
            "axes.labelsize": 9.0,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "legend.fontsize": 8.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linestyle": ":",
            "figure.figsize": ONE_COLUMN_FIGSIZE,
            "figure.dpi": 100,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.format": "pdf",
            "axes.prop_cycle": plt.cycler(color=PAPER_PALETTE),
        }
    )


@contextlib.contextmanager
def one_column_fig() -> Iterator[tuple[plt.Figure, plt.Axes]]:
    """Yield ``(fig, ax)`` at the IEEE single-column size (3.5 × 2.5 in).

    Examples
    --------
    >>> with one_column_fig() as (fig, ax):
    ...     _ = ax.plot([0, 1], [0, 1])
    """
    apply_paper_style()
    fig, ax = plt.subplots(figsize=ONE_COLUMN_FIGSIZE)
    try:
        yield fig, ax
    finally:
        plt.close(fig)


@contextlib.contextmanager
def two_column_fig(
    n_panels: int = 2,
) -> Iterator[tuple[plt.Figure, list[plt.Axes]]]:
    """Yield ``(fig, axes)`` at the IEEE two-column size (7.0 × 2.5 in).

    Examples
    --------
    >>> with two_column_fig(n_panels=2) as (fig, axes):
    ...     _ = axes[0].plot([0, 1], [0, 1])
    """
    apply_paper_style()
    fig, axes = plt.subplots(1, n_panels, figsize=TWO_COLUMN_FIGSIZE, squeeze=False)
    axes_list: list[plt.Axes] = list(axes[0])
    try:
        yield fig, axes_list
    finally:
        plt.close(fig)
