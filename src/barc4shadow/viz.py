from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np


_DEFAULT_COLOR_MAP = {
    "SRC": "darkred",
    "M": "olive",
    "ML": "darkgreen",
    "C": "indigo",
    "G": "steelblue",
    "S": "teal",
    "L": "peru",
    "CRL": "darkorange",
    "CMP": "slategray",
    "O": "teal",
    "X": "peru",
}

_DEFAULT_MARKER_MAP = {
    "SRC": "*",
    "M": "s",
    "ML": "H",
    "C": "p",
    "G": "D",
    "S": "o",
    "L": "^",
    "CRL": "v",
    "CMP": "h",
    "O": "o",
    "X": "X",
}

_DEFAULT_LEGEND_MAP = {
    "SRC": "source",
    "M": "mirror",
    "ML": "multilayer",
    "C": "crystal",
    "G": "grating",
    "S": "screen",
    "L": "lens",
    "CRL": "CRL",
    "CMP": "compound",
    "O": "obs. point",
    "X": "obs. point",
}


def plot_beamline(
    layout: dict[str, Any],
    show_source: bool = True,
    show_experiment: bool = True,
    draw_to_scale: bool = False,
    k: float = 1.0,
) -> None:
    """
    Plot a beamline layout.

    Parameters
    ----------
    layout : dict
        Layout dictionary with ``x``, ``y``, ``z`` arrays and
        ``elements["labels"]`` / ``elements["kinds"]``.
    show_source : bool, optional
        If False, omit the source marker.
    show_experiment : bool, optional
        If True and the final element is empty, display it as an observation
        point using kind ``X``.
    draw_to_scale : bool, optional
        If True, draw separate top and side views with equal metric scale.
    k : float, optional
        Font scaling factor.
    """
    try:
        start_plotting(k)
    except NameError:
        pass

    x = np.asarray(layout["x"], dtype=float)
    y = np.asarray(layout["y"], dtype=float)
    z = np.asarray(layout["z"], dtype=float)

    labels = list(layout["elements"]["labels"])
    kinds = list(layout["elements"]["kinds"])

    if len(x) != len(y) or len(x) != len(z):
        raise ValueError("`x`, `y`, and `z` must have the same length.")

    if len(labels) != len(x):
        raise ValueError("`elements['labels']` must match coordinate length.")

    if len(kinds) != len(x):
        raise ValueError("`elements['kinds']` must match coordinate length.")

    idxs = []

    for i, kind in enumerate(kinds):
        is_source = i == 0 and kind == "SRC"
        is_last = i == len(kinds) - 1

        if is_source and not show_source:
            continue

        if kind == "E":
            if show_experiment and is_last:
                idxs.append(i)
            continue

        idxs.append(i)

    if not idxs:
        raise ValueError("Nothing to plot after filtering.")

    def display_kind(i: int) -> str:
        if show_experiment and i == len(kinds) - 1 and kinds[i] == "E":
            return "X"
        return kinds[i]

    def style(ax, ylabel: str) -> None:
        ax.set_facecolor("white")
        ax.grid(True, which="both", color="gray", linestyle=":", linewidth=0.5)
        ax.tick_params(direction="in", top=True, right=True)
        ax.set_ylabel(ylabel)

        for spine in ("top", "right", "bottom", "left"):
            ax.spines[spine].set_color("black")

    def plot_points(ax, which: str, add_legend: bool = True) -> None:
        V = x if which == "top" else z

        ax.plot(y[idxs], V[idxs], color="0.6", lw=0.8, zorder=1)

        seen = set()

        for i in idxs:
            kind = display_kind(i)
            color = _DEFAULT_COLOR_MAP.get(kind, "black")
            marker = _DEFAULT_MARKER_MAP.get(kind, "o")
            label = _DEFAULT_LEGEND_MAP.get(kind) if add_legend and kind not in seen else None
            seen.add(kind)

            if kind == "O":
                ax.plot(
                    y[i],
                    V[i],
                    linestyle="none",
                    marker=marker,
                    markerfacecolor=color,
                    markeredgecolor="black",
                    markeredgewidth=0.8,
                    fillstyle="left",
                    markersize=9,
                    zorder=3,
                    label=label,
                )
            elif kind in {"M", "ML", "G", "C", "S", "L", "CRL", "CMP"}:
                ax.plot(
                    y[i],
                    V[i],
                    linestyle="none",
                    marker=marker,
                    markerfacecolor=color,
                    markeredgecolor="black",
                    markeredgewidth=0.5,
                    markersize=9,
                    zorder=3,
                    label=label,
                )
            else:
                ax.plot(
                    y[i],
                    V[i],
                    linestyle="none",
                    marker=marker,
                    markerfacecolor=color,
                    markeredgecolor="black",
                    markeredgewidth=0.5,
                    markersize=11,
                    zorder=3,
                    label=label,
                )

        if add_legend:
            handles, _ = ax.get_legend_handles_labels()
            if handles:
                ax.legend(loc="best", frameon=True)

    def square_limits_match_x(ax, Y: np.ndarray, V: np.ndarray) -> None:
        ymin, ymax = float(np.nanmin(Y)), float(np.nanmax(Y))
        yspan = max(ymax - ymin, 1e-9)
        yctr = 0.5 * (ymax + ymin)

        vmin, vmax = float(np.nanmin(V)), float(np.nanmax(V))
        vctr = 0.5 * (vmax + vmin)

        pad = 0.02 * yspan

        ax.set_xlim(yctr - 0.5 * yspan - pad, yctr + 0.5 * yspan + pad)
        ax.set_ylim(vctr - 0.5 * yspan - pad, vctr + 0.5 * yspan + pad)
        ax.set_aspect("equal", adjustable="box")

    if draw_to_scale:
        fig_top, ax_top = plt.subplots(figsize=(12, 12))
        fig_top.suptitle("Beamline layout — Top view (to scale)", fontsize=16 * k)
        style(ax_top, "top view [m]")
        plot_points(ax_top, which="top", add_legend=True)
        ax_top.set_xlabel("[m]")
        square_limits_match_x(ax_top, y[idxs], x[idxs])

        fig_side, ax_side = plt.subplots(figsize=(12, 12))
        fig_side.suptitle("Beamline layout — Side view (to scale)", fontsize=16 * k)
        style(ax_side, "side view [m]")
        plot_points(ax_side, which="side", add_legend=False)
        ax_side.set_xlabel("[m]")
        square_limits_match_x(ax_side, y[idxs], z[idxs])

        plt.show()

    else:
        fig, (ax_top, ax_side) = plt.subplots(
            2,
            1,
            sharex=True,
            figsize=(12, 6),
            gridspec_kw={"height_ratios": [1, 1]},
        )

        fig.suptitle("Beamline layout", fontsize=16 * k, x=0.5)

        style(ax_top, "top view [m]")
        style(ax_side, "side view [m]")
        ax_side.set_xlabel("[m]")

        plot_points(ax_top, which="top", add_legend=True)
        plot_points(ax_side, which="side", add_legend=False)

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()