from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

_PAIRED = plt.get_cmap("Paired").colors

_DEFAULT_COLOR_MAP = {
    "SRC": "darkred",
    "M": _PAIRED[0],
    "ML": _PAIRED[1],
    "C": _PAIRED[2],
    "G": _PAIRED[3],
    "BS": _PAIRED[4],
    "F": _PAIRED[5],
    "L": _PAIRED[6],
    "CRL": _PAIRED[7],
    "CMP": _PAIRED[8],
    "O": _PAIRED[9],
}

_DEFAULT_MARKER_MAP = {
    "SRC": "*",
    "M": "s",
    "ML": "H",
    "C": "p",
    "G": "D",
    "L": "^",
    "CRL": "v",
    "CMP": "h",
    "O": "o",
    "SL": "|",
    "BS": "x",
    "F":  "P",
}

_DEFAULT_LEGEND_MAP = {
    "SRC": "source",
    "M": "mirror",
    "ML": "multilayer",
    "C": "crystal",
    "G": "grating",
    "L": "lens",
    "CRL": "CRL",
    "CMP": "compound",
    "O": "obs. point",
    "SL": "slit",
    "BS": "obstruction",
    "F":  "filter",
}


def plot_beamline(
    layout: dict[str, Any],
    *,
    show_source: bool = True,
    show_experiment: bool = True,
    draw_to_scale: bool = False,
    k: float = 1.0,
    min_transverse_half_range: float | None = 1e-6,
    scale_transverse_fraction: float = 0.20,
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
    min_transverse_half_range : float or None, optional
        Minimum displayed half-range for transverse coordinates when
        ``draw_to_scale=False``. If the actual span is smaller than twice this
        value, limits are expanded around the data centre. Use None to disable.
    scale_transverse_fraction : float, optional
        Minimum full transverse span as a fraction of the optical-axis span
        when ``draw_to_scale=True``.
    """
    scale_fig_width = 12.0
    scale_min_fig_height = 2.5
    scale_max_fig_height = 8.0
    scale_padding_fraction = 0.03
    scale_transverse_margin = 1.20

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
            return "O"
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
            if kind == "SL":
                color = "black"
            else:
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
                    markeredgewidth=0.75,
                    fillstyle="left",
                    markersize=10,
                    zorder=3,
                    label=label,
                )
            elif kind == "SL":
                    ax.plot(
                        y[i],
                        V[i],
                        linestyle="none",
                        marker=marker,
                        markerfacecolor=color,
                        markeredgecolor="black",
                        markeredgewidth=1,
                        markersize=10,
                        zorder=3,
                        label=label,
                    )
            elif kind in {"M", "ML", "G", "C", "L", "CRL", "CMP", "F"}:
                ax.plot(
                    y[i],
                    V[i],
                    linestyle="none",
                    marker=marker,
                    markerfacecolor=color,
                    markeredgecolor="black",
                    markeredgewidth=0.75,
                    markersize=10,
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
                    markeredgewidth=0.75,
                    markersize=10,
                    zorder=3,
                    label=label,
                )

        if add_legend:
            handles, _ = ax.get_legend_handles_labels()
            if handles:
                ax.legend(loc="best", frameon=True)

    def finite_values(values: np.ndarray) -> np.ndarray:
        vals = np.asarray(values, dtype=float)
        vals = vals[np.isfinite(vals)]

        if vals.size == 0:
            raise ValueError("Cannot determine plot limits from non-finite coordinates.")

        return vals

    def finite_limits(values: np.ndarray) -> tuple[float, float]:
        vals = finite_values(values)

        vmin = float(np.min(vals))
        vmax = float(np.max(vals))

        if np.isclose(vmin, vmax):
            pad = max(abs(vmin) * 0.01, 1e-12)
            return vmin - pad, vmax + pad

        return vmin, vmax

    def padded_limits(values: np.ndarray, pad_fraction: float) -> tuple[float, float]:
        vmin, vmax = finite_limits(values)
        span = vmax - vmin
        pad = pad_fraction * span
        return vmin - pad, vmax + pad

    def apply_min_transverse_range(ax, values: np.ndarray) -> None:
        if min_transverse_half_range is None:
            return

        vmin, vmax = finite_limits(values)
        span = vmax - vmin
        min_span = 2.0 * min_transverse_half_range

        if span < min_span:
            ax.set_ylim(
                -min_transverse_half_range,
                min_transverse_half_range,
            )

    def common_scaled_limits() -> tuple[
        tuple[float, float],
        tuple[float, float],
        float,
    ]:
        xlim = padded_limits(y[idxs], scale_padding_fraction)
        horizontal_span = xlim[1] - xlim[0]

        x_vals = finite_values(x[idxs])
        z_vals = finite_values(z[idxs])

        max_abs_dev = max(
            float(np.max(np.abs(x_vals))),
            float(np.max(np.abs(z_vals))),
        )

        common_half_range = max(
            0.5 * scale_transverse_fraction * horizontal_span,
            scale_transverse_margin * max_abs_dev,
        )

        ylim = (-common_half_range, common_half_range)

        fig_height = scale_fig_width * (2.0 * common_half_range) / horizontal_span
        fig_height = float(np.clip(fig_height, scale_min_fig_height, scale_max_fig_height))

        return xlim, ylim, fig_height

    # def plot_scaled_view(
    #     which: str,
    #     title: str,
    #     ylabel: str,
    #     add_legend: bool,
    #     xlim: tuple[float, float],
    #     ylim: tuple[float, float],
    #     fig_height: float,
    # ) -> None:
    #     fig, ax = plt.subplots(figsize=(scale_fig_width, fig_height))
    #     fig.suptitle(title, fontsize=16 * k)

    #     style(ax, ylabel)
    #     plot_points(ax, which=which, add_legend=add_legend)

    #     ax.set_xlabel("[m]")
    #     ax.set_xlim(*xlim)
    #     ax.set_ylim(*ylim)
    #     ax.set_aspect("equal", adjustable="box")

    if draw_to_scale:
        xlim, ylim, fig_height = common_scaled_limits()

        fig, (ax_top, ax_side) = plt.subplots(
            2,
            1,
            sharex=True,
            figsize=(12.0, 2.0 * fig_height),
            gridspec_kw={"height_ratios": [1, 1]},
        )

        fig.suptitle("Beamline layout — To scale", fontsize=16 * k, x=0.5)

        style(ax_top, "top view [m]")
        style(ax_side, "side view [m]")
        ax_side.set_xlabel("[m]")

        plot_points(ax_top, which="top", add_legend=True)
        plot_points(ax_side, which="side", add_legend=False)

        ax_top.set_xlim(*xlim)
        ax_side.set_xlim(*xlim)

        ax_top.set_ylim(*ylim)
        ax_side.set_ylim(*ylim)

        ax_top.set_aspect("equal", adjustable="box")
        ax_side.set_aspect("equal", adjustable="box")

        plt.tight_layout(rect=[0, 0, 1, 0.95])
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

        apply_min_transverse_range(ax_top, x[idxs])
        apply_min_transverse_range(ax_side, z[idxs])

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()

def plot_beamline_configs(
    configs: Sequence[dict[str, Any]],
    config_labels: Sequence[str],
    *,
    show_source: bool = True,
    show_experiment: bool = True,
    draw_to_scale: bool = False,
    k: float = 1.0,
    min_transverse_half_range: float | None = 1e-6,
    scale_transverse_fraction: float = 0.20,
) -> None:
    """
    Overlay multiple beamline layouts.

    Parameters
    ----------
    configs : sequence of dict
        Layout dictionaries with ``x``, ``y``, ``z`` arrays and
        ``elements["labels"]`` / ``elements["kinds"]``.
    config_labels : sequence of str
        One legend label per configuration.
    show_source : bool, optional
        If False, omit the source marker.
    show_experiment : bool, optional
        If True and the final element is empty, display it as an observation
        point using kind ``X``.
    draw_to_scale : bool, optional
        If True, draw top and side views with equal metric scale.
    k : float, optional
        Font scaling factor.
    min_transverse_half_range : float or None, optional
        Minimum displayed half-range for transverse coordinates when
        ``draw_to_scale=False``.
    scale_transverse_fraction : float, optional
        Minimum full transverse span as a fraction of the optical-axis span
        when ``draw_to_scale=True``.
    """
    scale_fig_width = 12.0
    scale_min_fig_height = 2.5
    scale_max_fig_height = 8.0
    scale_padding_fraction = 0.03
    scale_transverse_margin = 1.20

    try:
        start_plotting(k)
    except NameError:
        pass

    if not configs:
        raise ValueError("`configs` is empty.")

    if len(config_labels) != len(configs):
        raise ValueError(
            f"`config_labels` must have length {len(configs)}, "
            f"got {len(config_labels)}."
        )

    config_colors = [
        "darkred",
        "olive",
        "steelblue",
        "teal",
        "peru",
        "slategray",
        "darkgreen",
        "indigo",
        "darkorange",
    ]

    prepared = []

    for layout, cfg_label in zip(configs, config_labels):
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

        if idxs:
            prepared.append(
                {
                    "x": x,
                    "y": y,
                    "z": z,
                    "kinds": kinds,
                    "idxs": idxs,
                    "label": str(cfg_label),
                }
            )

    if not prepared:
        raise ValueError("Nothing to plot after filtering.")

    def display_kind(kinds: list[str], i: int) -> str:
        if show_experiment and i == len(kinds) - 1 and kinds[i] == "E":
            return "O"
        return kinds[i]

    def style(ax, ylabel: str) -> None:
        ax.set_facecolor("white")
        ax.grid(True, which="both", color="gray", linestyle=":", linewidth=0.5)
        ax.tick_params(direction="in", top=True, right=True)
        ax.set_ylabel(ylabel)

        for spine in ("top", "right", "bottom", "left"):
            ax.spines[spine].set_color("black")

    def plot_points(ax, which: str, add_legend: bool = True) -> None:
        for i_cfg, data in enumerate(prepared):
            color = config_colors[i_cfg % len(config_colors)]
            idxs = data["idxs"]
            x = data["x"]
            y = data["y"]
            z = data["z"]
            kinds = data["kinds"]
            V = x if which == "top" else z

            ax.plot(
                y[idxs],
                V[idxs],
                color=color,
                lw=1.0,
                alpha=0.75,
                zorder=1,
                label=data["label"] if add_legend else None,
            )

            for i in idxs:
                kind = display_kind(kinds, i)
                marker = _DEFAULT_MARKER_MAP.get(kind, "o")

                if kind == "O":
                    ax.plot(
                        y[i],
                        V[i],
                        linestyle="none",
                        marker=marker,
                        markerfacecolor=color,
                        markeredgecolor="black",
                        markeredgewidth=0.75,
                        fillstyle="left",
                        markersize=10,
                        zorder=3,
                    )
                elif kind == "SL":
                    ax.plot(
                        y[i],
                        V[i],
                        linestyle="none",
                        marker=marker,
                        markerfacecolor=color,
                        markeredgecolor=color,
                        markeredgewidth=1,
                        markersize=10,
                        zorder=3,
                    )
                elif kind in {"M", "ML", "G", "C", "L", "CRL", "CMP", "F"}:
                    ax.plot(
                        y[i],
                        V[i],
                        linestyle="none",
                        marker=marker,
                        markerfacecolor=color,
                        markeredgecolor="black",
                        markeredgewidth=0.75,
                        markersize=10,
                        zorder=3,
                    )
                else:
                    ax.plot(
                        y[i],
                        V[i],
                        linestyle="none",
                        marker=marker,
                        markerfacecolor=color,
                        markeredgecolor="black",
                        markeredgewidth=0.75,
                        markersize=10,
                        zorder=3,
                    )

        if add_legend:
            handles, _ = ax.get_legend_handles_labels()
            if handles:
                ax.legend(loc="best", frameon=True)

    def finite_values(values: np.ndarray) -> np.ndarray:
        vals = np.asarray(values, dtype=float)
        vals = vals[np.isfinite(vals)]

        if vals.size == 0:
            raise ValueError("Cannot determine plot limits from non-finite coordinates.")

        return vals

    def finite_limits(values: np.ndarray) -> tuple[float, float]:
        vals = finite_values(values)

        vmin = float(np.min(vals))
        vmax = float(np.max(vals))

        if np.isclose(vmin, vmax):
            pad = max(abs(vmin) * 0.01, 1e-12)
            return vmin - pad, vmax + pad

        return vmin, vmax

    def padded_limits(values: np.ndarray, pad_fraction: float) -> tuple[float, float]:
        vmin, vmax = finite_limits(values)
        span = vmax - vmin
        pad = pad_fraction * span
        return vmin - pad, vmax + pad

    def apply_min_transverse_range(ax, values: np.ndarray) -> None:
        if min_transverse_half_range is None:
            return

        vmin, vmax = finite_limits(values)
        span = vmax - vmin
        min_span = 2.0 * min_transverse_half_range

        if span < min_span:
            ax.set_ylim(
                -min_transverse_half_range,
                min_transverse_half_range,
            )

    def all_axis_values(axis: str) -> np.ndarray:
        vals = []

        for data in prepared:
            idxs = data["idxs"]
            vals.append(data[axis][idxs])

        return np.concatenate(vals)

    def common_scaled_limits() -> tuple[
        tuple[float, float],
        tuple[float, float],
        float,
    ]:
        y_all = all_axis_values("y")
        x_all = all_axis_values("x")
        z_all = all_axis_values("z")

        xlim = padded_limits(y_all, scale_padding_fraction)
        horizontal_span = xlim[1] - xlim[0]

        x_vals = finite_values(x_all)
        z_vals = finite_values(z_all)

        max_abs_dev = max(
            float(np.max(np.abs(x_vals))),
            float(np.max(np.abs(z_vals))),
        )

        common_half_range = max(
            0.5 * scale_transverse_fraction * horizontal_span,
            scale_transverse_margin * max_abs_dev,
        )

        ylim = (-common_half_range, common_half_range)

        fig_height = scale_fig_width * (2.0 * common_half_range) / horizontal_span
        fig_height = float(np.clip(fig_height, scale_min_fig_height, scale_max_fig_height))

        return xlim, ylim, fig_height

    if draw_to_scale:
        xlim, ylim, fig_height = common_scaled_limits()

        fig, (ax_top, ax_side) = plt.subplots(
            2,
            1,
            sharex=True,
            figsize=(12.0, 2.0 * fig_height),
            gridspec_kw={"height_ratios": [1, 1]},
        )

        fig.suptitle("Beamline configurations — To scale", fontsize=16 * k, x=0.5)

        style(ax_top, "top view [m]")
        style(ax_side, "side view [m]")
        ax_side.set_xlabel("[m]")

        plot_points(ax_top, which="top", add_legend=True)
        plot_points(ax_side, which="side", add_legend=False)

        ax_top.set_xlim(*xlim)
        ax_side.set_xlim(*xlim)

        ax_top.set_ylim(*ylim)
        ax_side.set_ylim(*ylim)

        ax_top.set_aspect("equal", adjustable="box")
        ax_side.set_aspect("equal", adjustable="box")

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()

    else:
        fig, (ax_top, ax_side) = plt.subplots(
            2,
            1,
            sharex=True,
            figsize=(12, 6),
            gridspec_kw={"height_ratios": [1, 1]},
        )

        fig.suptitle("Beamline configurations", fontsize=16 * k, x=0.5)

        style(ax_top, "top view [m]")
        style(ax_side, "side view [m]")
        ax_side.set_xlabel("[m]")

        plot_points(ax_top, which="top", add_legend=True)
        plot_points(ax_side, which="side", add_legend=False)

        apply_min_transverse_range(ax_top, all_axis_values("x"))
        apply_min_transverse_range(ax_side, all_axis_values("z"))

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()