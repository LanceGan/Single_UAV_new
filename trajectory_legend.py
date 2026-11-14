"""Place trajectory legends inside a plotting axis with minimal path overlap."""

from __future__ import annotations

import numpy as np
from matplotlib.collections import PathCollection
from matplotlib.lines import Line2D
from matplotlib.patches import PathPatch
from matplotlib.path import Path


_CANDIDATES = (
    ("upper right", (0.985, 0.985)),
    ("upper left", (0.015, 0.985)),
    ("lower right", (0.985, 0.015)),
    ("lower left", (0.015, 0.015)),
)
_MARKER_SYMBOLS = {
    "o": "●",
    "^": "▲",
    "x": "x",
    "X": "x",
}
_LINESTYLE_SYMBOLS = {
    "-": "—",
    "solid": "—",
    "--": "--",
    ":": "..",
    "dotted": "..",
}


def _text_legend_entries(handles):
    entries = []
    colors = []
    for handle in handles:
        marker = handle.get_marker()
        linestyle = handle.get_linestyle()
        if marker in _MARKER_SYMBOLS:
            symbol = _MARKER_SYMBOLS[marker]
            color = handle.get_markerfacecolor()
            if str(color).lower() in {"none", "transparent"}:
                color = handle.get_color()
        else:
            symbol = _LINESTYLE_SYMBOLS.get(
                linestyle,
                "--" if isinstance(linestyle, tuple) else "—",
            )
            color = handle.get_color()
        proxy = Line2D([], [], linestyle="none", color="none")
        proxy.set_label(f"{symbol} {handle.get_label()}")
        entries.append(proxy)
        colors.append(color)
    return entries, colors


def _color_legend_text(legend, colors):
    for text, color in zip(legend.get_texts(), colors):
        text.set_color(color)


def _plotted_points(axis):
    point_sets = []
    for line in axis.lines:
        points = np.asarray(line.get_xydata(), dtype=float)
        if points.ndim == 2 and points.shape[1] >= 2 and len(points):
            point_sets.append(points[:, :2])
    for collection in axis.collections:
        try:
            offsets = np.asarray(collection.get_offsets(), dtype=float)
        except (TypeError, ValueError):
            continue
        if offsets.ndim == 2 and offsets.shape[1] >= 2 and len(offsets):
            point_sets.append(offsets[:, :2])
    if not point_sets:
        return np.empty((0, 2), dtype=float)
    return np.concatenate(point_sets, axis=0)


def _overlap_score(legend, axis, plotted_points):
    renderer = axis.figure.canvas.get_renderer()
    legend_bbox = legend.get_window_extent(renderer)
    axis_bbox = axis.get_window_extent(renderer)
    outside = (
        legend_bbox.x0 < axis_bbox.x0 - 0.5
        or legend_bbox.x1 > axis_bbox.x1 + 0.5
        or legend_bbox.y0 < axis_bbox.y0 - 0.5
        or legend_bbox.y1 > axis_bbox.y1 + 0.5
    )
    bbox = legend_bbox.transformed(
        axis.transData.inverted()
    )
    xmin, xmax = sorted((bbox.x0, bbox.x1))
    ymin, ymax = sorted((bbox.y0, bbox.y1))
    inside = (
        (plotted_points[:, 0] >= xmin)
        & (plotted_points[:, 0] <= xmax)
        & (plotted_points[:, 1] >= ymin)
        & (plotted_points[:, 1] <= ymax)
    )
    return int(outside), int(inside.sum())


def _mask_polyline(points, bounds):
    xmin, xmax, ymin, ymax = bounds
    masked_x = [float(points[0, 0])]
    masked_y = [float(points[0, 1])]
    for start, end in zip(points[:-1], points[1:]):
        dx = float(end[0] - start[0])
        dy = float(end[1] - start[1])
        breaks = [0.0, 1.0]
        if dx:
            for boundary in (xmin, xmax):
                value = float((boundary - start[0]) / dx)
                if 0.0 < value < 1.0:
                    y_value = float(start[1] + value * dy)
                    if ymin <= y_value <= ymax:
                        breaks.append(value)
        if dy:
            for boundary in (ymin, ymax):
                value = float((boundary - start[1]) / dy)
                if 0.0 < value < 1.0:
                    x_value = float(start[0] + value * dx)
                    if xmin <= x_value <= xmax:
                        breaks.append(value)
        breaks = sorted(set(breaks))
        for lower, upper in zip(breaks[:-1], breaks[1:]):
            if upper - lower <= 1e-12:
                continue
            midpoint = 0.5 * (lower + upper)
            inside = (
                xmin <= start[0] + midpoint * dx <= xmax
                and ymin <= start[1] + midpoint * dy <= ymax
            )
            if inside:
                masked_x.append(np.nan)
                masked_y.append(np.nan)
            else:
                masked_x.append(float(start[0] + upper * dx))
                masked_y.append(float(start[1] + upper * dy))
    return np.asarray(masked_x), np.asarray(masked_y)


def _mask_strokes_under_legend(axis, legend):
    renderer = axis.figure.canvas.get_renderer()
    bbox = legend.get_window_extent(renderer).transformed(
        axis.transData.inverted()
    )
    xmin, xmax = sorted((bbox.x0, bbox.x1))
    ymin, ymax = sorted((bbox.y0, bbox.y1))
    xpad = 0.01 * (xmax - xmin)
    ypad = 0.01 * (ymax - ymin)
    bounds = (xmin - xpad, xmax + xpad, ymin - ypad, ymax + ypad)
    artists = [
        *axis.lines,
        *axis.get_xgridlines(),
        *axis.get_ygridlines(),
    ]
    for artist in artists:
        points = np.asarray(artist.get_xydata(), dtype=float)
        if points.ndim != 2 or points.shape[1] < 2 or len(points) < 2:
            continue
        masked_x, masked_y = _mask_polyline(points[:, :2], bounds)
        artist.set_data(masked_x, masked_y)
    for collection in axis.collections:
        if not isinstance(collection, PathCollection):
            continue
        if float(collection.get_zorder()) < 4.0:
            continue
        offsets = np.asarray(collection.get_offsets(), dtype=float)
        if offsets.ndim != 2 or offsets.shape[1] < 2 or not len(offsets):
            continue
        inside = (
            (offsets[:, 0] >= bounds[0])
            & (offsets[:, 0] <= bounds[1])
            & (offsets[:, 1] >= bounds[2])
            & (offsets[:, 1] <= bounds[3])
        )
        if inside.any():
            collection.set_offsets(offsets[~inside, :2])


def _clip_map_under_legend(axis, legend):
    renderer = axis.figure.canvas.get_renderer()
    bbox = legend.get_window_extent(renderer).transformed(
        axis.transData.inverted()
    )
    xmin, xmax = sorted((bbox.x0, bbox.x1))
    ymin, ymax = sorted((bbox.y0, bbox.y1))
    xpad = 0.01 * (xmax - xmin)
    ypad = 0.01 * (ymax - ymin)
    x0, x1 = float(axis.get_xlim()[0]), float(axis.get_xlim()[1])
    y0, y1 = float(axis.get_ylim()[0]), float(axis.get_ylim()[1])
    vertices = [
        (x0, y0),
        (x1, y0),
        (x1, y1),
        (x0, y1),
        (x0, y0),
        (xmin - xpad, ymin - ypad),
        (xmin - xpad, ymax + ypad),
        (xmax + xpad, ymax + ypad),
        (xmax + xpad, ymin - ypad),
        (xmin - xpad, ymin - ypad),
    ]
    codes = [
        Path.MOVETO,
        Path.LINETO,
        Path.LINETO,
        Path.LINETO,
        Path.CLOSEPOLY,
        Path.MOVETO,
        Path.LINETO,
        Path.LINETO,
        Path.LINETO,
        Path.CLOSEPOLY,
    ]
    clip_patch = PathPatch(
        Path(vertices, codes),
        transform=axis.transData,
        facecolor="none",
        edgecolor="none",
    )
    artists = [
        *axis.collections,
        *axis.lines,
        *axis.get_xgridlines(),
        *axis.get_ygridlines(),
    ]
    for artist in artists:
        artist.set_clip_path(clip_patch)


def add_internal_legend(
    axis,
    handles,
    *,
    ncol=2,
    fontsize=5.8,
    candidates=_CANDIDATES,
):
    """Add a compact legend within ``axis`` at the least occupied corner."""
    plotted_points = _plotted_points(axis)
    legend_handles, legend_colors = _text_legend_entries(handles)
    best = None
    for order, (location, anchor) in enumerate(candidates):
        legend = axis.legend(
            handles=legend_handles,
            loc=location,
            bbox_to_anchor=anchor,
            bbox_transform=axis.transAxes,
            borderaxespad=0.0,
            ncol=ncol,
            fontsize=fontsize,
            frameon=True,
            fancybox=False,
            framealpha=0.96,
            facecolor="white",
            edgecolor="#C8C8C8",
            handlelength=0.0,
            columnspacing=1.1,
            handletextpad=0.0,
            borderpad=0.24,
            labelspacing=0.22,
        )
        _color_legend_text(legend, legend_colors)
        legend.set_zorder(30)
        axis.figure.canvas.draw()
        score = _overlap_score(legend, axis, plotted_points)
        legend.remove()
        ranked = (score, order)
        if best is None or ranked < best[0]:
            best = (ranked, location, anchor)

    _, location, anchor = best
    legend = axis.legend(
        handles=legend_handles,
        loc=location,
        bbox_to_anchor=anchor,
        bbox_transform=axis.transAxes,
        borderaxespad=0.0,
        ncol=ncol,
        fontsize=fontsize,
        frameon=True,
        fancybox=False,
        framealpha=0.96,
        facecolor="white",
        edgecolor="#C8C8C8",
        handlelength=0.0,
        columnspacing=1.1,
        handletextpad=0.0,
        borderpad=0.24,
        labelspacing=0.22,
    )
    _color_legend_text(legend, legend_colors)
    legend.set_zorder(30)
    axis.figure.canvas.draw()
    _mask_strokes_under_legend(axis, legend)
    _clip_map_under_legend(axis, legend)
    return legend
