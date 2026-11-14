"""Shared transparent trajectory legends and colorbar alignment helpers."""

from __future__ import annotations

from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter, MaxNLocator


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


def add_internal_legend(
    axis,
    handles,
    *,
    ncol=1,
    fontsize=4.6,
):
    """Add one transparent Times New Roman legend at the upper-left corner."""
    legend_handles, legend_colors = _text_legend_entries(handles)
    legend = axis.legend(
        handles=legend_handles,
        loc="upper left",
        bbox_to_anchor=(0.012, 0.988),
        bbox_transform=axis.transAxes,
        borderaxespad=0.0,
        ncol=ncol,
        prop=FontProperties(
            family="Times New Roman",
            size=fontsize,
        ),
        frameon=True,
        fancybox=False,
        framealpha=0.75,
        facecolor="white",
        edgecolor="#A6A6A6",
        handlelength=0.0,
        columnspacing=0.8,
        handletextpad=0.0,
        borderpad=0.0,
        labelspacing=0.10,
    )
    legend.get_frame().set_linewidth(0.5)
    for text, color in zip(legend.get_texts(), legend_colors):
        text.set_color(color)
    legend.set_zorder(20)
    return legend


def align_colorbar_to_axis(axis, color_axis):
    """Match the colorbar's physical vertical extent to the plotting axis."""
    axis.figure.canvas.draw()
    axis_position = axis.get_position()
    color_position = color_axis.get_position()
    color_axis.set_position(
        [
            color_position.x0,
            axis_position.y0,
            color_position.width,
            axis_position.height,
        ]
    )


def simplify_colorbar_ticks(colorbar, *, format_string, nbins=6):
    colorbar.locator = MaxNLocator(nbins=nbins)
    colorbar.formatter = FormatStrFormatter(format_string)
    colorbar.update_ticks()
