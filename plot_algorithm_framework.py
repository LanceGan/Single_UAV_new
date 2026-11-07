"""Publication figure for the proposed dual-radio-map-aware PRISM-TD3 method.

The drawing is deliberately self-contained: it does not read training results or
radio-map files, so regenerating the framework cannot disturb running experiments.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = ROOT / "paper_figures" / "algorithm_framework"

COLORS = {
    "ink": "#273444",
    "muted": "#637083",
    "line": "#AAB4C3",
    "panel": "#F7F9FC",
    "white": "#FFFFFF",
    "g2a": "#0072B2",
    "g2a_light": "#E5F3FA",
    "a2g": "#D55E00",
    "a2g_light": "#FFF0E8",
    "iga": "#009E73",
    "iga_light": "#E6F6F1",
    "td3": "#7A5195",
    "td3_light": "#F1EAF5",
    "gold": "#E69F00",
    "gold_light": "#FFF7DD",
    "success": "#2D8A54",
}


def _rounded_box(ax, x, y, w, h, face, edge, *, radius=0.12, lw=0.9,
                 linestyle="-", zorder=2):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.025,rounding_size={radius}",
        facecolor=face, edgecolor=edge, linewidth=lw,
        linestyle=linestyle, zorder=zorder,
    )
    ax.add_patch(patch)
    return patch


def _text(ax, x, y, text, *, size=6.0, weight="normal", color=None,
          ha="center", va="center", zorder=8, linespacing=1.15, rotation=0):
    return ax.text(
        x, y, text, fontsize=max(float(size), 5.0), fontweight=weight,
        color=color or COLORS["ink"], ha=ha, va=va,
        family="DejaVu Sans", linespacing=linespacing, rotation=rotation,
        rotation_mode="anchor", zorder=zorder,
    )


def _arrow(ax, start, end, *, color=None, lw=1.0, dashed=False,
           connection="arc3", mutation=7, zorder=5):
    arrow = FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=mutation,
        linewidth=lw, color=color or COLORS["ink"],
        linestyle="--" if dashed else "-", connectionstyle=connection,
        shrinkA=1.5, shrinkB=1.5, zorder=zorder,
    )
    ax.add_patch(arrow)
    return arrow


def _stage_panel(ax, x, y, w, h, number, title, color):
    _rounded_box(ax, x, y, w, h, COLORS["panel"], COLORS["line"], radius=0.16, lw=0.8)
    ax.add_patch(Rectangle((x, y + h - 0.62), w, 0.62, facecolor=color,
                           edgecolor=color, zorder=3))
    _rounded_box(ax, x + 0.14, y + h - 0.50, 0.36, 0.36,
                 COLORS["white"], COLORS["white"], radius=0.18, lw=0)
    _text(ax, x + 0.32, y + h - 0.32, str(number), size=6.6,
          weight="bold", color=color)
    _text(ax, x + 0.66, y + h - 0.31, title, size=6.8,
          weight="bold", color=COLORS["white"], ha="left")


def _mini_heatmap(ax, x, y, w, h, title, color, light, reverse=False):
    _rounded_box(ax, x, y, w, h, COLORS["white"], color, radius=0.08, lw=0.8)
    rows, cols = 4, 6
    values = np.array([
        [0.15, 0.25, 0.55, 0.85, 0.62, 0.20],
        [0.35, 0.70, 0.92, 0.60, 0.28, 0.18],
        [0.80, 0.58, 0.30, 0.15, 0.42, 0.76],
        [0.62, 0.22, 0.12, 0.44, 0.88, 0.68],
    ])
    if reverse:
        values = 1.0 - values
    rgb_light = np.array(mpl.colors.to_rgb(light))
    rgb_dark = np.array(mpl.colors.to_rgb(color))
    gx, gy = x + 0.10, y + 0.42
    gw, gh = w - 0.20, h - 0.55
    for row in range(rows):
        for col in range(cols):
            value = values[row, col]
            cell_color = rgb_light * (1 - value) + rgb_dark * value
            ax.add_patch(Rectangle(
                (gx + col * gw / cols, gy + (rows - 1 - row) * gh / rows),
                gw / cols + 0.005, gh / rows + 0.005,
                facecolor=cell_color, edgecolor="none", zorder=4,
            ))
    _text(ax, x + w / 2, y + 0.22, title, size=4.5, weight="bold", color=color,
          linespacing=0.95)


def _scenario_card(ax, x, y, w, h):
    _rounded_box(ax, x, y, w, h, COLORS["white"], COLORS["line"], radius=0.09, lw=0.8)
    _text(ax, x + 0.13, y + h - 0.18, "Fixed inspection scenario", size=5.5,
          weight="bold", ha="left")
    points = np.array([
        [x + 0.30, y + 0.35], [x + 0.78, y + 0.92], [x + 1.30, y + 0.45],
        [x + 1.73, y + 1.03], [x + 2.18, y + 0.58], [x + 2.62, y + 1.06],
        [x + 3.02, y + 0.40],
    ])
    ax.plot(points[:, 0], points[:, 1], color=COLORS["line"], linewidth=0.7,
            linestyle=":", zorder=4)
    for idx, (px, py) in enumerate(points):
        if idx == 0:
            marker, fc, label = "D", COLORS["success"], "S"
        elif idx == len(points) - 1:
            marker, fc, label = "s", COLORS["gold"], "E"
        else:
            marker, fc, label = "o", COLORS["iga"], str(idx)
        ax.scatter([px], [py], s=22 if idx in (0, len(points) - 1) else 18,
                   marker=marker, facecolor=fc, edgecolor=COLORS["white"],
                   linewidth=0.5, zorder=6)
        _text(ax, px, py - 0.19, label, size=4.8, weight="bold", color=COLORS["muted"])
    for px, py in ((x + 1.03, y + 0.28), (x + 2.40, y + 0.28)):
        tri = Polygon([(px, py), (px - 0.10, py - 0.18), (px + 0.10, py - 0.18)],
                      closed=True, facecolor=COLORS["g2a"], edgecolor="none", zorder=5)
        ax.add_patch(tri)
        ax.plot([px, px], [py, py + 0.19], color=COLORS["g2a"], lw=0.7, zorder=5)


def _weighted_graph(ax, x, y, w, h):
    _rounded_box(ax, x, y, w, h, COLORS["white"], COLORS["iga"], radius=0.08, lw=0.8)
    nodes = {
        "S": (x + 0.35, y + h / 2),
        "1": (x + 1.05, y + h - 0.32),
        "2": (x + 1.20, y + 0.30),
        "3": (x + 2.05, y + h - 0.30),
        "4": (x + 2.25, y + 0.34),
        "E": (x + w - 0.30, y + h / 2),
    }
    edges = [("S", "1"), ("S", "2"), ("1", "3"), ("1", "4"),
             ("2", "3"), ("2", "4"), ("3", "E"), ("4", "E")]
    for a, b in edges:
        (x1, y1), (x2, y2) = nodes[a], nodes[b]
        ax.plot([x1, x2], [y1, y2], color=COLORS["line"], lw=0.55, zorder=3)
    for label, (px, py) in nodes.items():
        fc = COLORS["gold"] if label in ("S", "E") else COLORS["iga"]
        ax.add_patch(Circle((px, py), 0.12, facecolor=fc, edgecolor=COLORS["white"], lw=0.5, zorder=5))
        _text(ax, px, py, label, size=4.5, weight="bold", color=COLORS["white"])
    _text(ax, x + w / 2, y + 0.15, "Payload- and radio-weighted graph", size=4.6,
          weight="bold", color=COLORS["iga"])


def _ga_cycle(ax, x, y, w, h):
    labels = ("Pop.", "Select", "Cross", "Mutate")
    xs = np.linspace(x + 0.42, x + w - 0.42, 4)
    for idx, (px, label) in enumerate(zip(xs, labels)):
        ax.add_patch(Circle((px, y + h / 2), 0.30, facecolor=COLORS["iga_light"],
                            edgecolor=COLORS["iga"], lw=0.8, zorder=4))
        _text(ax, px, y + h / 2, label, size=4.7, weight="bold", color=COLORS["iga"])
        if idx < 3:
            _arrow(ax, (px + 0.32, y + h / 2), (xs[idx + 1] - 0.32, y + h / 2),
                   color=COLORS["iga"], lw=0.8, mutation=5)
    _arrow(ax, (xs[-1], y + h / 2 - 0.31), (xs[0], y + h / 2 - 0.31),
           color=COLORS["iga"], lw=0.7, dashed=True,
           connection="arc3,rad=-0.18", mutation=5)


def _trajectory_icon(ax, x, y, w, h):
    _rounded_box(ax, x, y, w, h, COLORS["white"], COLORS["line"], radius=0.08, lw=0.8)
    pts = np.array([
        [x + 0.20, y + 0.24], [x + 0.50, y + 0.70], [x + 0.85, y + 0.45],
        [x + 1.18, y + 0.88], [x + 1.52, y + 0.58], [x + w - 0.20, y + 0.95],
    ])
    dense = []
    for a, b in zip(pts[:-1], pts[1:]):
        for t in np.linspace(0, 1, 12, endpoint=False):
            bend = 0.08 * np.sin(np.pi * t)
            dense.append((1 - t) * a + t * b + np.array([0, bend]))
    dense.append(pts[-1])
    dense = np.asarray(dense)
    ax.plot(dense[:, 0], dense[:, 1], color=COLORS["td3"], lw=1.5, zorder=4)
    ax.scatter(pts[:, 0], pts[:, 1], s=13, facecolor=COLORS["iga"],
               edgecolor=COLORS["white"], lw=0.4, zorder=5)
    _arrow(ax, tuple(dense[-5]), tuple(dense[-1]), color=COLORS["td3"], lw=1.2, mutation=6)


def _card(ax, x, y, w, h, title, lines, *, edge, fill=COLORS["white"],
          title_color=None, title_size=5.8, size=5.2, dashed=False):
    _rounded_box(
        ax, x, y, w, h, fill, edge, radius=0.10, lw=0.9,
        linestyle="--" if dashed else "-",
    )
    _text(
        ax, x + 0.16, y + h - 0.28, title, size=title_size, weight="bold",
        color=title_color or COLORS["ink"], ha="left",
    )
    body = "\n".join(lines)
    if body:
        _text(
            ax, x + w / 2, y + (h - 0.46) / 2 - 0.02, body,
            size=size, color=COLORS["ink"], linespacing=1.18,
        )


def build_framework_figure(output_dir: Path | str = DEFAULT_OUTPUT_DIR) -> dict[str, Path]:
    """Build the framework and return paths for SVG, PDF, and PNG outputs."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "font.size": 5.0,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "pdf.use14corefonts": False,
    })
    fig = plt.figure(figsize=(7.15, 4.15), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 9)
    ax.axis("off")

    _text(ax, 9.0, 8.62, "Proposed dual-radio-map-aware PRISM-TD3 framework",
          size=9.0, weight="bold")
    _text(ax, 9.0, 8.22,
          "Heterogeneous inspection payloads  |  Joint sequence and continuous-trajectory optimization",
          size=5.8, color=COLORS["muted"])

    panel_y, panel_h = 0.55, 7.25
    _stage_panel(ax, 0.25, panel_y, 3.10, panel_h, 1, "System inputs", COLORS["ink"])
    _stage_panel(ax, 3.62, panel_y, 4.18, panel_h, 2, "PRISM: payload + radio", COLORS["iga"])
    _stage_panel(ax, 8.07, panel_y, 6.45, panel_h, 3, "Map-aware TD3", COLORS["td3"])
    _rounded_box(
        ax, 14.80, panel_y, 2.92, panel_h, COLORS["panel"], COLORS["line"],
        radius=0.16, lw=0.8,
    )
    _text(ax, 16.26, panel_y + panel_h - 0.31, "Mission output", size=6.8,
          weight="bold", color="#9A6700")

    # Stage 1: mission inputs.
    _card(
        ax, 0.43, 6.10, 2.74, 1.12, "Inspection scenario",
        ["Fixed start/end", "Six inspection sites", "Motion and QoS limits"],
        edge=COLORS["line"], size=5.0,
    )
    _card(
        ax, 0.43, 4.80, 2.74, 0.98, "Heterogeneous payloads",
        ["Point-specific data volumes"], edge=COLORS["gold"],
        fill=COLORS["gold_light"], title_color="#7A5600", title_size=5.3, size=5.0,
    )
    _rounded_box(ax, 0.43, 2.80, 2.74, 0.92, COLORS["g2a_light"], COLORS["g2a"], radius=0.08, lw=0.8)
    _text(ax, 1.80, 3.43, "G2A outage", size=5.1, weight="bold", color=COLORS["g2a"])
    ax.add_patch(Rectangle((0.70, 2.98), 2.20, 0.28, facecolor=COLORS["g2a"], alpha=0.22, edgecolor="none"))
    _rounded_box(ax, 0.43, 1.68, 2.74, 0.92, COLORS["a2g_light"], COLORS["a2g"], radius=0.08, lw=0.8)
    _text(ax, 1.80, 2.31, "A2G uplink-SNR", size=5.1, weight="bold", color=COLORS["a2g"])
    ax.add_patch(Rectangle((0.70, 1.86), 2.20, 0.28, facecolor=COLORS["a2g"], alpha=0.30, edgecolor="none"))

    # Stage 2: PRISM.
    _card(
        ax, 3.82, 5.85, 3.78, 1.42, "Edge feature construction",
        ["flight time + payload upload", "+ G2A outage exposure"],
        edge=COLORS["iga"], fill=COLORS["iga_light"], title_color=COLORS["iga"], size=5.1,
    )
    _card(
        ax, 3.82, 4.05, 3.78, 1.30, "Permutation genetic search",
        ["Select  ->  Cross  ->  Mutate", "Exact $K!$ certificate"],
        edge=COLORS["iga"], size=5.1,
    )
    _rounded_box(ax, 3.82, 2.42, 3.78, 0.96, COLORS["iga"], COLORS["iga"], radius=0.12, lw=0.8)
    _text(ax, 5.71, 2.90, "Optimized inspection order  O*", size=5.2, weight="bold",
          color=COLORS["white"])

    # Stage 3: standard TD3 executor.
    _card(
        ax, 8.30, 6.02, 2.64, 1.15, "Environment",
        ["Move, query maps, upload", "Compute T and tau"],
        edge=COLORS["ink"], size=5.0,
    )
    _card(
        ax, 11.23, 6.02, 3.02, 1.15, "Training only",
        ["MapGuide demonstrations", "BC anchor"],
        edge=COLORS["gold"], fill=COLORS["gold_light"],
        title_color="#7A5600", size=5.0, dashed=True,
    )
    _card(
        ax, 8.30, 4.52, 2.64, 1.08, "Map-aware state",
        ["position, backlog, QoS", "A2G SNR, G2A outage"],
        edge=COLORS["g2a"], fill=COLORS["g2a_light"], title_color=COLORS["g2a"], size=5.0,
    )
    _card(
        ax, 11.23, 4.52, 3.02, 1.08, "Continuous action",
        ["relative heading offset", "flight distance"],
        edge=COLORS["a2g"], fill=COLORS["a2g_light"], title_color=COLORS["a2g"], size=5.0,
    )
    _card(
        ax, 8.30, 2.92, 2.64, 1.05, "Twin critics",
        ["Q1(s,a) and Q2(s,a)"],
        edge=COLORS["td3"], fill=COLORS["td3_light"], title_color=COLORS["td3"], size=5.0,
    )
    _rounded_box(ax, 11.23, 2.92, 3.02, 0.86, COLORS["td3"], COLORS["td3"], radius=0.14, lw=0.8)
    _text(ax, 12.74, 3.35, "Guide-free Actor (evaluation)", size=5.2, weight="bold",
          color=COLORS["white"])

    # Stage 4: mission output.
    _trajectory_icon(ax, 15.00, 5.92, 2.52, 1.18)
    _text(ax, 16.26, 5.66, "Continuous trajectory Q*", size=5.4, weight="bold",
          color=COLORS["td3"])
    _card(
        ax, 15.00, 4.53, 2.52, 0.96, "Objective",
        ["min  T + tau_total"], edge=COLORS["gold"],
        fill=COLORS["gold_light"], title_color="#7A5600", size=5.2,
    )
    _card(
        ax, 15.00, 3.27, 2.52, 0.96, "Hard QoS constraint",
        ["T_out <= T_out^max"], edge=COLORS["g2a"],
        fill=COLORS["g2a_light"], title_color=COLORS["g2a"], size=5.2,
    )
    _card(
        ax, 15.00, 1.46, 2.52, 1.45, "Mission record",
        ["T, tau_total, T_out", "flight distance"],
        edge=COLORS["line"], size=5.0,
    )

    # Main pipeline and the only secondary loop.
    _arrow(ax, (3.35, 3.58), (3.82, 3.58), color=COLORS["ink"], lw=1.3)
    _arrow(ax, (7.60, 3.58), (8.30, 3.58), color=COLORS["iga"], lw=1.3)
    _arrow(ax, (14.52, 3.58), (15.00, 3.58), color=COLORS["td3"], lw=1.3)
    _arrow(ax, (5.71, 5.85), (5.71, 5.35), color=COLORS["iga"], lw=1.0)
    _arrow(ax, (5.71, 4.05), (5.71, 3.38), color=COLORS["iga"], lw=1.0)
    _arrow(ax, (10.94, 6.02), (10.94, 5.60), color=COLORS["ink"], lw=1.0)
    _arrow(ax, (12.74, 6.02), (12.74, 5.60), color=COLORS["gold"], lw=0.9, dashed=True)
    _arrow(ax, (12.74, 4.52), (12.74, 3.78), color=COLORS["td3"], lw=1.0)
    _arrow(ax, (10.94, 2.92), (10.94, 2.35), color=COLORS["td3"], lw=0.9)
    _arrow(ax, (11.23, 2.35), (14.52, 2.35), color=COLORS["td3"], lw=0.9)
    _arrow(ax, (14.52, 2.35), (15.00, 2.35), color=COLORS["td3"], lw=0.9)

    stem = output_dir / "prism_td3_framework"
    svg_path = stem.with_suffix(".svg")
    pdf_path = stem.with_suffix(".pdf")
    png_path = stem.with_suffix(".png")
    tiff_path = stem.with_suffix(".tiff")
    metadata = {
        "Title": "Proposed dual-radio-map-aware PRISM-TD3 framework",
        "Subject": "UAV inspection sequence and trajectory optimization",
        "Creator": "Matplotlib",
    }
    fig.savefig(svg_path, format="svg", facecolor="white", metadata={"Title": metadata["Title"]})
    # Matplotlib writes trailing spaces in multi-line SVG path data. Normalize
    # them so generated artifacts remain clean in Git and reproducible reviews.
    svg_text = svg_path.read_text(encoding="utf-8")
    svg_path.write_text(
        "\n".join(line.rstrip() for line in svg_text.splitlines()) + "\n",
        encoding="utf-8",
    )
    fig.savefig(pdf_path, format="pdf", facecolor="white", metadata=metadata)
    fig.savefig(png_path, format="png", dpi=600, facecolor="white",
                metadata={"Title": metadata["Title"]})
    fig.savefig(tiff_path, format="tiff", dpi=600, facecolor="white")
    plt.close(fig)
    return {"svg": svg_path, "pdf": pdf_path, "png": png_path, "tiff": tiff_path}


if __name__ == "__main__":
    generated = build_framework_figure()
    for kind, path in generated.items():
        print(f"{kind.upper()}: {path}")
